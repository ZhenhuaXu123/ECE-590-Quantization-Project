import os
import time
import torch
import copy
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F

from stable_diffusion_pytorch import util
from stable_diffusion_pytorch import pipeline, model_loader

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.observer import ObserverBase, PerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure


# ---------------- 配置区域 ----------------
DEVICE = "cpu"
torch.backends.quantized.engine = "fbgemm"
torch.set_grad_enabled(False)

OUTPUT_DIR = Path("outputs_kld")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "a puppy wearing a hat",
    "a red sports car on a rainy street",
]
SEED = 42
N_STEPS = 30
SAMPLER = "k_lms"
CFG_SCALE = 7.5

# ✅ 现在：校准就用上面这 3 个 prompt
CALIBRATE_PROMPTS = PROMPTS
CALIB_STEPS = 10                 # 每条 prompt 校准跑几步（只影响校准速度）


# ---------------- 工具函数 ----------------
def save_images(imgs, prefix: str):
    paths = []
    for i, im in enumerate(imgs):
        p = OUTPUT_DIR / f"{prefix}_{i}.png"
        im.save(p)
        paths.append(p)
    return paths


def load_image_tensor(path: Path):
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor


def measure_model_stats(model, name="model"):
    import io
    total_params = sum(p.numel() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = len(buffer.getvalue())
    size_mb = size_bytes / (1024 * 1024)
    print(f"[Stats] {name}: params={total_params:,}, size={size_mb:.2f} MB")
    return total_params, size_bytes


# ---------------- KLD / Entropy 校准 ----------------
@torch.no_grad()
def kld_threshold_from_hist(hist: torch.Tensor, absmax: float, num_quant=128):
    """
    TensorRT/TPU-MLIR 风格 KL-divergence 阈值搜索。
    hist: |x| 直方图 (bins,)
    absmax: 直方图上界
    num_quant: int8 对称量化 level 数=128
    返回最佳 threshold T
    """
    hist = hist.float()
    bins = hist.numel()
    eps = 1e-8

    best_kl = None
    best_i = bins

    for i in range(num_quant, bins + 1, num_quant):
        fp_hist = hist.clone()
        outliers = fp_hist[i:].sum()
        fp32_hist = fp_hist[:i]
        fp32_hist[-1] += outliers

        P = fp32_hist / (fp32_hist.sum() + eps)

        step = i // num_quant
        q_hist = torch.zeros(num_quant)

        for q in range(num_quant):
            start = q * step
            end = min((q + 1) * step, i)
            if end > start:
                q_hist[q] = fp32_hist[start:end].sum()

        if q_hist.sum() == 0:
            continue

        q_hist = q_hist / (q_hist.sum() + eps)

        Q = torch.zeros(i)
        for q in range(num_quant):
            start = q * step
            end = min((q + 1) * step, i)
            if end > start and q_hist[q] > 0:
                Q[start:end] = q_hist[q] / (end - start)

        Q = Q / (Q.sum() + eps)

        P_safe = P.clamp_min(eps)
        Q_safe = Q.clamp_min(eps)
        kl = (P_safe * (P_safe / Q_safe).log()).sum().item()

        if best_kl is None or kl < best_kl:
            best_kl = kl
            best_i = i

    threshold = (best_i + 0.5) * float(absmax) / bins
    return float(threshold)


class KLDObserver(ObserverBase):
    """
    激活 KLD observer (对称 int8)：
    forward: 累积 |x| histogram
    calculate_qparams: KLD 搜索 threshold -> scale = T/127
    """
    def __init__(
        self,
        bins=2048,
        num_quant=128,
        dtype=torch.qint8,
        qscheme=None,
        quant_min=None,
        quant_max=None,
        reduce_range=None,
        **kwargs
    ):
        # 旧版 torch 的 ObserverBase 不认 qscheme/quant_min/...，所以只传 dtype
        super().__init__(dtype=dtype)

        if qscheme is None:
            qscheme = torch.per_tensor_symmetric
        self.qscheme = qscheme

        # FakeQuantize 期望 observer 上有 quant_min / quant_max
        self.quant_min = -128 if quant_min is None else int(quant_min)
        self.quant_max = 127 if quant_max is None else int(quant_max)
        self.reduce_range = False if reduce_range is None else bool(reduce_range)

        self.bins = bins
        self.num_quant = num_quant
        self.dtype = dtype

        self.register_buffer("absmax", torch.tensor(0.0))
        self.register_buffer("hist", torch.zeros(bins))

    def forward(self, x):
        x = x.detach().float()
        if x.numel() == 0:
            return x
        abs_x = x.abs()
        cur_absmax = abs_x.max()
        if cur_absmax > self.absmax:
            self.absmax = cur_absmax

        if self.absmax > 0:
            h = torch.histc(
                abs_x.cpu(),
                bins=self.bins,
                min=0.0,
                max=float(self.absmax)
            )
            self.hist += h
        return x

    def calculate_qparams(self):
        if self.absmax == 0 or self.hist.sum() == 0:
            scale = torch.tensor(1.0)
            zero_point = torch.tensor(0)
            return scale, zero_point

        T = kld_threshold_from_hist(
            self.hist,
            float(self.absmax),
            num_quant=self.num_quant
        )
        scale = max(T / 127.0, 1e-8)
        scale = torch.tensor(scale)
        zero_point = torch.tensor(0)
        return scale, zero_point


# ---------------- build KLD-PTQ 模型 ----------------
def build_ptq_model_kld(unet_fp32, device):
    unet = copy.deepcopy(unet_fp32).eval().to(device)

    # activation 用 KLDObserver
    act_fake_quant = FakeQuantize.with_args(
        observer=KLDObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        reduce_range=False,
        bins=2048,
        num_quant=128,
    )

    # weight 用 per-channel minmax（默认）
    w_fake_quant = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-128,
        quant_max=127,
        ch_axis=0,
    )

    custom_qconfig = torch.ao.quantization.QConfig(
        activation=act_fake_quant,
        weight=w_fake_quant
    )

    qmap = QConfigMapping().set_global(None)
    qmap = qmap.set_object_type(torch.nn.Conv2d, custom_qconfig) \
               .set_object_type(torch.nn.Linear, custom_qconfig)

    # example inputs：保持你之前朴素 PTQ 的形式
    example_lat = torch.randn(
        1, 4,
        unet_fp32.latent_h, unet_fp32.latent_w,
        device=device
    ) if hasattr(unet_fp32, "latent_h") else torch.randn(1, 4, 64, 64, device=device)

    example_t = util.get_time_embedding(500, torch.float32).to(device)
    example_ctx = torch.randn(1, 77, 768, device=device)

    prepared = prepare_fx(
        unet, qmap,
        example_inputs=(example_lat, example_t, example_ctx)
    )

    # 校准阶段只用 pipeline.generate
    with torch.inference_mode():
        for idx, prompt in enumerate(CALIBRATE_PROMPTS):
            _ = pipeline.generate(
                [prompt],
                models={"diffusion": prepared},
                seed=SEED,
                n_inference_steps=CALIB_STEPS,
                sampler=SAMPLER,
                device=device,
            )
            print(f"[KLD-PTQ] Calibrated {idx+1}/{len(CALIBRATE_PROMPTS)} prompts")

    quant_ref = convert_fx(prepared)
    return quant_ref


# ---------------- 导出压缩 int8 checkpoint ----------------
def export_int8_checkpoint(model, path: str):
    """
    把模型的 state_dict 压成 int8 + scale：
    - 对所有 float 权重做对称 per-tensor 量化，保存 int8 权重 + 一个 float32 scale
    - 非 float 参数（比如缓冲区、BN 统计等）原样保存
    这个文件不能直接 load_state_dict，到时候需要写一个反量化 loader。
    """
    state = model.state_dict()
    int8_weights = {}
    scales = {}
    other = {}

    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.numel() > 0:
            w = tensor.cpu()
            # 对称 per-tensor quant
            s = w.abs().max() / 127.0
            s = float(s.item()) if s > 0 else 1.0
            s = max(s, 1e-8)
            w_int8 = torch.clamp(torch.round(w / s), -127, 127).to(torch.int8)

            int8_weights[name] = w_int8
            scales[name] = torch.tensor(s, dtype=torch.float32)
        else:
            # 比如 bias、位置编码、缓冲区之类：原样保存（数量相对很少）
            other[name] = tensor.cpu()

    package = {
        "int8_weights": int8_weights,
        "scales": scales,
        "other_tensors": other,
    }
    torch.save(package, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[Export] Saved compressed int8 checkpoint to {path}, size={size_mb:.2f} MB")


# ---------------- main ----------------
def main():
    torch.manual_seed(SEED)
    models = model_loader.preload_models(DEVICE)

    # Baseline
    print("Running baseline inference …")
    imgs_base = pipeline.generate(
        PROMPTS, models=models, seed=SEED,
        n_inference_steps=N_STEPS, sampler=SAMPLER, device=DEVICE
    )
    base_paths = save_images(imgs_base, "base")

    unet_base = models["diffusion"]
    params_base, size_base = measure_model_stats(unet_base, "UNet_FP32")
    print(f"Baseline UNet params: {params_base}, size {size_base/1e6:.2f} MB")

    # Build KLD-PTQ model
    print("Building KLD-PTQ model …")
    unet_q = build_ptq_model_kld(models["diffusion"], DEVICE)

    models_ptq = models.copy()
    models_ptq["diffusion"] = unet_q

    params_ptq, size_ptq = measure_model_stats(unet_q, "UNet_KLD_PTQ")
    print(f"KLD-PTQ UNet params: {params_ptq}, size {size_ptq/1e6:.2f} MB")

    # ✅ 导出压缩后的 INT8 checkpoint（大约是原来的 1/4 大小）
    export_int8_checkpoint(unet_q, "diffusion_int8_kld_compressed.pt")

    # KLD-PTQ inference（用 quant_ref 直接跑）
    print("Running KLD-PTQ inference …")
    t0 = time.time()
    imgs_ptq = pipeline.generate(
        PROMPTS, models=models_ptq, seed=SEED,
        n_inference_steps=N_STEPS, sampler=SAMPLER, device=DEVICE
    )
    t1 = time.time()
    ptq_time = (t1 - t0) / len(PROMPTS)
    print(f"KLD-PTQ avg time per prompt: {ptq_time:.2f}s")
    ptq_paths = save_images(imgs_ptq, "ptq_kld")

    # Metrics
    print("Computing metrics …")
    base_tensors = torch.cat([load_image_tensor(p) for p in base_paths], dim=0).to(DEVICE)
    ptq_tensors = torch.cat([load_image_tensor(p) for p in ptq_paths], dim=0).to(DEVICE)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)
    lpips_val = lpips_metric(ptq_tensors, base_tensors)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    ssim_val = ssim_metric(ptq_tensors, base_tensors)

    print(f"LPIPS (KLD-PTQ vs Base): {lpips_val.item():.4f}")
    print(f"SSIM  (KLD-PTQ vs Base): {ssim_val.item():.4f}")

    report = OUTPUT_DIR / "report_ptq_kld.md"
    with open(report, "w") as f:
        f.write("# KLD-PTQ Experiment Report\n\n")
        f.write(f"- Baseline UNet params: {params_base}, size: {size_base/1e6:.2f} MB\n")
        f.write(f"- KLD-PTQ  UNet params: {params_ptq}, size: {size_ptq/1e6:.2f} MB\n")
        f.write(f"- KLD-PTQ avg inference time per prompt: {ptq_time:.2f} s\n\n")
        f.write("## Metrics (KLD-PTQ vs Baseline)\n")
        f.write(f"- LPIPS: {lpips_val.item():.4f}\n")
        f.write(f"- SSIM : {ssim_val.item():.4f}\n")

    print(f"Report saved to {report}")


if __name__ == "__main__":
    main()
