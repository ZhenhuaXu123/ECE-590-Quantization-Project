import time
import copy
from datetime import datetime, timedelta
from pathlib import Path
import io

import torch
import torch.nn as nn

from stable_diffusion_pytorch import util
from stable_diffusion_pytorch import pipeline, model_loader

from torch.ao.quantization import QConfigMapping, QConfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

from PIL import Image
from torchvision import transforms

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure


# ============================================================
# Config (match PTQ/KLD standard)
# ============================================================
DEVICE = "cpu"
torch.backends.quantized.engine = "fbgemm"
torch.manual_seed(42)

OUTPUT_DIR = Path("outputs_qat_cpu_matched")
OUTPUT_DIR.mkdir(exist_ok=True)

# Same eval prompts as PTQ evaluation
PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "a puppy wearing a hat",
    "a red sports car on a rainy street",
]
SEED = 42
N_STEPS = 30
SAMPLER = "k_lms"

# Same calibration prompt list file you used for PTQ/KLD
CALIB_PROMPT_FILE = "calibration_prompts.txt"
with open(CALIB_PROMPT_FILE, "r", encoding="utf-8") as f:
    TRAIN_PROMPTS = [line.strip() for line in f if line.strip()]

# Make this equal to PTQ calibration steps (e.g., 10)
TRAIN_STEPS = 10

# "run more hours": increase epochs and/or MAX_TRAIN_PROMPTS
QAT_EPOCHS = 1
MAX_TRAIN_PROMPTS = len(TRAIN_PROMPTS)

LR = 1e-6

# ETA measurement
WARMUP_ITERS = 1
MEASURE_ITERS = 3


# ============================================================
# Helpers
# ============================================================
def measure_model_stats(model, name="model"):
    total_params = sum(p.numel() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = len(buffer.getvalue())
    size_mb = size_bytes / (1024 * 1024)
    print(f"[Stats] {name}: params={total_params:,}, size={size_mb:.2f} MB")
    return total_params, size_bytes


def save_images(imgs, prefix: str):
    paths = []
    for i, im in enumerate(imgs):
        p = OUTPUT_DIR / f"{prefix}_{i}.png"
        im.save(p)
        paths.append(p)
    return paths


def load_image_tensor(path: Path):
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(img).unsqueeze(0)  # (1,3,H,W)
    return t


def require_baseline_images():
    base_paths = [OUTPUT_DIR / f"base_{i}.png" for i in range(len(PROMPTS))]
    missing = [p for p in base_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Baseline images not found. Expected:\n"
            + "\n".join(str(p) for p in base_paths)
            + "\n\nYou already generated them earlier; please keep them in outputs_qat_cpu_matched/."
        )
    return base_paths


# ============================================================
# Naive MinMax QAT qconfig
# ============================================================
def get_naive_minmax_qconfig() -> QConfig:
    act_fake_quant = FakeQuantize.with_args(
        observer=MinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-128,
        quant_max=127,
    )
    w_fake_quant = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-128,
        quant_max=127,
        ch_axis=0,
    )
    return QConfig(activation=act_fake_quant, weight=w_fake_quant)


def build_qat_unet_minmax(unet_fp32: nn.Module, device: str):
    """
    Insert FakeQuant (MinMax observer) into Conv2d / Linear.
    Forward signature must match this repo: unet(latents, time_embedding, context)
    """
    unet = copy.deepcopy(unet_fp32).train().to(device)

    qconfig = get_naive_minmax_qconfig()
    qmap = QConfigMapping().set_global(None)
    qmap = qmap.set_object_type(torch.nn.Conv2d, qconfig).set_object_type(torch.nn.Linear, qconfig)

    # example inputs matching SD latent/context shapes
    example_lat = torch.randn(1, 4, 64, 64, device=device)
    example_t = util.get_time_embedding(500, torch.float32).to(device)
    example_ctx = torch.randn(1, 77, 768, device=device)

    prepared = prepare_fx(unet, qmap, example_inputs=(example_lat, example_t, example_ctx))
    return prepared


# ============================================================
# Teacher wrapper: accumulate differentiable loss inside pipeline sampling loop
# ============================================================
class QATTeacherWrapper(nn.Module):
    def __init__(self, student_qat: nn.Module, teacher_fp32: nn.Module, mse_weight: float = 1.0):
        super().__init__()
        self.student = student_qat
        self.teacher = teacher_fp32.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.mse_weight = mse_weight
        self.loss_accum = None

    def reset_loss(self):
        self.loss_accum = None

    def forward(self, latents, time_embedding, context):
        # teacher: no grad
        with torch.no_grad():
            t_out = self.teacher(latents, time_embedding, context)

        # student: requires grad
        s_out = self.student(latents, time_embedding, context)

        loss = (s_out - t_out).pow(2).mean() * self.mse_weight
        self.loss_accum = loss if self.loss_accum is None else (self.loss_accum + loss)
        return s_out


# ============================================================
# One training update = one prompt sampling (TRAIN_STEPS) + backward + step
# ============================================================
def qat_train_one_prompt(models_train, wrapper: QATTeacherWrapper, prompt: str):
    wrapper.reset_loss()

    # After patch_pipeline_for_qat.py, pipeline.generate will enable grad when diffusion is in train mode.
    _ = pipeline.generate(
        [prompt],
        models=models_train,
        seed=SEED,
        n_inference_steps=TRAIN_STEPS,
        sampler=SAMPLER,
        device=DEVICE,
    )

    if wrapper.loss_accum is None:
        raise RuntimeError(
            "loss_accum is None. It means pipeline.generate didn't call models['diffusion'] forward."
        )
    if not wrapper.loss_accum.requires_grad:
        raise RuntimeError(
            "loss does not require grad. This means pipeline.generate is still under no_grad internally.\n"
            "You MUST run: python3 patch_pipeline_for_qat.py (once) to patch stable_diffusion_pytorch/pipeline.py."
        )
    return wrapper.loss_accum


# ============================================================
# Main
# ============================================================
def main():
    print("[QAT-CPU] loading models …")
    models = model_loader.preload_models(DEVICE)

    # Reuse existing baseline images, do NOT regenerate
    base_paths = require_baseline_images()
    print("[QAT-CPU] baseline images found, skipping baseline inference.")
    for p in base_paths:
        print(f"[QAT-CPU] using baseline: {p}")

    # Build teacher + student(QAT)
    print("[QAT-CPU] building QAT UNet (matched with PTQ) …")
    teacher_fp32 = copy.deepcopy(models["diffusion"]).eval().to(DEVICE)
    student_qat = build_qat_unet_minmax(models["diffusion"], DEVICE)

    # Stats
    measure_model_stats(teacher_fp32, "UNet_FP32")
    measure_model_stats(student_qat, "UNet_QAT_Prepared")

    # Wrap diffusion
    wrapper = QATTeacherWrapper(student_qat, teacher_fp32, mse_weight=1.0)
    models_train = models.copy()
    models_train["diffusion"] = wrapper

    optimizer = torch.optim.AdamW(student_qat.parameters(), lr=LR)

    # ========================================================
    # Time estimate
    # ========================================================
    print("[QAT-CPU] estimating time …")
    torch.set_grad_enabled(True)

    # warmup
    for _ in range(WARMUP_ITERS):
        loss = qat_train_one_prompt(models_train, wrapper, TRAIN_PROMPTS[0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # measure
    t0 = time.time()
    last_loss = None
    for _ in range(MEASURE_ITERS):
        last_loss = qat_train_one_prompt(models_train, wrapper, TRAIN_PROMPTS[0])
        optimizer.zero_grad(set_to_none=True)
        last_loss.backward()
        optimizer.step()
    t1 = time.time()

    sec_per_update = (t1 - t0) / float(MEASURE_ITERS)
    n_prompts = min(MAX_TRAIN_PROMPTS, len(TRAIN_PROMPTS))
    total_updates = QAT_EPOCHS * n_prompts
    est_seconds = sec_per_update * total_updates
    eta_time = datetime.now() + timedelta(seconds=est_seconds)

    print(f"[QAT-CPU] train_steps_per_prompt = {TRAIN_STEPS}")
    print(f"[QAT-CPU] avg sec/update = {sec_per_update:.2f}")
    print(f"[QAT-CPU] total updates = {total_updates} (epochs={QAT_EPOCHS}, prompts={n_prompts})")
    print(f"[QAT-CPU] estimated total = {est_seconds/60:.1f} minutes ({est_seconds/3600:.2f} hours)")
    print(f"[QAT-CPU] ETA (local) = {eta_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if last_loss is not None:
        print(f"[QAT-CPU] (after measure) loss = {last_loss.item():.6f}")

    # ========================================================
    # QAT training loop
    # ========================================================
    print("[QAT-CPU] start QAT training …")
    for ep in range(QAT_EPOCHS):
        for i in range(n_prompts):
            prompt = TRAIN_PROMPTS[i]
            loss = qat_train_one_prompt(models_train, wrapper, prompt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0 or i == 0:
                print(f"[QAT-CPU] epoch {ep} | update {i+1}/{n_prompts} | loss={loss.item():.6f}")

    # ========================================================
    # Convert to INT8 & save
    # ========================================================
    student_qat.eval()
    int8_unet = convert_fx(student_qat).eval()
    ckpt_path = Path("diffusion_int8_qat_minmax.pt")
    torch.save(int8_unet.state_dict(), ckpt_path)
    print(f"[QAT-CPU] saved {ckpt_path}")

    # ========================================================
    # Inference using QAT INT8 UNet (same eval settings as PTQ)
    # ========================================================
    print("[QAT-CPU] running QAT inference …")
    torch.set_grad_enabled(False)

    models_qat = models.copy()
    models_qat["diffusion"] = int8_unet

    t0 = time.time()
    imgs_qat = pipeline.generate(
        PROMPTS,
        models=models_qat,
        seed=SEED,
        n_inference_steps=N_STEPS,
        sampler=SAMPLER,
        device=DEVICE,
    )
    t1 = time.time()
    print(f"[QAT-CPU] avg inference time per prompt: {(t1 - t0)/len(PROMPTS):.2f}s")

    qat_paths = save_images(imgs_qat, "qat_minmax")

    # ========================================================
    # Metrics: LPIPS / SSIM (QAT vs baseline images on disk)
    # ========================================================
    print("[QAT-CPU] computing metrics …")
    base_tensors = torch.cat([load_image_tensor(p) for p in base_paths], dim=0).to(DEVICE)
    qat_tensors = torch.cat([load_image_tensor(p) for p in qat_paths], dim=0).to(DEVICE)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    lpips_val = lpips_metric(qat_tensors, base_tensors)
    ssim_val = ssim_metric(qat_tensors, base_tensors)

    print(f"LPIPS (QAT vs Base): {lpips_val.item():.4f}")
    print(f"SSIM  (QAT vs Base): {ssim_val.item():.4f}")
    print("[QAT-CPU] done.")


if __name__ == "__main__":
    main()
