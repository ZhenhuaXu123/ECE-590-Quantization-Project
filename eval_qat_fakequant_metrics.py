import os
import time
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from stable_diffusion_pytorch import util
from stable_diffusion_pytorch import pipeline, model_loader

from torch.ao.quantization import QConfigMapping, QConfig
from torch.ao.quantization.quantize_fx import prepare_fx
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
import torch.ao.quantization as aoq

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


# ------------------------- helpers -------------------------
def read_prompts(path: Path) -> List[str]:
    prompts = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        prompts.append(s)
    return prompts


def load_image_tensor_01(path: Path) -> torch.Tensor:
    """Return float tensor in [0,1], shape (1,3,H,W)."""
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t


def save_images(imgs: List[Image.Image], out_dir: Path, prefix: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, im in enumerate(imgs):
        p = out_dir / f"{prefix}_{i}.png"
        im.save(p)
        paths.append(p)
    return paths


def get_naive_minmax_qconfig() -> QConfig:
    # activation: per-tensor symmetric int8 MinMax
    act_fake_quant = FakeQuantize.with_args(
        observer=MinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-128,
        quant_max=127,
    )
    # weight: per-channel symmetric int8 MinMax
    w_fake_quant = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-128,
        quant_max=127,
        ch_axis=0,
    )
    return QConfig(activation=act_fake_quant, weight=w_fake_quant)


def build_qat_unet_minmax_shell(unet_fp32: nn.Module, device: str) -> nn.Module:
    """
    Build a QAT(FakeQuant) "shell" matching qat_naive_diffusion.py:
      - insert FakeQuant into Conv2d/Linear via FX prepare
      - DO NOT convert_fx (so weights remain fp32-ish, state_dict is large)
    """
    unet = unet_fp32.eval().to(device)  # start from eval; we will control observer/fq explicitly

    qconfig = get_naive_minmax_qconfig()
    qmap = QConfigMapping().set_global(None)
    qmap = qmap.set_object_type(torch.nn.Conv2d, qconfig).set_object_type(torch.nn.Linear, qconfig)

    # example inputs matching this repo's UNet forward signature:
    # unet(latents, time_embedding, context)
    example_lat = torch.randn(1, 4, 64, 64, device=device)
    example_t = util.get_time_embedding(500, torch.float32).to(device)   # (1,320) in this repo
    example_ctx = torch.randn(1, 77, 768, device=device)

    prepared = prepare_fx(unet, qmap, example_inputs=(example_lat, example_t, example_ctx))
    return prepared


def freeze_observer_enable_fakequant(m: nn.Module):
    # typical QAT-eval setting: freeze ranges, keep fake-quant on
    m.apply(aoq.disable_observer)
    m.apply(aoq.enable_fake_quant)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu"])
    ap.add_argument("--engine", default="fbgemm", choices=["fbgemm", "qnnpack"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--qat_ckpt", type=str, default="diffusion_int8_qat_minmax.pt",
                    help="3.20GB QAT(FakeQuant) checkpoint you saved during QAT")
    ap.add_argument("--baseline_dir", type=str, default="outputs_qat_cpu_matched",
                    help="Directory that contains base_0.png, base_1.png, ...")
    ap.add_argument("--prompts_file", type=str, default="calibration_prompts.txt",
                    help="One prompt per line; number of prompts should match number of base_*.png")
    ap.add_argument("--sampler", type=str, default="k_lms")
    ap.add_argument("--eval_steps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="outputs_qat_fakequant_eval")
    args = ap.parse_args()

    # --- runtime knobs ---
    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = args.engine
    torch.manual_seed(args.seed)

    device = args.device
    out_dir = Path(args.out_dir)
    baseline_dir = Path(args.baseline_dir)
    prompts_path = Path(args.prompts_file)

    # --- find baseline images ---
    base_paths = sorted(baseline_dir.glob("base_*.png"), key=lambda p: p.name)
    if len(base_paths) == 0:
        raise FileNotFoundError(f"No baseline images found in {baseline_dir} (expect base_*.png)")

    prompts = read_prompts(prompts_path)
    if len(prompts) != len(base_paths):
        raise RuntimeError(
            f"prompts ({len(prompts)}) must match baseline images ({len(base_paths)}). "
            f"Fix by editing {prompts_path} or regenerating base_*.png."
        )

    print(f"[Eval-QAT-FakeQuant] DEVICE={device}, engine={args.engine}, threads={args.threads}")
    print(f"[Eval-QAT-FakeQuant] baseline images = {len(base_paths)} from {baseline_dir}")
    print(f"[Eval-QAT-FakeQuant] prompts        = {len(prompts)} from {prompts_path}")
    print(f"[Eval-QAT-FakeQuant] QAT ckpt        = {args.qat_ckpt}")

    # --- load fp32 pipeline/models ---
    print("[Eval-QAT-FakeQuant] loading models ...")
    models = model_loader.preload_models(device)
    unet_fp32 = models["diffusion"]

    # --- build QAT shell & load 3.2GB ckpt ---
    print("[Eval-QAT-FakeQuant] building QAT(FakeQuant) shell (MinMax, matches training) ...")
    unet_qat = build_qat_unet_minmax_shell(unet_fp32, device)

    print("[Eval-QAT-FakeQuant] loading QAT state_dict (fp32-ish) ...")
    state = torch.load(args.qat_ckpt, map_location=device)
    missing, unexpected = unet_qat.load_state_dict(state, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  (first 20 missing)   :", missing[:20])
    if len(unexpected) > 0:
        print("  (first 20 unexpected):", unexpected[:20])

    unet_qat.eval()
    freeze_observer_enable_fakequant(unet_qat)

    models_qat = models.copy()
    models_qat["diffusion"] = unet_qat

    # --- run QAT(FakeQuant) inference ---
    print("[Eval-QAT-FakeQuant] running fakequant inference ...")
    t0 = time.time()
    imgs_qat = pipeline.generate(
        prompts,
        models=models_qat,
        seed=args.seed,
        n_inference_steps=args.eval_steps,
        sampler=args.sampler,
        device=device,
    )
    t1 = time.time()
    avg_sec = (t1 - t0) / len(prompts)
    print(f"[Eval-QAT-FakeQuant] avg inference time per prompt: {avg_sec:.2f}s")

    qat_paths = save_images(imgs_qat, out_dir, "qat_fakequant")
    print(f"[Eval-QAT-FakeQuant] saved {len(qat_paths)} images to {out_dir}")

    # --- metrics: LPIPS/SSIM per-sample mean±std; FID over sets ---
    print("[Eval-QAT-FakeQuant] computing metrics (LPIPS/SSIM/FID) ...")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    lpips_vals = []
    ssim_vals = []

    with torch.no_grad():
        for bp, qp in zip(base_paths, qat_paths):
            b = load_image_tensor_01(bp).to(device)   # (1,3,H,W) float in [0,1]
            q = load_image_tensor_01(qp).to(device)

            # LPIPS expects [-1,1] in many conventions; torchmetrics LPIPS accepts [0,1] too,
            # but to be consistent with common practice we map to [-1,1].
            b_lp = b * 2 - 1
            q_lp = q * 2 - 1
            lp = lpips(q_lp, b_lp).item()
            ss = ssim(q, b).item()

            lpips_vals.append(lp)
            ssim_vals.append(ss)

            # FID expects uint8 in [0,255], shape (N,3,H,W)
            b_u8 = (b.clamp(0, 1) * 255.0).to(torch.uint8)
            q_u8 = (q.clamp(0, 1) * 255.0).to(torch.uint8)
            fid.update(b_u8, real=True)
            fid.update(q_u8, real=False)

    import math
    def mean_std(xs):
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
        return m, math.sqrt(v)

    lp_m, lp_s = mean_std(lpips_vals)
    ss_m, ss_s = mean_std(ssim_vals)
    fid_val = fid.compute().item()

    print("========== QAT(FakeQuant) vs FP32 Baseline ==========")
    print(f"LPIPS (mean±std): {lp_m:.4f} ± {lp_s:.4f}")
    print(f"SSIM  (mean±std): {ss_m:.4f} ± {ss_s:.4f}")
    print(f"FID            : {fid_val:.4f}")
    print("=====================================================")

    report = out_dir / "qat_fakequant_report.txt"
    with open(report, "w") as f:
        f.write("QAT FakeQuant Evaluation Report\n")
        f.write(f"- qat_ckpt      : {args.qat_ckpt}\n")
        f.write(f"- baseline_dir  : {baseline_dir}\n")
        f.write(f"- prompts_file  : {prompts_path}\n")
        f.write(f"- eval_steps    : {args.eval_steps}\n")
        f.write(f"- sampler       : {args.sampler}\n")
        f.write(f"- avg_sec/prompt: {avg_sec:.2f}\n\n")
        f.write(f"LPIPS (mean±std): {lp_m:.4f} ± {lp_s:.4f}\n")
        f.write(f"SSIM  (mean±std): {ss_m:.4f} ± {ss_s:.4f}\n")
        f.write(f"FID            : {fid_val:.4f}\n")
    print(f"[Eval-QAT-FakeQuant] report saved to {report}")


if __name__ == "__main__":
    main()


# python3 eval_qat_fakequant_metrics.py \
#   --qat_ckpt diffusion_int8_qat_minmax.pt \
#   --baseline_dir outputs_qat_cpu_matched \
#   --prompts_file calibration_prompts.txt \
#   --eval_steps 30 \
#   --sampler k_lms \
#   --threads 8 \
#   --out_dir outputs_qat_fakequant_eval