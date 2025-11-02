import os
import time
import torch
import copy
import gc
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from stable_diffusion_pytorch import util

# 需要安装：torchmetrics  +  piq（或用 torchmetrics 本身支持的 LPIPS）
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

from stable_diffusion_pytorch import pipeline, model_loader
from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# ——————————— 配置区域 ———————————
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "a puppy wearing a hat",
    "a red sports car on a rainy street",
]
SEED = 42
N_STEPS = 30
SAMPLER = "k_lms"
CFG_SCALE = 7.5  # 如果你的 pipeline 支持的话

# 校准用 prompt（可以与评测 prompt 一致或略不同）
CALIBRATE_PROMPTS = PROMPTS

# ——————————— 函数定义 ———————————
def save_images(imgs, prefix: str):
    for i, im in enumerate(imgs):
        p = OUTPUT_DIR / f"{prefix}_{i}.png"
        im.save(p)
    return [OUTPUT_DIR / f"{prefix}_{i}.png" for i in range(len(imgs))]
# def save_images(imgs, prefix: str):
# 定义一个保存图片的工具函数。imgs 是可迭代的图像对象列表（通常是 PIL.Image.Image），prefix 是文件名前缀字符串。
# for i, im in enumerate(imgs):
# 枚举每张图像和它的索引 i，准备逐张保存。
# p = OUTPUT_DIR / f"{prefix}_{i}.png"
# 用 pathlib 的路径拼接语法生成目标文件路径。要求全局存在 OUTPUT_DIR: Path，否则会报错。最终文件名形如 dogs_0.png、dogs_1.png。
# im.save(p)
# 调用 PIL 的 Image.save 把当前图像写盘。默认 PNG 编码。
# return [OUTPUT_DIR / f"{prefix}_{i}.png" for i in range(len(imgs))]
# 返回刚才保存的所有文件路径列表，便于后续记录或打印。注意：这里再次根据 len(imgs) 拼路径，而不是使用循环里真实写出的 p 列表；如果中途某张保存失败，这里仍会“返回”不存在的路径——严格来说可以改进为在循环中收集成功保存的路径。

def load_image_tensor(path: Path):
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor  # shape (1,3,H,W), float in [0,1]

# def load_image_tensor(path: Path):
# 定义一个把磁盘图片读成 模型可用张量 的函数。path 是 Path 类型。
# img = Image.open(path).convert("RGB")
# 用 PIL 打开图像，并强制转为 3 通道 RGB（避免灰度/带 alpha 的不一致）。
# tensor = transforms.ToTensor()(img).unsqueeze(0)
# 用 torchvision.transforms.ToTensor() 把 PIL 图像转换为 torch.FloatTensor，范围 [0,1]，形状 (3,H,W)；随后 unsqueeze(0) 在最前面加上 batch 维，得到 (1,3,H,W)。
# return tensor # shape (1,3,H,W), float in [0,1]
# 返回 4D 张量，浮点（torch.float32），值域 [0,1]。注释说明了形状与范围。
# 小坑/建议
# 这个函数没有做 resize/中心裁剪/归一化（比如减均值除方差），如果下游模型需要特定预处理，要在外面补上。
# 若图片是 CMYK 等色彩空间，.convert("RGB") 也能规整到一致的 3 通道。

def build_ptq_model(unet_fp32, device):
    unet = copy.deepcopy(unet_fp32).eval().to(device)
    # 只量化 Conv2d 和 Linear
    qmap = QConfigMapping().set_global(None) #先把“默认量化”关掉
    default = get_default_qconfig_mapping("x86") #拿一份“后端推荐的量化配置”这套 x86 默认 qconfig 的含义就是：激活用 8-bit（quint8，非对称），权重用 8-bit（qint8，对称、通常 per-channel）。所以它就是 INT8 PTQ。
    qmap = qmap.set_object_type(torch.nn.Conv2d, default.global_qconfig) \
               .set_object_type(torch.nn.Linear, default.global_qconfig)
    #模型里所有的 Conv2d / Linear 模块都会被量化（按 default 的策略）；其他类型全部不量化
    # Prepare
    example_lat = torch.randn(1,4,unet_fp32.latent_h, unet_fp32.latent_w, device=device) if hasattr(unet_fp32, "latent_h") else torch.randn(1,4,64,64,device=device)
    example_t = util.get_time_embedding(500, torch.float32).to(device)
    example_ctx = torch.randn(1,77,768, device=device)
    prepared = prepare_fx(unet, qmap, example_inputs=(example_lat, example_t, example_ctx)) #相当于构造三个样例告诉fx U-Net 前向到底接收什么形状/类型的张量、哪些分支会走到、哪些层需要插观察器
    # 校准
    with torch.inference_mode():
        for prompt in CALIBRATE_PROMPTS: #校准阶段跑的就是 “插入了 observer 的原模型（prepared）”
            _ = pipeline.generate([prompt], models={"diffusion": prepared}, seed=SEED, n_inference_steps=5, sampler=SAMPLER, device=device)
            # 运行过程中，每一层的prepared中的 observer 被喂入真实激活数据，min/max 被更新
    # Convert
    quant_ref = convert_fx(prepared)
    return quant_ref

#     def build_ptq_model(unet_fp32, device):
# 定义一个**基于 FX 图模式的 PTQ（后训练量化）**流程，输入是一个浮点 UNet 模型 unet_fp32，以及放置模型/样例输入的 device（"cpu" 或 "cuda"）。
# unet = copy.deepcopy(unet_fp32).eval().to(device)
# deepcopy：避免原模型被量化改写（FX 准备/转换会插入观测器/替换模块）。
# .eval()：切到评估模式，量化准备/校准必须在 eval，否则像 Dropout/BatchNorm 行为会扰动观测统计。
# .to(device)：把副本移动到指定设备。> 注意：PyTorch 传统 int8 量化的算子主要在 CPU（fbgemm/qnnpack），如果 device="cuda"，后面真正的量化内核未必可用，常见做法是 CPU 上量化/推理。
# # 只量化 Conv2d 和 Linear
# 仅对卷积、全连接做量化（U-Net 里其它模块如 GroupNorm/SiLU 通常保持浮点）。
# qmap = QConfigMapping().set_global(None)
# 新建一个量化配置映射（FX API），并把全局默认设为 None（即默认不量化）。这样可以精确控制只量化哪些类型。
# default = get_default_qconfig_mapping("x86")
# 取一个针对 "x86" 平台的默认 qconfig 映射（底层使用 fbgemm）。这通常等价于：激活采用 per-tensor 对称/非对称 min-max 观察器，权重采用 per-channel 对称 int8（具体依据 PyTorch 版本）。
# 需要 torch.backends.quantized.engine = "fbgemm"（很多版本会自动随 x86 设置好）。
# qmap = qmap.set_object_type(torch.nn.Conv2d, default.global_qconfig) \
# .set_object_type(...) 为特定模块类型指定 qconfig。这里给 Conv2d 绑定 default.global_qconfig（来自上一行的默认配置）。反斜杠换行。
# .set_object_type(torch.nn.Linear, default.global_qconfig)
# 同理，给 Linear 绑定相同 qconfig。到此为止，只有 Conv2d/Linear 会被量化，其他类型保持浮点。
# # Prepare
# 进入 FX 量化的“准备”阶段：插入观察器、记录标定统计所需的信息。这一步需要样例输入来跟踪/捕获图。
# example_lat = torch.randn(1,4,unet_fp32.latent_h, unet_fp32.latent_w, device=device) if hasattr(unet_fp32, "latent_h") else torch.randn(1,4,64,64,device=device)
# 构造样例的潜空间张量（diffusion U-Net 的输入之一，通常是 (N,C,H,W)，这里默认通道数 4 对应 Stable Diffusion 的 latent 通道）：
# 如果 unet_fp32 带有 latent_h/latent_w 属性，就按该尺寸造随机张量；
# 否则退化到 64×64。
# 形状 (1,4,H,W)，数据分布标准正态。
# 这只用于追踪图与插观察器，不是语义正确的校准数据；真正的校准在下面会跑简短推理。
# example_t = torch.tensor([500.], device=device)
# 构造样例的**时间步（timestep）**输入。扩散模型里 U-Net 通常接收一个标量/张量表示扩散步数。这里给一个浮点标量 500.（形状 (1,)）。
# 具体 dtype/形状取决于你的 U-Net 实现，有些是 long 步数或 (1,)/() 标量。若不匹配会在 prepare_fx 报错。
# example_ctx = torch.randn(1,77,768, device=device)
# 构造样例的条件上下文（例如 CLIP 文本嵌入）。(batch=1, seq=77, dim=768) 与 Stable Diffusion v1.* 的文本编码维度一致。
# 如果用的不是 SD v1.*，这个维度可能不同（如 SDXL 文本编码维度变化）。这只用于追踪。
# prepared = prepare_fx(unet, qmap, example_inputs=(example_lat, example_t, example_ctx))
# FX 准备：
# 用 example_inputs 跑一遍（symbolic trace + 运行）以建立 FX Graph；
# 按 qmap 在可量化节点处插入观测器模块（observer/fake-quant）。
# 返回的 prepared 仍是浮点执行，但带有统计器用于收集激活/权重范围。
# # 校准
# 下一步通过实际的短推理让 observer 看到真实分布，从而记录 min/max 或直方图等统计。
# with torch.inference_mode():
# 在无梯度上下文中推理，更快更省内存，也避免污染缓冲区。
# for prompt in CALIBRATE_PROMPTS:
# 遍历若干用于校准的文本提示（关键：覆盖常见分布，越多样越好）。需要全局存在 CALIBRATE_PROMPTS 列表。
# _ = pipeline.generate([prompt], models={"unet": prepared}, seed=SEED, n_inference_steps=5, sampler=SAMPLER, device=device)
# 调用你项目里的 pipeline.generate（自定义接口）进行短程采样：
# models={"unet": prepared}：把带 observer 的 UNet 注入到生成管线里，其它子模块（VAE、文本编码器等）仍用默认版本；
# seed=SEED 固定随机性；
# n_inference_steps=5 用极少的扩散步做快速推理，目的是让观测器见到真实张量分布；
# sampler=SAMPLER 选定采样器（PLMS/ DDIM 等）；
# device=device 指定运行设备。
# 这几次推理结束后，prepared 内的 observer 就收集到了权重/激活的范围统计。
# # Convert
# 准备好统计后，进入转换阶段：把“带观察器的浮点模型”变成“量化参考或量化内核版本”。
# quant_ref = convert_fx(prepared)
# 将 prepared 转成量化模型：
# 在较新的 PyTorch（2.x）里，convert_fx 常返回参考量化模型（reference model）：在关键张量处插入 Quantize/DeQuantize 节点，核心算子仍可能是浮点实现（便于跨后端对齐/导出）；
# 在老一些的 1.13/2.0 配置下，也可能直接替换为 nn.quantized 的 Conv/Linear（fbgemm 内核，CPU int8 真算）。
# 实际效果取决于 PyTorch 版本 & 后端设置。如果你想得到真正 int8 内核并在 CPU 上跑，加上：
# import torch
# torch.backends.quantized.engine = "fbgemm"  # x86
# unet_cpu = unet.to("cpu")                   # 量化内核通常只支持 CPU
# 并确保 prepare/convert 都在 CPU 上进行。
# 若你看到输出模块是 QuantizedConv2d/QuantizedLinear，那就是内核替换成功；若是大量 quantize_per_tensor/dequantize 节点，说明是 reference graph。
# return quant_ref
# 返回量化（或参考量化）后的 UNet，用于后续推理或对比评测。
# 这段 PTQ 流程的关键点/常见问题
# 设备选择：传统 int8 量化主要在 CPU 后端（fbgemm/qnnpack），把 device="cuda" 传给 prepare_fx/convert_fx 经常不会得到真正的 int8 内核；建议在 CPU 上量化与推理，或转用 PT2E/weight-only CUDA 方案。
# 示例输入维度：example_ctx=(1,77,768) 与 SD v1.* 对齐；如果你换了文本编码器或用了 SDXL，需要改维度，否则 prepare_fx 失败。
# 校准数据：n_inference_steps=5 只是为了快；更稳妥的做法是用 20～50 步、更多样的 prompts，以避免量化失真。
# 只量化 Conv/Linear：U-Net 中的大头计算在这两类算子上，这样最省心也最稳。进一步量化激活函数前后的张量需更细化的 qconfig 和校准策略。
# 模块融合（fuse）：经典 CNN 会在量化前把 Conv+BN+ReLU 融合；U-Net 常用 GroupNorm/SiLU，可融合空间有限。
# 版本差异：PyTorch 1.13 的 FX 量化和 2.x 的 PT2E（prepare_pt2e/convert_pt2e）在 API 和产物上有差异；你的代码是FX 旧路径，能跑就继续，但想上 CUDA/int8 或更优内核时建议迁移 PT2E。

def measure_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    return total_params, size_bytes

def main():
    torch.manual_seed(SEED)
    models = model_loader.preload_models(DEVICE)
    # Baseline
    print("Running baseline inference …")
    imgs_base = pipeline.generate(PROMPTS, models=models, seed=SEED, n_inference_steps=N_STEPS, sampler=SAMPLER, device=DEVICE)
    base_paths = save_images(imgs_base, "base")
    # Record stats baseline
    unet_base = models["diffusion"]
    params_base, size_base = measure_model_stats(unet_base)
    print(f"Baseline UNet params: {params_base}, size {size_base/1e6:.2f} MB")

    # Build PTQ model
    print("Building PTQ model …")
    unet_q = build_ptq_model(models["diffusion"], DEVICE)
    models_ptq = models.copy()
    models_ptq["diffusion"] = unet_q
    params_ptq, size_ptq = measure_model_stats(unet_q)
    print(f"PTQ UNet params: {params_ptq}, size {size_ptq/1e6:.2f} MB")

    # PTQ inference & time
    print("Running PTQ inference …")
    t0 = time.time()
    imgs_ptq = pipeline.generate(PROMPTS, models=models_ptq, seed=SEED, n_inference_steps=N_STEPS, sampler=SAMPLER, device=DEVICE)
    t1 = time.time()
    ptq_time = (t1 - t0) / len(PROMPTS)
    print(f"PTQ avg time per prompt: {ptq_time:.2f}s")
    ptq_paths = save_images(imgs_ptq, "ptq")

    # 计算指标
    print("Computing metrics …")
    # Load images as tensors
    base_tensors = torch.cat([load_image_tensor(p) for p in base_paths], dim=0).to(DEVICE)
    ptq_tensors = torch.cat([load_image_tensor(p) for p in ptq_paths], dim=0).to(DEVICE)

    # LPIPS
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)
    lpips_val = lpips_metric(ptq_tensors, base_tensors)
    # SSIM
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    ssim_val = ssim_metric(ptq_tensors, base_tensors)

    print(f"LPIPS (PTQ vs Base): {lpips_val.item():.4f}")
    print(f"SSIM  (PTQ vs Base): {ssim_val.item():.4f}")

    # 写入 Markdown 报告
    report = OUTPUT_DIR / "report_ptq.md"
    with open(report, "w") as f:
        f.write("# PTQ Experiment Report\n\n")
        f.write(f"- Baseline UNet params: {params_base}, size: {size_base/1e6:.2f} MB\n")
        f.write(f"- PTQ    UNet params: {params_ptq}, size: {size_ptq/1e6:.2f} MB\n")
        f.write(f"- PTQ avg inference time per prompt: {ptq_time:.2f} s\n\n")
        f.write("## Metrics (PTQ vs Baseline)\n")
        f.write(f"- LPIPS: {lpips_val.item():.4f}\n")
        f.write(f"- SSIM : {ssim_val.item():.4f}\n")
    print(f"Report saved to {report}")

if __name__ == "__main__":
    main()