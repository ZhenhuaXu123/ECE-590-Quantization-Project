import os
from transformers import CLIPModel  # 补充导入语句
import subprocess  # 用于执行外部脚本
import time
import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.linalg import sqrtm

# 导入你项目的核心模块
from PTQ import build_ptq_model  # 朴素PTQ模型构建函数（与PTQ.py中的函数名一致）
from ptq_kld_diffusion import build_ptq_model_kld  # KLD-PTQ模型构建函数
from stable_diffusion_pytorch import model_loader, pipeline  # Stable Diffusion核心模块
from stable_diffusion_pytorch.util import get_file_path
print("CLIP模型路径：", get_file_path("ckpt/clip.pt"))

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./evaluation_results"  # 评估结果和生成图像保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 生成配置（与训练/量化的配置一致）
SEED = 42
N_INFERENCE_STEPS = 10  # 与校准的推理步数一致
SAMPLER = "k_lms"  # Stable Diffusion采样器
BATCH_SIZE = 1  # 单张生成，保证延迟统计准确
PROMPT_NUM = 200  # 评估用提示词数量
REPEAT_TIMES = 5  # 每种模型重复运行次数，用于计算标准差

# 图像预处理配置（适配LPIPS/SSIM/FID的输入要求）
IMG_SIZE = 512
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
])

# ===================== 工具函数 =====================
# 自动下载并生成CLIP权重
def prepare_clip_weights():
    clip_path = os.path.join("data/ckpt", "clip.pt")
    if not os.path.exists(clip_path):
        print("未找到CLIP权重，正在自动下载并生成clip.pt...")
        os.makedirs("data/ckpt", exist_ok=True)
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        torch.save(model.state_dict(), clip_path)
        print(f"CLIP权重已生成：{clip_path}")
    else:
        print(f"CLIP权重已存在：{clip_path}")
# 执行权重准备函数
prepare_clip_weights()


def set_seed(seed):
    """设置随机种子，保证实验可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_image(tensor, path):
    """将tensor保存为PIL图像"""
    tensor = tensor.clamp(-1, 1) * 0.5 + 0.5  # 反归一化到[0,1]
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray((tensor * 255).astype(np.uint8))
    img.save(path)

def load_image(path: str) -> torch.Tensor:
    """加载图像并转换为模型输入的tensor"""
    img = Image.open(path).convert("RGB")
    tensor = TRANSFORM(img)
    # 运行时类型校验：确保是PyTorch张量
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"期望torch.Tensor类型，实际得到{type(tensor)}，请检查TRANSFORM的定义")  
    return tensor.unsqueeze(0).to(DEVICE)


def calculate_fid(real_feats, fake_feats):
    """计算FID值（适配小批量数据，避免内存溢出）"""
    real_feats = real_feats.cpu().numpy()
    fake_feats = fake_feats.cpu().numpy()
    
    # 计算均值和协方差
    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    
    # 计算FID
    covmean = sqrtm(sigma1 @ sigma2, disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# 提示词加载函数
def load_evaluation_prompts(prompt_file="calibration_prompts.txt", num_prompts=None):
    """
    加载生成的校准提示词文件
    :param prompt_file: 提示词文件路径
    :param num_prompts: 截取的提示词数量（None则加载全部）
    :return: 提示词列表
    """
    if not os.path.exists(prompt_file):
        # 若文件不存在，执行generate_calibration_prompts.py生成
        subprocess.run(["python", "generate_calibration_prompts.py"], check=True)
    # 读取提示词
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                prompts.append(line)
    # 截取指定数量的提示词
    if num_prompts is not None and num_prompts > 0:
        prompts = prompts[:num_prompts]
    return prompts

# ===================== 模型加载 =====================
def load_all_models():
    """加载FP32基线、朴素PTQ、KLD-PTQ三种模型"""
    # 加载Stable Diffusion基础模型
    base_models = model_loader.preload_models(DEVICE)
    fp32_diffusion = base_models["diffusion"]
    
    # 构建量化模型（复用你项目中的量化函数）
    naive_ptq_diffusion = build_ptq_model(fp32_diffusion, DEVICE)  # 朴素PTQ
    kld_ptq_diffusion = build_ptq_model_kld(fp32_diffusion, DEVICE)     # KLD-PTQ
    
    # 封装为字典，方便批量评估
    models = {
        "FP32_Baseline": {
            "diffusion": fp32_diffusion,
            "other": {k: v for k, v in base_models.items() if k != "diffusion"}
        },
        "Naive_PTQ": {
            "diffusion": naive_ptq_diffusion,
            "other": {k: v for k, v in base_models.items() if k != "diffusion"}
        },
        "KLD_PTQ": {
            "diffusion": kld_ptq_diffusion,
            "other": {k: v for k, v in base_models.items() if k != "diffusion"}
        }
    }
    return models

# ===================== 核心评估模块 =====================
def evaluate_efficiency(model_dict, prompts):
    """评估模型效率：大小、推理延迟"""
    efficiency_results = {}
    for model_name, model_parts in tqdm(model_dict.items(), desc="Evaluating Efficiency"):
        diffusion_model = model_parts["diffusion"]
        all_latencies = []
        
        # 1. 计算模型大小（仅统计diffusion模型，因其他模块未量化）
        param_size = 0
        for param in diffusion_model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in diffusion_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # 2. 统计推理延迟（重复运行，排除偶然因素）
        for _ in range(REPEAT_TIMES):
            set_seed(SEED + _)  # 每次种子不同，避免生成相同图像的优化
            total_time = 0.0
            for prompt in prompts:
                t0 = time.time()
                # 生成图像（复用你项目的pipeline）
                _ = pipeline.generate(
                    prompts=[prompt],
                    models={**model_parts["other"], "diffusion": diffusion_model},
                    seed=SEED + _,
                    n_inference_steps=N_INFERENCE_STEPS,
                    sampler=SAMPLER,
                    device=DEVICE
                )
                t1 = time.time()
                total_time += (t1 - t0)
            # 计算单张图像平均延迟
            avg_latency = total_time / len(prompts)
            all_latencies.append(avg_latency)
        
        # 计算统计结果（平均值±标准差）
        latency_mean = np.mean(all_latencies)
        latency_std = np.std(all_latencies)
        
        efficiency_results[model_name] = {
            "model_size_mb": round(model_size_mb, 2),
            "latency_mean_s": round(latency_mean, 2),
            "latency_std_s": round(latency_std, 2),
            "compression_ratio": round(efficiency_results["FP32_Baseline"]["model_size_mb"] / model_size_mb, 2) 
            if model_name != "FP32_Baseline" else "-"
        }
    
    return efficiency_results

def evaluate_quality(model_dict, prompts):
    """评估图像质量：LPIPS、SSIM、FID"""
    # 初始化评估指标
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)  # LPIPS越小越好
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)        # SSIM越大越好
    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)                 # FID越小越好
    
    quality_results = {}
    fp32_images = []  # 保存FP32生成的图像，作为质量基准
    
    # 第一步：生成FP32基线图像并保存
    fp32_model = model_dict["FP32_Baseline"]
    for idx, prompt in tqdm(enumerate(prompts), desc="Generating FP32 Images"):
        set_seed(SEED + idx)
        img = pipeline.generate(
            prompts=[prompt],
            models={**fp32_model["other"], "diffusion": fp32_model["diffusion"]},
            seed=SEED + idx,
            n_inference_steps=N_INFERENCE_STEPS,
            sampler=SAMPLER,
            device=DEVICE
        )[0]  # 取单张图像
        img_path = os.path.join(SAVE_DIR, f"FP32_{idx}.png")
        save_image(img, img_path)
        fp32_images.append(load_image(img_path))
    
    # 第二步：评估量化模型的图像质量
    for model_name, model_parts in tqdm(model_dict.items(), desc="Evaluating Quality"):
        if model_name == "FP32_Baseline":
            continue  # FP32作为基准，无需自评估
        
        lpips_vals, ssim_vals = [], []
        fake_feats = []
        
        for idx, prompt in enumerate(prompts):
            set_seed(SEED + idx)
            # 生成量化模型的图像
            img = pipeline.generate(
                prompts=[prompt],
                models={**model_parts["other"], "diffusion": model_parts["diffusion"]},
                seed=SEED + idx,
                n_inference_steps=N_INFERENCE_STEPS,
                sampler=SAMPLER,
                device=DEVICE
            )[0]
            img_path = os.path.join(SAVE_DIR, f"{model_name}_{idx}.png")
            save_image(img, img_path)
            quant_img = load_image(img_path)
            fp32_img = fp32_images[idx]
            
            # 计算LPIPS和SSIM
            lpips_val = lpips(quant_img, fp32_img).item()
            ssim_val = ssim(quant_img, fp32_img).item()
            lpips_vals.append(lpips_val)
            ssim_vals.append(ssim_val)
            
            # 累积FID特征（FID需要批量计算）
            fid.update(quant_img, real=False)
            fake_feats.append(quant_img.squeeze(0).cpu())
        
        # 计算FID（与FP32基线对比）
        fid.update(torch.cat(fp32_images), real=True)
        fid_val = fid.compute().item()
        fid.reset()  # 重置FID指标，用于下一个模型
        
        # 计算统计结果
        quality_results[model_name] = {
            "lpips_mean": round(np.mean(lpips_vals), 4),
            "lpips_std": round(np.std(lpips_vals), 4),
            "ssim_mean": round(np.mean(ssim_vals), 4),
            "ssim_std": round(np.std(ssim_vals), 4),
            "fid": round(fid_val, 2)
        }
    
    return quality_results


# ===================== 主函数 =====================
def main():
    # 1. 加载提示词（调用工具函数板块的load_evaluation_prompts）
    prompts = load_evaluation_prompts(num_prompts=PROMPT_NUM)
    # 保存评估用提示词
    with open(os.path.join(SAVE_DIR, "evaluation_prompts.json"), "w") as f:
        json.dump(prompts, f, indent=4)
    
    
    # 2. 加载所有模型
    models = load_all_models()
    
    # 3. 评估模型效率
    efficiency_results = evaluate_efficiency(models, prompts)
    with open(os.path.join(SAVE_DIR, "efficiency_results.json"), "w") as f:
        json.dump(efficiency_results, f, indent=4)
    
    # 4. 评估图像质量
    quality_results = evaluate_quality(models, prompts)
    with open(os.path.join(SAVE_DIR, "quality_results.json"), "w") as f:
        json.dump(quality_results, f, indent=4)
    
    # 5. 打印汇总结果
    print("\n=== 模型效率评估结果 ===")
    for model_name, res in efficiency_results.items():
        print(f"{model_name}:")
        print(f"  模型大小: {res['model_size_mb']}MB | 压缩比: {res['compression_ratio']}")
        print(f"  平均延迟: {res['latency_mean_s']}±{res['latency_std_s']}s")
    
    print("\n=== 图像质量评估结果 ===")
    for model_name, res in quality_results.items():
        print(f"{model_name}:")
        print(f"  LPIPS: {res['lpips_mean']}±{res['lpips_std']} | SSIM: {res['ssim_mean']}±{res['ssim_std']} | FID: {res['fid']}")

if __name__ == "__main__":
    main()