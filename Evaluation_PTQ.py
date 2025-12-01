import os
import time
import psutil
import torch
import json
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import FrechetInceptionDistance
from stable_diffusion_pytorch.model_loader import load_all_models  # 模型加载函数
from stable_diffusion_pytorch.pipeline import generate  # 文生图生成函数
from torchvision import transforms
# ===================== 全局配置 =====================
DEVICE = "cpu"
SEED = 42
N_INFERENCE_STEPS = 30
SAMPLER = "k_lms"
# .npz评估集路径（下载的文件路径）
NPZ_EVAL_PATH = "ECE-590-11-Quantization-project-Evaluation/ECE-590-Quantization-Project/data/eval_set/coco_val_2017_512x512.npz"
# 评估集采样数量（避免全量计算耗时，可根据需求调整）
EVAL_SAMPLE_NUM = 5
npz_data = np.load("ECE-590-11-Quantization-project-Evaluation/ECE-590-Quantization-Project/data/eval_set/coco_val_2017_512x512.npz", allow_pickle=True)
print("npz文件包含的key：", npz_data.files)
NPZ_EVAL_PATH = "ECE-590-11-Quantization-project-Evaluation/ECE-590-Quantization-Project/data/eval_set/coco_val_2017_512x512.npz"
# ===================== 工具函数：加载.npz评估集 =====================
def load_npz_eval_set(npz_path, sample_num=5):
    """
    加载.npz格式的文生图评估集
    返回：真实图像张量（N,3,H,W）、对应的文本提示词列表
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到.npz评估集：{npz_path}，请先下载并放在指定路径")
    
    # 加载.npz文件（NumPy压缩格式）
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # 提取图像张量和提示词（不同数据集的key可能不同，需根据npz文件调整）
    # 常见key："images"（图像张量）、"captions"（提示词）、"prompts"（提示词）
    images = torch.from_numpy(npz_data["images"]).to(DEVICE)  # (N,3,512,512)
    captions = npz_data["captions"].tolist()  # 提示词列表
    
    # 采样指定数量的样本（避免全量计算）
    if sample_num < len(images):
        indices = torch.randperm(len(images))[:sample_num]
        images = images[indices]
        captions = [captions[i] for i in indices]
    
    # 图像归一化（适配模型输入，根据npz文件的像素范围调整）
    if images.max() > 1.0:
        images = images / 255.0  # 若像素范围是0-255，转为0-1
    images = (images - 0.5) / 0.5  # 归一化到[-1,1]（Stable Diffusion输入格式）
    
    print(f"成功加载.npz评估集：{npz_path}，采样{len(images)}个样本")
    return images, captions

# ===================== 效率指标计算 =====================
def calculate_model_size(model, save_dir="./tmp_model"):
    """计算模型磁盘存储大小（MB）"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "tmp_model.pt")
    torch.save(model.state_dict(), save_path)
    model_size = os.path.getsize(save_path) / (1024 * 1024)
    os.remove(save_path)
    os.rmdir(save_dir)
    return model_size

def evaluate_efficiency(model_dict, prompts, warmup=5, repeat=10):
    """评估模型推理效率（时间、显存、吞吐量、体积）"""
    efficiency_results = {}
    for model_name, model_parts in model_dict.items():
        diffusion = model_parts["diffusion"]
        other_models = model_parts["other"]
        
        # 1. 预热（消除初始化开销）
        print(f"[{model_name}] 预热推理...")
        for _ in range(warmup):
            generate(
                prompts=[prompts[0]],
                models={**other_models, "diffusion": diffusion},
                seed=SEED,
                n_inference_steps=N_INFERENCE_STEPS,
                sampler=SAMPLER,
                device=DEVICE
            )
        
        # 2. 推理时间与吞吐量
        print(f"[{model_name}] 计算推理时间...")
        start_time = time.time()
        for _ in range(repeat):
            generate(
                prompts=[prompts[0]],
                models={**other_models, "diffusion": diffusion},
                seed=SEED,
                n_inference_steps=N_INFERENCE_STEPS,
                sampler=SAMPLER,
                device=DEVICE
            )
        total_time = time.time() - start_time
        avg_time = total_time / repeat
        throughput = repeat / total_time
        
        # 3. 显存/内存占用
        process = psutil.Process(os.getpid())
        alloc_mem = process.memory_info().rss / (1024 * 1024)
        max_alloc_mem = alloc_mem
        
        # 4. 模型体积（仅Diffusion主模型，可扩展为全模型）
        model_size = calculate_model_size(diffusion)
        
        efficiency_results[model_name] = {
            "model_size_mb": round(model_size, 2),
            "avg_infer_time_s": round(avg_time, 4),
            "throughput_img_per_s": round(throughput, 4),
            "alloc_mem_mb": round(alloc_mem, 2),
            "max_alloc_mem_mb": round(max_alloc_mem, 2)
        }
    return efficiency_results

# ===================== 图像质量指标计算（适配.npz） =====================
def evaluate_quality(model_dict, real_images, real_prompts):
    """评估图像质量（FID、LPIPS、PSNR、SSIM），输入为.npz加载的真实图像和提示词"""
    # 初始化指标
    print("归一化前 real_images最小值:", real_images.min().item())
    print("归一化前 real_images最大值:", real_images.max().item())
    # 把real_images从[-3, 1]缩放到[-1, 1]
    min_val = -3.0
    max_val = 1.0
    real_images = (real_images - min_val) / (max_val - min_val)  # 缩放到[0,1]
    real_images = real_images * 2 - 1  # 转成[-1,1]
    
    # 再打印归一化后的范围（验证是否生效）
    print("归一化后 real_images最小值:", real_images.min().item())
    print("归一化后 real_images最大值:", real_images.max().item())

    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)
    lpips = LPIPS(net_type='vgg').to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)  # 适配[-1,1]归一化
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    # 真实图像送入FID（标记为real=True）
    fid.update(real_images, real=True)
    
    quality_results = {}
    for model_name, model_parts in model_dict.items():
        generated_images = []
        print(type(real_prompts), real_prompts)  # 应该输出 <class 'list'> 或 <class 'tuple'>，且非空
        print(f"[{model_name}] 生成评估图像...")
        # 根据.npz的真实提示词生成图像
        for i, prompt in enumerate(real_prompts):
            img = generate(
                prompts=[prompt],
                models={**model_parts["other"], "diffusion": model_parts["diffusion"]},
                seed=SEED,
                n_inference_steps=N_INFERENCE_STEPS,
                sampler=SAMPLER,
                device=DEVICE
            )[0]
            # 正常清晰的PIL Image应该是"RGB"模式，像素范围是(0,255)
            print(f"img模式: {img.mode}, 像素范围: {img.getextrema()}")
            # 命名规则：模型名_第i个prompt.png（避免不同模型/不同prompt的图重名）
            img_save_name = f"{model_name}_prompt_{i}.png"
            img.save(img_save_name)  # 保存PIL Image到当前目录
            print(f"已保存图片：{img_save_name}")
            # 返回的是PIL Image，形状是(H, W, 3)
            # 1. PIL Image转numpy数组（此时是0-255的uint8类型）
            img_np = np.array(img)  # 形状是(H, W, 3)
            #print("img_np的形状是：", img_np.shape) #（1， 512， 512， 3）
            # 2. 归一化（先缩放到0-2，再减1得到-1到1)
            img_np = (img_np.astype(np.float32) / 127.5) - 1.0  
            # 3. 转成tensor并移到设备
            img_tensor = torch.tensor(img_np).to(DEVICE)
            generated_images.append(img_tensor.squeeze(0))  # 移除batch维度
        
        # 拼接生成图像张量
        generated_tensor = torch.stack(generated_images) # 形状：(批次, 512, 512, 3)
        #维度置换（通道在后 → 通道在前）
        # 把通道从最后一维移到第2维 → 变成(批次, 3, 512, 512)
        generated_tensor_permute = generated_tensor.permute(0, 3, 1, 2)  
        # 恢复数据范围：把-1~1转成0~255（因为img_np是-1~1，FID需要0~255）
        generated_tensor_permute = ((generated_tensor_permute + 1) * 127.5).clamp(0, 255).float()  
        print(f"生成图像的数量: {generated_tensor_permute.shape[0]}")  # 打印批次维度（第0维）
        # 计算FID
        fid.update(generated_tensor_permute, real=False)
        fid_score = fid.compute().item()
        fid.reset()  # 重置FID用于下一个模型
        
        # 计算LPIPS、PSNR、SSIM（需保证真实图像和生成图像形状一致）
        lpips_score = lpips(generated_tensor_permute, real_images).mean().item()
        psnr_score = psnr((generated_tensor_permute + 1) / 2, (real_images + 1) / 2).mean().item()  # 转回0-1范围
        ssim_score = ssim((generated_tensor_permute + 1) / 2, (real_images + 1) / 2).mean().item()
        
        quality_results[model_name] = {
            "fid": round(fid_score, 4),
            "lpips": round(lpips_score, 4),
            "psnr": round(psnr_score, 4),
            "ssim": round(ssim_score, 4)
        }
    return quality_results

# ===================== 主函数与结果整合 =====================
def main():
    # 1. 加载.npz格式的文生图评估集
    print("===== 加载.npz文生图评估集 =====")
    real_images, real_prompts = load_npz_eval_set(NPZ_EVAL_PATH, EVAL_SAMPLE_NUM)
    # 1.1 检查real_prompts的长度（决定生成图像的数量）
    print(f"real_prompts长度: {len(real_prompts)}")
    if len(real_prompts) < 2:
        raise ValueError("real_prompts长度必须≥2（FID需要至少2个生成样本）")
    # 1.2 检查real_images的批次大小（假设real_images是张量，形状为[N, C, H, W]）
    print(f"real_images批次大小: {real_images.shape[0]}")
    if real_images.shape[0] < 2:
        raise ValueError("real_images的批次大小必须≥2（FID需要至少2个真实样本）")
    # 2. 加载所有待评估模型（FP32、朴素PTQ、KLD-PTQ）
    print("\n===== 加载待评估模型 =====")
    models = {
        "FP32_Baseline": load_all_models(DEVICE, quant_type="fp32"),  # FP32基准
        "Naive_PTQ": load_all_models(DEVICE, quant_type="naive_ptq"),  # 朴素PTQ
        "KLD_PTQ": load_all_models(DEVICE, quant_type="kld_ptq")      # KLD-PTQ
    }
    
    # 3. 评估效率指标（模型大小、推理时间、显存）
    print("\n===== 评估模型效率指标 =====")
    efficiency_results = evaluate_efficiency(models, real_prompts)
    
    # 4. 评估图像质量指标（FID、LPIPS、PSNR、SSIM）
    print("\n===== 评估图像质量指标 =====")
    quality_results = evaluate_quality(models, real_images, real_prompts)
    
    # 5. 整合结果并生成报告
    final_report = {}
    for model_name in models:
        final_report[model_name] = {
            "efficiency": efficiency_results[model_name],
            "quality": quality_results[model_name]
        }
    
    # 打印可视化报告
    print("\n===== 文生图模型量化评估最终报告 =====")
    for name, result in final_report.items():
        print(f"\n【{name}】")
        print("--- 效率指标 ---")
        for k, v in result["efficiency"].items():
            print(f"{k}: {v}")
        print("--- 图像质量 ---")
        for k, v in result["quality"].items():
            print(f"{k}: {v}")
    
    # 保存报告到JSON文件
    with open("ptq_evaluation_report_npz.json", "w") as f:
        json.dump(final_report, f, indent=4)
    print("\n评估报告已保存至：ptq_evaluation_report_npz.json")

if __name__ == "__main__":
    main()