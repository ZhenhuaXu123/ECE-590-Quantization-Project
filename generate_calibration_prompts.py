import random

# 主题类别保证 prompt 覆盖各种语境（extremely重要！）
TOPIC_GROUPS = {
    "人物": [
        "a portrait of an old man with wrinkles",
        "a smiling young woman with blue hair",
        "a realistic close-up face of a teenager",
        "a cyberpunk girl with neon reflections",
        "a boy reading a book near the window"
    ],
    "动物": [
        "a puppy wearing sunglasses",
        "a tiger running in the forest",
        "a parrot sitting on a branch",
        "a cartoon-style cat with big eyes",
        "a goldfish in a crystal bowl"
    ],
    "城市": [
        "a rainy street in Tokyo at night",
        "a futuristic city with flying cars",
        "a busy marketplace in Morocco",
        "a neon-lit alley with signs",
        "a skyline view at sunrise"
    ],
    "自然": [
        "a mountain landscape with fog",
        "a river flowing through rocks",
        "a beach with turquoise water",
        "a dense forest with sun rays",
        "a desert with sand dunes"
    ],
    "物体": [
        "a vintage camera on a table",
        "a futuristic robot arm",
        "a pair of red sneakers",
        "a golden pocket watch",
        "a ceramic teapot with patterns"
    ],
    "风格": [
        "an oil painting of a ship in storm",
        "a watercolor portrait of a girl",
        "a Van Gogh style landscape",
        "a 3D render of a crystal dragon",
        "a pixel art scene of a village"
    ],
    "抽象": [
        "an abstract geometric pattern",
        "a surreal floating island",
        "a fractal with rainbow colors",
        "a glowing sphere inside darkness",
        "a distorted mirror reflection"
    ],
    "科幻": [
        "a robot standing on Mars",
        "a spaceship entering hyperspace",
        "a hologram interface in blue color",
        "an android looking at the sky",
        "a sci-fi laboratory interior"
    ],
}

# 每类扩充数量
TARGET_PER_GROUP = 25   # 8 类 × 25 = 200 prompts

all_prompts = []

for topic, base_list in TOPIC_GROUPS.items():
    expanded = []
    for _ in range(TARGET_PER_GROUP):
        base = random.choice(base_list)
        style = random.choice([
            "in ultra-detailed style",
            "4K resolution",
            "cinematic lighting",
            "extremely detailed",
            "hyper realistic",
            "digital art",
            "soft lighting",
            "dramatic shadows"
        ])
        suffix = random.choice([
            "",
            ", trending on artstation",
            ", highly detailed",
            ", photorealistic",
            ", masterpiece"
        ])
        prompt = f"{base}, {style}{suffix}".strip()
        expanded.append(prompt)

    all_prompts.extend(expanded)

# 输出文件
with open("calibration_prompts.txt", "w", encoding="utf-8") as f:
    for p in all_prompts:
        f.write(p + "\n")

print("已生成 calibration_prompts.txt，共 200 条 prompt。")
