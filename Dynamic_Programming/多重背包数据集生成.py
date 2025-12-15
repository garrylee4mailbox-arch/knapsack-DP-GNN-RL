import os
import csv
import random

def generate_multiple_knapsack_instances_absolute_limit(
    num_instances=8,                 # ← 改为 8 组
    items_per_instance=200,          # 200 种物品
    capacity_range=(1500, 2000),     # 背包容量 1500~2000
    weight_range=(1, 100),           # 物品重量 1~100
    value_range=(1, 300),
    count_range=(0, 50),             # 每种物品最多 50 个
    output_dir="C:/Users/Mansycc/Desktop/wku/3440",
    filename="multiple_knapsack_extreme.csv"
):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        for idx in range(1, num_instances + 1):
            if idx > 1:
                writer.writerow([])  # 空行分隔不同实例

            W = random.randint(*capacity_range)
            n = items_per_instance

            writer.writerow([f"Instance {idx}"])
            writer.writerow(["capacity", W])
            writer.writerow(["n", n])
            writer.writerow(["weight", "value", "count"])

            for _ in range(n):
                w = random.randint(*weight_range)
                w = min(w, W)  # 确保重量不超过背包容量（逻辑更合理）
                v = random.randint(*value_range)
                c = random.randint(*count_range)
                writer.writerow([w, v, c])

    print("🔥 已成功生成 8 组【传统二维DP极限规模】多重背包实例")
    print(f"   文件路径: {filepath}")
    print(f"   配置详情: n={items_per_instance}, W∈{capacity_range}, count≤{count_range[1]}")
    print("   ⚠️ 注意：每个实例求解可能需要数分钟至数十分钟，请耐心等待！")

# 执行生成
if __name__ == "__main__":
    generate_multiple_knapsack_instances_absolute_limit()