import pandas as pd
import numpy as np
import os

np.random.seed(2025)

# 配置：物品数大，但容量小 → 保证 n*W <= 2e8
configs = [
    (5000, 3000),    # 15M 状态
    (6000, 4000),    # 24M
    (7000, 5000),    # 35M
    (8000, 6000),    # 48M
    (10000, 7000),   # 70M
    (12000, 8000),   # 96M
    (15000, 9000),   # 135M
    (20000, 10000)   # 200M —— 传统DP极限！
]

print("正在生成「传统DP可解」的万级背包数据集...")
print("特点：物品数达2万，但容量控制在1万以内，确保DP可行\n")

data = {}

for i, (n_items, capacity) in enumerate(configs, start=1):
    print(f"▶ 第 {i} 组: {n_items:,} 物品, 容量 = {capacity:,} (状态数 ≈ {n_items * capacity:,})")
    
    # 重量范围：1 ~ 100（确保总重远大于容量，问题有意义）
    weights = np.random.randint(1, 101, size=n_items)
    # 价值：与重量正相关
    values = (weights * np.random.uniform(0.8, 1.3)).astype(int)
    values = np.clip(values, 1, None)
    
    # 构造列（不同组长度不同，pandas 自动对齐）
    data[f'weight{i}'] = weights
    data[f'value{i}'] = values
    data[f'cap{i}'] = [capacity]

# 创建 DataFrame
df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

# 保存到桌面
output_path = "C:/Users/Mansycc/Desktop/wku/3440/dp_solvable_ultra_large.xlsx"
df.to_excel(output_path, sheet_name="Sheet1", index=False)

print(f"\n✅ 数据集已生成！")
print(f"📁 路径: {output_path}")
print("\n💡 提示：第8组 (20k物品, W=10k) 是传统DP的性能极限，预计耗时 30~60 秒")