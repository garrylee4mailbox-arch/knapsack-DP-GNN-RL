import pandas as pd
import time

def knapsack_01_with_path(weights, values, capacity):
    """
    使用二维DP求解0-1背包，并返回选中的物品（0/1列表）
    返回: (max_value, selected_list)
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0, [0] * n

    # 创建 DP 表
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # 填充 DP 表
    for i in range(1, n + 1):
        wi = weights[i - 1]
        vi = values[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if wi <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - wi] + vi)

    # 回溯找出选中的物品
    selected = [0] * n
    w = capacity
    for i in range(n, 0, -1):
        # 如果当前值不等于上一行的值，说明第 i 个物品被选了
        if dp[i][w] != dp[i - 1][w]:
            selected[i - 1] = 1
            w -= weights[i - 1]

    return dp[n][capacity], selected

def main():
    input_path = "C:/Users/Mansycc/Desktop/wku/3440/dp_solvable_ultra_large.xlsx"
    df = pd.read_excel(input_path, sheet_name="Sheet1")
    
    print("所有列名:", df.columns.tolist())
    results = []

    for group_idx in range(8):
        weight_col = f"weight{group_idx + 1}"
        value_col = f"value{group_idx + 1}"
        cap_col = f"cap{group_idx + 1}"

        weights = df[weight_col].dropna().astype(int).tolist()
        values = df[value_col].dropna().astype(int).tolist()

        cap_series = df[cap_col].dropna()
        if cap_series.empty:
            print(f"第 {group_idx + 1} 组：缺少容量 {cap_col}，跳过")
            continue
        capacity = int(cap_series.iloc[0])

        if len(weights) != len(values):
            print(f"第 {group_idx + 1} 组：weight 和 value 长度不一致！跳过")
            continue
        if not weights:
            print(f"第 {group_idx + 1} 组：无物品数据，跳过")
            continue

        start = time.perf_counter()
        max_val, selected = knapsack_01_with_path(weights, values, capacity)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000

        # 将 selected 转为字符串，如 "1,0,1,1,..."
        selected_str = ','.join(map(str, selected))

        results.append({
            'Group': group_idx + 1,
            'Capacity': capacity,
            'Items': len(weights),
            'Max Value': max_val,
            'Time(ms)': elapsed_ms,
            'Selected (0/1)': selected_str  # 新增列
        })

        print(f"第 {group_idx + 1} 组 | 容量: {capacity:>8} | 物品数: {len(weights):>2} | 最大价值: {max_val:>10} | 耗时: {elapsed_ms:>8.4f} ms")
        print(f"选中物品: {selected_str}\n")

    # 导出到桌面
    output_path = "C:/Users/Mansycc/Desktop/wku/3440/knapsack_results.xlsx"
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_path, index=False)
    print(f"✅ 结果已保存至：{output_path}")

if __name__ == "__main__":
    main()