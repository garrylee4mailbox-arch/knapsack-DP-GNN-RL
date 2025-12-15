import os
import csv
import time

def multiple_knapsack_2d_with_traceback(W, weights, values, counts):
    n = len(weights)
    if W <= 0:
        return 0, [0] * n

    # 创建 (n+1) x (W+1) 的 DP 表
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # 填表
    for i in range(1, n + 1):
        w, v, c = weights[i-1], values[i-1], counts[i-1]
        if w == 0:
            for j in range(W + 1):
                dp[i][j] = dp[i-1][j]  # weight=0 视为不可用或价值0
            continue
        for j in range(W + 1):
            dp[i][j] = dp[i-1][j]  # k = 0
            max_k = min(c, j // w)
            for k in range(1, max_k + 1):
                prev_j = j - k * w
                candidate = dp[i-1][prev_j] + k * v
                if candidate > dp[i][j]:
                    dp[i][j] = candidate

    # 回溯（从大到小搜索 k，提高正确率）
    selected = [0] * n
    j = W
    for i in range(n, 0, -1):
        w, v, c = weights[i-1], values[i-1], counts[i-1]
        current_val = dp[i][j]
        best_k = 0

        if w == 0:
            best_k = 0
        elif dp[i-1][j] == current_val:
            best_k = 0
        else:
            max_k = min(c, j // w)
            for k in range(max_k, 0, -1):  # 从大到小
                prev_j = j - k * w
                if prev_j >= 0 and dp[i-1][prev_j] + k * v == current_val:
                    best_k = k
                    break

        selected[i-1] = best_k
        j -= best_k * w
        if j < 0:
            j = 0  # 容错

    return dp[n][W], selected


def parse_instances_from_csv(filepath):
    instances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    i = 0
    while i < len(rows):
        row = rows[i]
        if not row or not row[0].strip():
            i += 1
            continue
        if row[0].startswith("Instance"):
            try:
                W = int(rows[i+1][1])
                n = int(rows[i+2][1])
                items = []
                for j in range(n):
                    r = rows[i+4+j]
                    items.append((int(r[0]), int(r[1]), int(r[2])))
                instances.append({'id': int(row[0].split()[1]), 'W': W, 'items': items})
                i += 4 + n
            except Exception as e:
                print(f"⚠️ 解析 Instance 失败 at line {i}: {e}")
                i += 1
        else:
            i += 1
    return instances


def solve_and_export_results(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "multiple_knapsack_results_full.csv")

    instances = parse_instances_from_csv(input_csv)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id", "capacity_W", "max_value",
            "selected_counts", "total_weight_used",
            "runtime_seconds", "status"
        ])

        for inst in instances:
            inst_id = inst['id']
            W = inst['W']
            items = inst['items']
            weights = [it[0] for it in items]
            values = [it[1] for it in items]
            counts = [it[2] for it in items]

            print(f"\n🚀 开始求解 Instance {inst_id} (W={W}, n={len(items)})...")

            try:
                start_time = time.perf_counter()
                max_val, selected = multiple_knapsack_2d_with_traceback(W, weights, values, counts)
                end_time = time.perf_counter()
                runtime = round(end_time - start_time, 4)
                total_weight = sum(w * k for w, k in zip(weights, selected))

                writer.writerow([
                    inst_id, W, max_val, str(selected),
                    total_weight, runtime, "solved"
                ])
                print(f"✅ Instance {inst_id} 完成 | 最优值: {max_val} | 耗时: {runtime}s")

            except MemoryError:
                writer.writerow([inst_id, W, "", "", "", "", "error: MemoryError (out of memory)"])
                print(f"💥 Instance {inst_id} 失败: 内存不足 (W={W})")
            except KeyboardInterrupt:
                writer.writerow([inst_id, W, "", "", "", "", "error: interrupted by user"])
                print("\n🛑 用户中断！保存已运行结果。")
                break
            except Exception as e:
                writer.writerow([inst_id, W, "", "", "", "", f"error: {str(e)}"])
                print(f"❌ Instance {inst_id} 异常: {e}")

    print(f"\n🎉 所有求解任务结束！结果已保存至:\n   {output_path}")


# 主程序入口
if __name__ == "__main__":
    input_file = r"C:/Users/Mansycc/Desktop/wku/3440/multiple_knapsack_extreme.csv"
    output_dir = r"C:/Users/Mansycc/Desktop/wku/3440"
    solve_and_export_results(input_file, output_dir)