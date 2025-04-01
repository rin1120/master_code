import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# パラメータ設定
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 3
NUM_CONTENTS_TO_SEARCH = 10  # 探索するコンテンツ数
NUM_ANTS = 10
NUM_ITERATIONS = 100

ALPHA = 1.0   # フェロモンの指数
BETA = 1.0    # SOM類似度の指数
RHO = 0.1     # 蒸発率
Q = 100

TIMES_TO_SEARCH_HOP = 50
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

USE_FIXED_START_NODE = False

# ε-greedy戦略用パラメータ（リセットなしの場合のみ使用）
USE_EPSILON = True
EPSILON = 0.1  # 10%の確率でランダム選択

# 停滞検出に基づく部分リセット用パラメータ（リセットなしの場合のみ使用）
USE_STAGNATION_RESET = False
stagnation_window = 10         # 直近10イテレーション
stagnation_threshold = 0.01    # 変動率が1%未満なら停滞と判断
reset_factor = 0.5             # 部分リセット時、フェロモン値を半減

# CSVファイルから属性ベクトルを準備
file_path = "500_movies.csv"  # 適宜修正
df = pd.read_csv(file_path)
N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec) for vec in cont_vector]

# ----------------------------------------------------------------------------
# 2つのコンテンツIDの距離を計算する関数
# ----------------------------------------------------------------------------
def compute_content_distance(cid_a, cid_b):
    vect_a = cont_vector_array[cid_a - 1]
    vect_b = cont_vector_array[cid_b - 1]
    return np.linalg.norm(vect_a - vect_b)

# ----------------------------------------------------------------------------
# ネットワーク関連の関数
# ----------------------------------------------------------------------------
def get_neighbors(x, y, size):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

def get_init_cache_storage(size):
    cache_storage = np.empty((size, size), dtype=object)
    for i in range(size):
        for j in range(size):
            cache_storage[i][j] = []
    return cache_storage

def get_init_network_vector(size):
    incr = VECTOR_INCREMENT
    net_vector = np.array([np.arange(0, 1 + incr, incr) for _ in range(N)])
    return np.random.choice(net_vector.flatten(), (size, size, N))

def initialize_pheromone_trails(size, num_attributes):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            for neighbor in get_neighbors(x, y, size):
                pheromone_trails[(curr_node, neighbor)] = np.ones(num_attributes)
    return pheromone_trails

def initialize_single_pheromone_trails(size):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            for neighbor in get_neighbors(x, y, size):
                pheromone_trails[(curr_node, neighbor)] = 1.0
    return pheromone_trails

def cache_prop(size):
    cache_storage = get_init_cache_storage(size)
    net_vector_array = get_init_network_vector(size)
    cache_num = TIME_TO_CACHE_PER_CONTENT
    cache_hops = TIMES_TO_CACHE_HOP
    alpha_zero = LEARNING_RATE
    for _ in range(cache_num):
        for cid in range(1, cont_num + 1):
            curr = (np.random.randint(size), np.random.randint(size))
            vect = cont_vector_array[cid - 1]
            hoped_node = []
            for _2 in range(cache_hops):
                min_dist = float('inf')
                closest = None
                for neighbor in get_neighbors(curr[0], curr[1], size):
                    if neighbor not in hoped_node:
                        neighbor_vect = net_vector_array[neighbor]
                        dist = np.linalg.norm(vect - neighbor_vect)
                        if dist < min_dist:
                            min_dist = dist
                            closest = neighbor
                if closest is None:
                    break
                hoped_node.append(curr)
                curr = closest
            hoped_node.append(curr)
            cache_storage[curr[0]][curr[1]].append(cid)
            tmp = 0
            total_hops = len(hoped_node)
            for node in hoped_node:
                tmp += 1
                alpha = alpha_zero * (tmp / total_hops)
                net_vector_array[node] += alpha * (vect - net_vector_array[node])
    return cache_storage, net_vector_array

# ----------------------------------------------------------------------------
# 複数コンテンツ探索タスク生成
# ----------------------------------------------------------------------------
def generate_multi_contents_tasks(size, k=NUM_CONTENTS_TO_SEARCH):
    tasks = []
    for _ in range(k):
        cid = np.random.randint(1, cont_num + 1)
        if USE_FIXED_START_NODE:
            start_node = (size // 2, size // 2)
        else:
            start_node = (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start_node))
    return tasks

# ----------------------------------------------------------------------------
# 統計量計算
# ----------------------------------------------------------------------------
def compute_stats_from_costs(iteration_costs_data):
    medians, q1s, q3s, means, success_rates = [], [], [], [], []
    for costs in iteration_costs_data:
        medians.append(np.median(costs))
        q1s.append(np.percentile(costs, 25))
        q3s.append(np.percentile(costs, 75))
        means.append(np.mean(costs))
        total = len(costs)
        success_count = sum(1 for c in costs if c < TIMES_TO_SEARCH_HOP)
        success_rates.append((success_count / total) * 100)
    return medians, q1s, q3s, means, success_rates

# ----------------------------------------------------------------------------
# 既存手法による探索
# ----------------------------------------------------------------------------
def search_prop(cache_storage, net_vector_array, size, content_tasks):
    cache_hit = 0
    total_hops_used = 0
    for (cid, start_node) in content_tasks:
        searched_node = []
        curr = start_node
        found = False
        hops_used = 0
        for _ in range(TIMES_TO_SEARCH_HOP):
            hops_used += 1
            if cid in cache_storage[curr[0]][curr[1]]:
                found = True
                break
            for nx, ny in get_neighbors(curr[0], curr[1], size):
                if cid in cache_storage[nx][ny]:
                    found = True
                    break
            if found:
                break
            min_dist = float('inf')
            closest = None
            vect = cont_vector_array[cid - 1]
            for neighbor in get_neighbors(curr[0], curr[1], size):
                if neighbor not in searched_node:
                    dist = np.linalg.norm(vect - net_vector_array[neighbor])
                    if dist < min_dist:
                        min_dist = dist
                        closest = neighbor
            searched_node.append(curr)
            if all(nb in searched_node for nb in get_neighbors(curr[0], curr[1], size)):
                break
            if closest is None:
                break
            curr = closest
        if found:
            cache_hit += 1
        total_hops_used += hops_used
    avg_hops = total_hops_used / len(content_tasks) if content_tasks else 0.0
    return cache_hit, avg_hops

# ----------------------------------------------------------------------------
# ① 単一フェロモン方式 (リセットあり)
# ----------------------------------------------------------------------------
def multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks):
    results = []
    for (cid, start_node) in content_tasks:
        pheromone_trails = initialize_single_pheromone_trails(size)
        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([])
            continue
        vect = cont_vector_array[cid - 1]
        iteration_data = []
        for _ in range(NUM_ITERATIONS):
            iter_costs = []
            all_paths = []
            all_costs = []
            for _ in range(NUM_ANTS):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                cost = 0
                found = False
                for _ in range(TIMES_TO_SEARCH_HOP):
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break
                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed = [n for n in neighbors if n not in visited]
                    if not allowed:
                        break
                    if USE_EPSILON and random.random() < EPSILON:
                        next_node = random.choice(allowed)
                    else:
                        probs = []
                        denom = 0
                        for n in allowed:
                            edge = (current_node, n)
                            tau = pheromone_trails[edge]
                            dist = np.linalg.norm(vect - net_vector_array[n])
                            eta = 1.0 / (dist + 1e-6)
                            score = (tau ** ALPHA) * (eta ** BETA)
                            probs.append(score)
                            denom += score
                        if denom == 0:
                            break
                        probs = [p / denom for p in probs]
                        next_node = random.choices(allowed, weights=probs)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    cost += 1
                    current_node = next_node
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break
                if found:
                    all_paths.append(path)
                    all_costs.append(cost)
                    iter_costs.append(cost)
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)
            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    delta = Q / cost
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta
            iteration_data.append(iter_costs)
        results.append(iteration_data)
    return results

# ----------------------------------------------------------------------------
# ② 属性フェロモン方式 共通処理
# reset_pheromone=True: リセットあり、False: リセットなし
# ここでは、固定更新（ブーストなし）で、ε-greedyと停滞検出による部分リセットは
# リセットなしのときのみ適用する
def multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, reset_pheromone=True):
    results = []
    if not reset_pheromone:
        global_pheromone_trails = initialize_pheromone_trails(size, N)
    for (cid, start_node) in content_tasks:
        if reset_pheromone:
            pheromone_trails = initialize_pheromone_trails(size, N)
        else:
            pheromone_trails = global_pheromone_trails
        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([])
            continue
        vect = cont_vector_array[cid - 1]
        iteration_data = []
        best_cost = TIMES_TO_SEARCH_HOP  # 最良結果の記録（更新はするが固定更新に戻す）
        avg_costs = []  # 停滞検出用
        for _ in range(NUM_ITERATIONS):
            iter_costs = []
            all_paths = []
            all_costs = []
            for _ in range(NUM_ANTS):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                cost = 0
                found = False
                for _ in range(TIMES_TO_SEARCH_HOP):
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break
                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed = [n for n in neighbors if n not in visited]
                    if not allowed:
                        break
                    # ここで、ε-greedyはリセットなしの場合のみ適用
                    if not reset_pheromone and USE_EPSILON and random.random() < EPSILON:
                        next_node = random.choice(allowed)
                    else:
                        probs = []
                        denom = 0
                        for n in allowed:
                            edge = (current_node, n)
                            tau_list = pheromone_trails[edge]
                            dist = np.linalg.norm(vect - net_vector_array[n])
                            eta = 1.0 / (dist + 1e-6)
                            if np.sum(vect) > 0:
                                A_ij = np.sum((tau_list ** ALPHA) * vect) / np.sum(vect)
                            else:
                                A_ij = 0
                            score = A_ij * (eta ** BETA)
                            probs.append(score)
                            denom += score
                        if denom == 0:
                            break
                        probs = [p / denom for p in probs]
                        next_node = random.choices(allowed, weights=probs)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    cost += 1
                    current_node = next_node
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break
                if found:
                    all_paths.append(path)
                    all_costs.append(cost)
                    iter_costs.append(cost)
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)
            # 停滞検出（リセットなしの場合のみ適用）
            if not reset_pheromone:
                avg_cost = np.mean(iter_costs) if iter_costs else TIMES_TO_SEARCH_HOP
                avg_costs.append(avg_cost)
                if USE_STAGNATION_RESET and len(avg_costs) >= stagnation_window:
                    window = avg_costs[-stagnation_window:]
                    variation = (max(window) - min(window)) / (min(window) + 1e-6)
                    if variation < stagnation_threshold:
                        for edge in pheromone_trails:
                            pheromone_trails[edge] *= reset_factor
                        avg_costs = []
            # 更新：最良結果の更新（best_costは記録のみ）
            if all_costs:
                iteration_best = min(all_costs)
                if iteration_best < best_cost:
                    best_cost = iteration_best
            # フェロモンの蒸発更新
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)
            # 固定更新：delta = Q * vect / cost
            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    delta = (Q * vect) / cost
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta
            iteration_data.append(iter_costs)
        results.append(iteration_data)
    return results

def multi_contents_attrib_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks):
    return multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, reset_pheromone=True)

def multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks):
    return multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, reset_pheromone=False)

def gather_stats_for_content(content_index, single_all_runs, attrib_reset_all_runs, attrib_noreset_all_runs):
    s_stat = average_iteration_data_across_runs(single_all_runs, content_index)
    ar_stat = average_iteration_data_across_runs(attrib_reset_all_runs, content_index)
    an_stat = average_iteration_data_across_runs(attrib_noreset_all_runs, content_index)
    return (s_stat, ar_stat, an_stat)

def plot_three_metrics_for_content(stat_single, stat_attrib_reset, stat_attrib_noreset, content_label):
    if (stat_single is None) or (stat_attrib_reset is None) or (stat_attrib_noreset is None):
        print(f"No data for content {content_label}")
        return
    s_med, s_q1, s_q3, s_mean, s_succ = stat_single
    ar_med, ar_q1, ar_q3, ar_mean, ar_succ = stat_attrib_reset
    an_med, an_q1, an_q3, an_mean, an_succ = stat_attrib_noreset
    its = range(1, NUM_ITERATIONS + 1)
    plt.figure()
    plt.plot(its, s_med, label='Single(Reset)', color='red', marker='o')
    plt.fill_between(its, s_q1, s_q3, color='red', alpha=0.2)
    plt.plot(its, ar_med, label='Attrib(Reset)', color='blue', marker='s')
    plt.fill_between(its, ar_q1, ar_q3, color='blue', alpha=0.2)
    plt.plot(its, an_med, label='Attrib(NoReset)', color='green', marker='^')
    plt.fill_between(its, an_q1, an_q3, color='green', alpha=0.2)
    plt.title(f"Content {content_label}: Median & Quartiles")
    plt.xlabel("Iteration")
    plt.ylabel("Hops")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(its, s_succ, label='Single(Reset)', color='red', marker='o')
    plt.plot(its, ar_succ, label='Attrib(Reset)', color='blue', marker='s')
    plt.plot(its, an_succ, label='Attrib(NoReset)', color='green', marker='^')
    plt.title(f"Content {content_label}: Success Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(its, s_mean, label='Single(Reset)', color='red', marker='o')
    plt.plot(its, ar_mean, label='Attrib(Reset)', color='blue', marker='s')
    plt.plot(its, an_mean, label='Attrib(NoReset)', color='green', marker='^')
    plt.title(f"Content {content_label}: Average Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Average Hops")
    plt.legend()
    plt.show()

def average_iteration_data_across_runs(all_runs, content_index=0):
    iteration_data_runs = []
    for run_data in all_runs:
        if len(run_data) > content_index and len(run_data[content_index]) > 0:
            iteration_data_runs.append(run_data[content_index])
    if not iteration_data_runs:
        return None
    num_iterations = len(iteration_data_runs[0])
    combined_iteration_data = []
    for it in range(num_iterations):
        merged = []
        for one_run in iteration_data_runs:
            merged.extend(one_run[it])
        combined_iteration_data.append(merged)
    return compute_stats_from_costs(combined_iteration_data)

def main():
    size = 50
    cache_storage, net_vector_array = cache_prop(size)
    content_tasks = generate_multi_contents_tasks(size, k=NUM_CONTENTS_TO_SEARCH)
    print("Generated Content Tasks:", content_tasks)
    if len(content_tasks) >= 2:
        A = content_tasks[0][0]
        B = content_tasks[1][0]
        distance = compute_content_distance(A, B)
        print(f"Distance between content {A} and {B} = {distance:.3f}")
    single_all_runs = []
    attrib_reset_all_runs = []
    attrib_noreset_all_runs = []
    for _sim in range(TIME_TO_SIMULATE):
        single_res = multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks)
        attrib_reset_res = multi_contents_attrib_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks)
        attrib_noreset_res = multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks)
        single_all_runs.append(single_res)
        attrib_reset_all_runs.append(attrib_reset_res)
        attrib_noreset_all_runs.append(attrib_noreset_res)
    cache_hit, avg_hops = search_prop(cache_storage, net_vector_array, size, content_tasks)
    success_rate = (cache_hit / len(content_tasks)) * 100
    print("\n=== Existing Method (Same Tasks) ===")
    print(f"  - #Tasks : {len(content_tasks)}")
    print(f"  - Cache Hit: {cache_hit}")
    print(f"  - Success Rate: {success_rate:.2f}%")
    print(f"  - Avg Hops: {avg_hops:.2f}")
    for content_idx in range(NUM_CONTENTS_TO_SEARCH):
        print(f"=== [Comparison] Content #{content_idx+1} ===")
        stat_s, stat_ar, stat_an = gather_stats_for_content(content_idx, single_all_runs, attrib_reset_all_runs, attrib_noreset_all_runs)
        plot_three_metrics_for_content(stat_s, stat_ar, stat_an, content_label=content_idx+1)

if __name__ == "__main__":
    main()
