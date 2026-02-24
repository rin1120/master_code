# 山代法＋比較手法（コンテンツ毎にフェロモン管理）＋提案手法（属性毎にフェロモン管理）
# 提案・比較手法は類似度を考慮しない経路選択
# 比較手法、提案手法どちらにもローカルベストの探索を実装
# 探索性能は全ての探索を反映（成功時と失敗時の両方）
# コンテンツ生成手法をランダム、zipf、類似の3種類実装

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import namedtuple
from collections import defaultdict

# ------------------------------------------------------------
# 乱数を固定（再現性向上）
# ------------------------------------------------------------
np.random.seed(42)
random.seed(42)

# ----------------------------------------------------------------------------
# パラメータ設定
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 3
NUM_CONTENTS_TO_SEARCH = 100  # 探索するコンテンツ数
NUM_ANTS = 10
NUM_ITERATIONS = 10

ALPHA = 2.0   # フェロモンの指数
BETA = 1.0    # SOM類似度の指数（未使用だが保持）
Q = 100

# --- 可変蒸発・付加パラメータ ----------------------------------------
RHO       = 0.10   # ベース蒸発率
RHO_MAX   = 0.9    # 蒸発率の上限
LAMBDA_R  = 1      # 低品質経路の追加蒸発強度
BOOST     = 5      # 最良パスへの付加増強係数
# ---------------------------------------------------------------

TIMES_TO_SEARCH_HOP = 100
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

USE_FIXED_START_NODE = False

# ε-greedy戦略（※転移の差を見やすくするためデフォルトOFF）
USE_EPSILON = True
EPSILON = 0.01

# 実験モード（'r'|'z'|'s'|'s_nodup'）
TASK_MODE = 'r'

# CSVファイルから属性ベクトルを準備（列は属性のみ）
file_path = "r500_movies.csv"
df = pd.read_csv(file_path)
N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec, dtype=float) for vec in cont_vector]

# --- ジャンルとコンテンツIDの対応辞書を作成 ---
non_genre_cols = ['id', 'release_date', 'revenue', 'runtime']
genre_cols = [col for col in df.columns if col not in non_genre_cols]
genre_to_cids = defaultdict(list)
for genre_name in genre_cols:
    cids_in_genre = df[df[genre_name] == 1]['id'].tolist()
    if cids_in_genre:
        genre_to_cids[genre_name] = cids_in_genre

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
                pheromone_trails[(curr_node, neighbor)] = np.ones(num_attributes, dtype=float)
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
# コンテンツ探索タスク生成
# ----------------------------------------------------------------------------
def generate_multi_contents_tasks_r(size, k=NUM_CONTENTS_TO_SEARCH):
    tasks = []
    for _ in range(k):
        cid = np.random.randint(1, cont_num + 1)
        start_node = (size // 2, size // 2) if USE_FIXED_START_NODE else (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start_node))
    return tasks

def generate_multi_contents_tasks_z(size, k=NUM_CONTENTS_TO_SEARCH, popularity_ratio=0.01, request_concentration=0.8):
    tasks = []
    num_popular = int(cont_num * popularity_ratio)
    all_cids = np.arange(1, cont_num + 1)
    popular_cids = np.random.choice(all_cids, num_popular, replace=False)
    unpopular_cids = np.setdiff1d(all_cids, popular_cids)
    for _ in range(k):
        cid = random.choice(popular_cids) if random.random() < request_concentration else random.choice(unpopular_cids)
        start_node = (size // 2, size // 2) if USE_FIXED_START_NODE else (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start_node))
    return tasks

def generate_multi_contents_tasks_s(size, k=NUM_CONTENTS_TO_SEARCH, burst_size=10):
    tasks = []
    available_genres = list(genre_to_cids.keys())
    if not available_genres:
        raise ValueError("ジャンル情報が見つかりませんでした。")
    while len(tasks) < k:
        current_genre = random.choice(available_genres)
        cids_in_genre = genre_to_cids[current_genre]
        for _ in range(burst_size):
            if len(tasks) >= k: break
            cid = random.choice(cids_in_genre)
            start_node = (size // 2, size // 2) if USE_FIXED_START_NODE else (np.random.randint(size), np.random.randint(size))
            tasks.append((cid, start_node))
    return tasks

# 同一CID再登場なし（任意で使用）
def generate_multi_contents_tasks_s_nodup(size, k=NUM_CONTENTS_TO_SEARCH, burst_size=10):
    tasks, used = [], set()
    genres = list(genre_to_cids.keys())
    while len(tasks) < k and genres:
        g = random.choice(genres)
        pool = [cid for cid in genre_to_cids[g] if cid not in used]
        random.shuffle(pool)
        for _ in range(burst_size):
            if len(tasks) >= k or not pool: break
            cid = pool.pop()
            used.add(cid)
            start_node = (size // 2, size // 2) if USE_FIXED_START_NODE else (np.random.randint(size), np.random.randint(size))
            tasks.append((cid, start_node))
        if not pool:
            genres.remove(g)
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
# デバッグ用レコード（CSV 行にそのまま使う）
# ----------------------------------------------------------------------------
DebugRec = namedtuple("DebugRec",
                      ["method", "cid", "start", "found",
                       "hops", "cost", "path"])

# ----------------------------------------------------------------------------
# 生成タスクの可視化
# ----------------------------------------------------------------------------
def visualize_task_distribution(tasks, title):
    if not tasks:
        print("タスクリストが空です。"); return
    cids = [task[0] for task in tasks]
    counts = pd.Series(cids).value_counts()
    plt.figure(figsize=(15, 6))
    counts.plot(kind='bar', width=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel("コンテンツID")
    plt.ylabel("要求回数")
    if len(counts) > 50:
        plt.xticks([])
    else:
        plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# 既存手法（山代）による探索（ベースライン）
# ----------------------------------------------------------------------------
def search_prop(cache_storage, net_vector_array, size, content_tasks, log_list):
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
                found = True; break
            for nx, ny in get_neighbors(curr[0], curr[1], size):
                if cid in cache_storage[nx][ny]:
                    found = True; break
            if found: break
            min_dist = float('inf'); closest = None
            vect = cont_vector_array[cid - 1]
            for neighbor in get_neighbors(curr[0], curr[1], size):
                if neighbor not in searched_node:
                    dist = np.linalg.norm(vect - net_vector_array[neighbor])
                    if dist < min_dist:
                        min_dist = dist; closest = neighbor
            searched_node.append(curr)
            if all(nb in searched_node for nb in get_neighbors(curr[0], curr[1], size)):
                break
            if closest is None: break
            curr = closest
        if found:
            cache_hit += 1
            log_list.append(DebugRec("BASE", cid, start_node, curr, hops_used, hops_used, []))
        total_hops_used += hops_used
    avg_hops = total_hops_used / len(content_tasks) if content_tasks else 0.0
    return cache_hit, avg_hops

# ----------------------------------------------------------------------------
# ① 従来手法（コンテンツ毎にフェロモン管理）
# ----------------------------------------------------------------------------
def multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks, log_list, pheromone_dict=None):
    results = []
    if pheromone_dict is None:
        pheromone_dict = {}

    for (cid, start_node) in content_tasks:
        if cid not in pheromone_dict:
            pheromone_dict[cid] = initialize_single_pheromone_trails(size)
        pheromone_trails = pheromone_dict[cid]

        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([]); continue

        iteration_data = []
        best_cost = TIMES_TO_SEARCH_HOP

        for _ in range(NUM_ITERATIONS):
            iter_costs, all_paths, all_costs = [], [], []
            for _ in range(NUM_ANTS):
                path, visited = [], set()
                current_node = start_node
                path.append(current_node); visited.add(current_node)
                cost = 0; found = False
                for _ in range(TIMES_TO_SEARCH_HOP):
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True; break
                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed = [n for n in neighbors if n not in visited]
                    if not allowed: break
                    if USE_EPSILON and random.random() < EPSILON:
                        next_node = random.choice(allowed)
                    else:
                        probs, denom = [], 0.0
                        for n in allowed:
                            edge = (current_node, n)
                            score = float((pheromone_trails[edge]) ** ALPHA)
                            probs.append(score); denom += score
                        if denom <= 0:
                            next_node = random.choice(allowed)
                        else:
                            probs = [p / denom for p in probs]
                            next_node = random.choices(allowed, weights=probs)[0]
                    path.append(next_node); visited.add(next_node)
                    cost += 1; current_node = next_node
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True; break
                if found:
                    all_paths.append(path); all_costs.append(cost); iter_costs.append(cost)
                    log_list.append(DebugRec("SINGLE", cid, start_node, current_node, cost, cost, path))
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)

            if all_costs:
                iteration_best = min(all_costs)
                if iteration_best < best_cost: best_cost = iteration_best

            iteration_data.append(iter_costs)

            # 蒸発
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)

            # 追加蒸発（悪い経路）
            if best_cost > 0:
                for path, cost in zip(all_paths, all_costs):
                    delta_p = max(0.0, (cost / best_cost) - 1.0)
                    if delta_p == 0.0: continue
                    rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)
                    for a, b in zip(path[:-1], path[1:]):
                        pheromone_trails[(a, b)] = max(pheromone_trails[(a, b)] * extra_mul, 1e-6)

            # 付加
            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    Q_eff = Q * BOOST if cost <= best_cost else Q
                    delta = Q_eff / cost
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        pheromone_trails[edge] += delta

        results.append(iteration_data)
    return results

# ----------------------------------------------------------------------------
# ② 提案手法（属性毎にフェロモン管理）
#    ※ 正規化ベクトル v_norm を用い、総付加量を一定化
# ----------------------------------------------------------------------------
def multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, log_list):
    results = []
    tag = "ATTRIB_NR"
    global_pheromone_trails = initialize_pheromone_trails(size, N)

    for (cid, start_node) in content_tasks:
        pheromone_trails = global_pheromone_trails

        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([]); continue

        vect = cont_vector_array[cid - 1]
        s = max(1.0, float(vect.sum()))
        v_norm = vect / s  # L1=1 正規化（総量一定）

        iteration_data = []
        best_cost = TIMES_TO_SEARCH_HOP

        for _ in range(NUM_ITERATIONS):
            iter_costs, all_paths, all_costs = [], [], []
            for _ in range(NUM_ANTS):
                path, visited = [], set()
                current_node = start_node
                path.append(current_node); visited.add(current_node)
                cost = 0; found = False

                for _ in range(TIMES_TO_SEARCH_HOP):
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True; break
                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed = [n for n in neighbors if n not in visited]
                    if not allowed: break

                    if USE_EPSILON and random.random() < EPSILON:
                        next_node = random.choice(allowed)
                    else:
                        probs, denom = [], 0.0
                        for n in allowed:
                            edge = (current_node, n)
                            tau_vec = pheromone_trails[edge]
                            score = float(np.dot(np.power(tau_vec, ALPHA), v_norm))
                            score = max(score, 1e-12)  # 数値的フォールバック
                            probs.append(score); denom += score
                        if denom <= 0:
                            next_node = random.choice(allowed)
                        else:
                            probs = [p / denom for p in probs]
                            next_node = random.choices(allowed, weights=probs)[0]

                    path.append(next_node); visited.add(next_node)
                    cost += 1; current_node = next_node
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True; break

                if found:
                    all_paths.append(path); all_costs.append(cost); iter_costs.append(cost)
                    log_list.append(DebugRec(tag, cid, start_node, current_node, cost, cost, path))
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)

            if all_costs:
                iteration_best = min(all_costs)
                if iteration_best < best_cost: best_cost = iteration_best

            iteration_data.append(iter_costs)

            # 蒸発
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            # 追加蒸発（悪い経路）
            if best_cost > 0:
                for path, cost in zip(all_paths, all_costs):
                    delta_p = max(0.0, (cost / best_cost) - 1.0)
                    if delta_p == 0.0: continue
                    rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        pheromone_trails[edge] *= extra_mul
                        pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            # 付加（総量一定）
            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    Q_eff = Q * BOOST if cost <= best_cost else Q
                    delta_vec = (Q_eff / cost) * v_norm
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        pheromone_trails[edge] += delta_vec

        results.append(iteration_data)
    return results

def multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks, log_list):
    return multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, log_list)

# ----------------------------------------------------------------------------
# 統計・可視化ユーティリティ
# ----------------------------------------------------------------------------
def gather_stats_for_content(content_index, single_all_runs, attrib_noreset_all_adapt):
    s_stat = average_iteration_data_across_runs(single_all_runs, content_index)
    an_a_stat = average_iteration_data_across_runs(attrib_noreset_all_adapt, content_index)
    return (s_stat, an_a_stat)

def plot_metrics_for_content(stat_single, stat_attrib_noreset_adpat, content_label):
    if (stat_single is None) or (stat_attrib_noreset_adpat is None):
        print(f"No data for content {content_label}")
        return
    s_med, s_q1, s_q3, s_mean, s_succ = stat_single
    an_a_med, an_a_q1, an_a_q3, an_a_mean, an_a_succ = stat_attrib_noreset_adpat
    its = range(1, NUM_ITERATIONS + 1)
    plt.figure()
    plt.plot(its, s_med, label='Single', color='red', marker='o')
    plt.fill_between(its, s_q1, s_q3, color='red', alpha=0.2)
    plt.plot(its, an_a_med, label='Attrib', color='blue', marker='s')
    plt.fill_between(its, an_a_q1, an_a_q3, color='blue', alpha=0.2)
    plt.title(f"Content {content_label}: Median & Quartiles")
    plt.xlabel("Iteration"); plt.ylabel("Hops"); plt.legend(); plt.show()

    plt.figure()
    plt.plot(its, s_succ, label='Single', color='red', marker='o')
    plt.plot(its, an_a_succ, label='Attrib', color='blue', marker='s')
    plt.title(f"Content {content_label}: Success Rate")
    plt.xlabel("Iteration"); plt.ylabel("Success Rate (%)"); plt.legend(); plt.show()

    plt.figure()
    plt.plot(its, s_mean, label='Single', color='red', marker='o')
    plt.plot(its, an_a_mean, label='Attrib', color='blue', marker='s')
    plt.title(f"Content {content_label}: Average Cost")
    plt.xlabel("Iteration"); plt.ylabel("Average Hops"); plt.legend(); plt.show()

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

def gather_overall_stats(all_runs):
    overall_iter_data = []
    for it in range(NUM_ITERATIONS):
        all_costs = []
        for sim_run in all_runs:
            for content_result in sim_run:
                if len(content_result) > it:
                    all_costs.extend(content_result[it])
        overall_iter_data.append(all_costs)
    return compute_stats_from_costs(overall_iter_data)

def plot_overall_metrics(stats_dict):
    font_size = 18
    its = range(1, NUM_ITERATIONS + 1)

    plt.figure()
    plt.plot(its, stats_dict['Single'][0], label='Single', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][0], label='Attrib', color='blue', marker='s')
    plt.title("Overall: Median"); plt.xlabel("Iteration"); plt.ylabel("Median Hops")
    plt.legend(); plt.xlim(left=0); plt.ylim(bottom=0); plt.tick_params(labelsize=font_size); plt.show()

    plt.figure()
    plt.plot(its, stats_dict['Single'][3], label='proposed method 1', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][3], label='proposed method 2', color='blue', marker='s')
    plt.title("Overall: Average Cost")
    plt.xlabel("Iteration", fontsize=font_size); plt.ylabel("Average number of hops", fontsize=font_size)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14); plt.legend(fontsize=font_size)
    plt.xlim(left=0); plt.ylim(bottom=0); plt.show()

    plt.figure()
    plt.plot(its, stats_dict['Single'][4], label='比較方式', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][4], label='提案方式', color='blue', marker='s')
    plt.title("Overall: Success Rate")
    plt.xlabel("Iteration", fontsize=font_size); plt.ylabel("Success Rate (%)", fontsize=font_size)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14); plt.legend(fontsize=font_size)
    plt.xlim(left=0); plt.ylim(bottom=0); plt.show()

# --- 追加：タスク順の「各タスクの1イテ目」曲線を可視化（転移の効果を直視） ---
def first_iter_curve(all_runs, agg='median'):
    curves = []
    for run in all_runs:
        ys = []
        for task_res in run:
            if not task_res:
                ys.append(np.nan); continue
            costs_iter0 = task_res[0]  # そのタスクの1イテ目
            if len(costs_iter0) == 0:
                ys.append(np.nan)
            else:
                ys.append(float(np.median(costs_iter0) if agg == 'median' else np.mean(costs_iter0)))
        curves.append(ys)
    arr = np.array(curves, dtype=float)
    return np.nanmedian(arr, axis=0)

def plot_first_iter_task_curve(single_all_runs, attrib_all_runs):
    s = first_iter_curve(single_all_runs)
    a = first_iter_curve(attrib_all_runs)
    xs = range(1, len(s) + 1)
    plt.figure()
    plt.plot(xs, s, label='Single (iter-1)', marker='o')
    plt.plot(xs, a, label='Attrib (iter-1)', marker='s')
    plt.xlabel("Task index (要求順)")
    plt.ylabel("Median hops at task's 1st iteration")
    plt.title("Transfer Effect: 1st-iteration vs Task Order")
    plt.legend()
    plt.show()

# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def main():
    size = 50
    cache_storage, net_vector_array = cache_prop(size)

    if TASK_MODE == 'z':
        content_tasks = generate_multi_contents_tasks_z(size, k=NUM_CONTENTS_TO_SEARCH)
    elif TASK_MODE == 's':
        content_tasks = generate_multi_contents_tasks_s(size, k=NUM_CONTENTS_TO_SEARCH)
    elif TASK_MODE == 's_nodup':
        content_tasks = generate_multi_contents_tasks_s_nodup(size, k=NUM_CONTENTS_TO_SEARCH)
    else:
        content_tasks = generate_multi_contents_tasks_r(size, k=NUM_CONTENTS_TO_SEARCH)

    print("Generated Content Tasks:", content_tasks)

    debug_logs = []
    single_all_runs = []
    attrib_noreset_all_adapt = []

    print("コンテンツ生成の分布を可視化します...")
    visualize_task_distribution(content_tasks, "Task Distribution")

    for _sim in range(TIME_TO_SIMULATE):
        single_res = multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks, debug_logs)
        attrib_noreset_res = multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks, debug_logs)
        single_all_runs.append(single_res)
        attrib_noreset_all_adapt.append(attrib_noreset_res)

    cache_hit, avg_hops = search_prop(cache_storage, net_vector_array, size, content_tasks, debug_logs)
    success_rate = (cache_hit / len(content_tasks)) * 100

    # CSV 出力
    df_out = pd.DataFrame(debug_logs)
    df_out.to_csv("debug_log.csv", index=False, encoding="utf-8-sig")
    print(f"\n>>> {len(df_out)} 行を書き出しました → debug_log.csv")

    print("\n=== Existing Method (Same Tasks) ===")
    print(f"  - #Tasks : {len(content_tasks)}")
    print(f"  - Cache Hit: {cache_hit}")
    print(f"  - Success Rate: {success_rate:.2f}%")
    print(f"  - Avg Hops: {avg_hops:.2f}")

    # 総合統計
    overall_stats = {
        'Single': gather_overall_stats(single_all_runs),
        'NR-Adapt': gather_overall_stats(attrib_noreset_all_adapt),
    }
    plot_overall_metrics(overall_stats)

    # ★ 転移効果の“決定打”可視化
    plot_first_iter_task_curve(single_all_runs, attrib_noreset_all_adapt)

if __name__ == "__main__":
    main()


