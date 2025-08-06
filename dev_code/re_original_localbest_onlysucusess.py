# 山代法＋比較手法（コンテンツ毎にフェロモン管理）＋提案手法（属性毎にフェロモン管理）
#比較手法、提案手法どちらにもローカルベストの探索を実装
#探索性能は探索成功時のみ

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import namedtuple  
from collections import defaultdict

# ----------------------------------------------------------------------------
# パラメータ設定
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 3
NUM_CONTENTS_TO_SEARCH = 100  # 探索するコンテンツ数
NUM_ANTS = 10
NUM_ITERATIONS = 10

ALPHA = 1.0   # フェロモンの指数
BETA = 1.0    # SOM類似度の指数
Q = 100

# --- 可変蒸発・付加パラメータ ----------------------------------------
RHO      = 0.10   # ベース蒸発率 (従来の RHO と同値で可)
RHO_MAX   = 0.9   # 蒸発率の上限
LAMBDA_R  = 1   # 拡張係数 λ   (0.1〜0.5 で様子を見ると良い),1だとこの変数がないと同じ
BOOST = 5 # フェロモンの付加量を調整するための係数
# ---------------------------------------------------------------

TIMES_TO_SEARCH_HOP = 100
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

USE_FIXED_START_NODE = False

# ε-greedy戦略用パラメータ（リセットなしの場合のみ使用）
USE_EPSILON = True
EPSILON = 0.01  # 10%の確率でランダム選択


# CSVファイルから属性ベクトルを準備
file_path = "500_movies.csv"  # 適宜修正
#file_path = "10000_movies.csv"  # 適宜修正
df = pd.read_csv(file_path)
N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec) for vec in cont_vector]

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
# コンテンツ探索タスク生成
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
        # 成功した探索のコストのみを抽出
        successful_costs = [c for c in costs if c < TIMES_TO_SEARCH_HOP]
        # 成功した探索があればその平均を、なければ上限値を平均ホップ数とする
        if successful_costs:
            means.append(np.mean(successful_costs))
        else:
            means.append(TIMES_TO_SEARCH_HOP)
        total = len(costs)
        success_count = sum(1 for c in costs if c < TIMES_TO_SEARCH_HOP)
        success_rates.append((success_count / total) * 100)
    return medians, q1s, q3s, means, success_rates

# ----------------------------------------------------------------------------
# デバッグ用レコード（CSV 行にそのまま使う）        ★
# ----------------------------------------------------------------------------
DebugRec = namedtuple("DebugRec",
                      ["method", "cid", "start", "found",
                       "hops", "cost", "path"])


# ----------------------------------------------------------------------------
# 既存手法による探索
# ----------------------------------------------------------------------------
def search_prop(cache_storage, net_vector_array, size, content_tasks, log_list):
    cache_hit = 0
    #成功した探索の合計ホップ数を記録する変数を追加
    total_successful_hops_used = 0
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
            #デバッグ用レコードを追加
            total_successful_hops_used += hops_used
            log_list.append(DebugRec("BASE", cid, start_node, curr, hops_used, hops_used, []))
    avg_hops = total_successful_hops_used / cache_hit if cache_hit > 0 else TIMES_TO_SEARCH_HOP
    return cache_hit, avg_hops

# ----------------------------------------------------------------------------
# ① 従来手法（コンテンツ毎にフェロモン管理）
# ----------------------------------------------------------------------------
def multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks, log_list, pheromone_dict=None):
    results = []
    if pheromone_dict is None:
        pheromone_dict = {}

    for (cid, start_node) in content_tasks:
        # ID用のフェロもの有無を確認
        if cid not in pheromone_dict:
            # 存在しない場合、初期化
            pheromone_dict[cid] = initialize_single_pheromone_trails(size)
        # 存在する場合は再利用
        pheromone_trails = pheromone_dict[cid]
        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([])
            continue
        vect = cont_vector_array[cid - 1]
        iteration_data = []
        best_cost = TIMES_TO_SEARCH_HOP  # 最良結果の記録（固定更新なので更新は記録のみ）
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
                    # ε-greedyはリセットなしの場合のみ適用
                    if  USE_EPSILON and random.random() < EPSILON:
                        next_node = random.choice(allowed)
                    else:
                        probs = []
                        denom = 0
                        for n in allowed:
                            edge = (current_node, n)
                            tau = pheromone_trails[edge]
                            dist = np.linalg.norm(vect - net_vector_array[n])
                            eta = 1.0 / (dist + 1e-6)
                            if np.sum(vect) > 0:
                                A_ij = np.sum((tau ** ALPHA) * vect) / np.sum(vect)
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
                    #デバック用レコードの追加
                    log_list.append(DebugRec("SINGLE", cid, start_node,current_node, cost, cost, path))
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)
            best_iter_cost = min(all_costs) if all_costs else TIMES_TO_SEARCH_HOP
            # 前更新：最良結果の記録（best_costは記録のみ）
            if all_costs:
                iteration_best = min(all_costs)
                if iteration_best < best_cost:
                    best_cost = iteration_best
            iteration_data.append(iter_costs)
            # フェロモンの蒸発更新
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)
            #　ローカルの最良結果に応じて、相対的に悪い経路のフェロモンを追加で蒸発
            if best_cost > 0:
                for path, cost in zip(all_paths, all_costs):
                    delta_p = max(0.0, (cost / best_cost) - 1.0)
                    if delta_p == 0.0:
                        continue  # 最良パス
                    rho_eff   = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)
                    for a, b in zip(path[:-1], path[1:]):
                        pheromone_trails[(a, b)] = max(
                            pheromone_trails[(a, b)] * extra_mul, 1e-6)

            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    if cost <= best_cost:
                        Q_eff = Q * BOOST
                    else:
                        Q_eff = Q
                    delta = Q_eff/ cost
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta

        results.append(iteration_data)
    return results

# ----------------------------------------------------------------------------
# ② 提案手法（属性毎にフェロモン管理）
# ----------------------------------------------------------------------------
def multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, log_list):
    results = []
    tag =  "ATTRIB_NR"
    
    global_pheromone_trails = initialize_pheromone_trails(size, N)

    for (cid, start_node) in content_tasks:
        pheromone_trails = global_pheromone_trails
        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([])
            continue
        vect = cont_vector_array[cid - 1]
        iteration_data = []
        best_cost = TIMES_TO_SEARCH_HOP  # 最良結果の記録（固定更新なので更新は記録のみ）
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
                    # ε-greedyはリセットなしの場合のみ適用
                    if  USE_EPSILON and random.random() < EPSILON:
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
                    #デバック用コードの追加
                    log_list.append(DebugRec(tag, cid, start_node,current_node, cost, cost, path))
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)
                #　イタレーションベストを求める
            best_iter_cost = min(all_costs) if all_costs else TIMES_TO_SEARCH_HOP
            # 前更新：最良結果の記録（best_costは記録のみ）
            if all_costs:
                iteration_best = min(all_costs)
                if iteration_best < best_cost:
                    best_cost = iteration_best
            iteration_data.append(iter_costs)
            # フェロモンの蒸発更新
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)
            
            #　ローカルの最良結果に応じて、相対的に悪い経路のフェロモンを追加で蒸発
            if best_cost >0:
                for path, cost in zip(all_paths, all_costs):
                    # best_costで全探索の最良結果、best_iter_costでイタレーションの最良結果
                    delta_p = max(0.0, (cost / best_cost) - 1.0)
                    if delta_p == 0.0:
                        continue   # 最良パスなので追加蒸発なし
                    rho_eff   = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)   # 追加蒸発分
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] *= extra_mul
                        pheromone_trails[edge]  = np.maximum(pheromone_trails[edge], 1e-6)

            # ローカルの最良結果に応じて、フェロモンの付加量を調整
            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    #リセットなしの場合、最良パスのコストよりも良い経路はフェロモンを増加
                    if  cost <= best_cost:
                        Q_eff = Q * BOOST
                    else:
                        Q_eff = Q
                    delta = (Q_eff * vect) / (cost * np.sum(vect))
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta
        results.append(iteration_data)
    return results



def multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks, log_list):
    return multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, log_list)


def gather_stats_for_content(content_index, single_all_runs,  attrib_noreset_all_adapt):
    s_stat = average_iteration_data_across_runs(single_all_runs, content_index)
    an_a_stat = average_iteration_data_across_runs(attrib_noreset_all_adapt, content_index)
    return (s_stat, an_a_stat)

def plot_metrics_for_content(stat_single,  stat_attrib_noreset_adpat, content_label):
    if (stat_single is None)  or (stat_attrib_noreset_adpat is None):
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
    plt.xlabel("Iteration")
    plt.ylabel("Hops")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(its, s_succ, label='Single', color='red', marker='o')
    plt.plot(its, an_a_succ, label='Attrib', color='blue', marker='s')
    plt.title(f"Content {content_label}: Success Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(its, s_mean, label='Single', color='red', marker='o')
    plt.plot(its, an_a_mean, label='Attrib', color='blue', marker='s')
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

# --- 新規関数: 総合結果の統計量を計算する ---
def gather_overall_stats(all_runs):
    # all_runs: 各シミュレーション毎の結果（各コンテンツごとにイテレーションごとのコストリスト）
    overall_iter_data = []
    for it in range(NUM_ITERATIONS):
        all_costs = []
        for sim_run in all_runs:
            # sim_runは各コンテンツ結果のリスト
            for content_result in sim_run:
                if len(content_result) > it:
                    all_costs.extend(content_result[it])
        overall_iter_data.append(all_costs)
    return compute_stats_from_costs(overall_iter_data)

def plot_overall_metrics(stats_dict):
    # stats_dict: {'Single': stats, 'AttribReset': stats, 'NR-Adapt': stats, 'NR-Base': stats}
    font_size = 18
    its = range(1, NUM_ITERATIONS + 1)
    plt.figure()
    plt.plot(its, stats_dict['Single'][0], label='Single', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][0], label='Attrib', color='blue', marker='s')
    plt.title("Overall: Median")
    plt.xlabel("Iteration")
    plt.ylabel("Median Hops")
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tick_params(labelsize=font_size)
    plt.show()

    plt.figure()
    plt.plot(its, stats_dict['Single'][3], label='proposed method 1', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][3], label='proposed method 2', color='blue', marker='s')
    plt.title("Overall: Average Cost")
    plt.xlabel("Iteration", fontsize=font_size)
    plt.ylabel("Average number of hops", fontsize=font_size)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14)  
    plt.legend(fontsize=font_size)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

    plt.figure()
    plt.plot(its, stats_dict['Single'][4], label='比較方式', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][4], label='提案方式', color='blue', marker='s')
    plt.title("Overall: Success Rate")
    plt.xlabel("Iteration", fontsize=font_size)
    plt.ylabel("Success Rate (%)", fontsize=font_size)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.legend(fontsize=font_size)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

def main():
    size = 50
    cache_storage, net_vector_array = cache_prop(size)
    content_tasks = generate_multi_contents_tasks(size, k=NUM_CONTENTS_TO_SEARCH)
    print("Generated Content Tasks:", content_tasks)
    
    debug_logs = []

    single_all_runs = []
    attrib_noreset_all_adapt = []

    for _sim in range(TIME_TO_SIMULATE):
        single_res = multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks, debug_logs)
        attrib_noreset_res = multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks, debug_logs)

        single_all_runs.append(single_res)
        attrib_noreset_all_adapt.append(attrib_noreset_res)

    cache_hit, avg_hops = search_prop(cache_storage, net_vector_array, size, content_tasks, debug_logs)
    success_rate = (cache_hit / len(content_tasks)) * 100

    #CSV 出力
    df = pd.DataFrame(debug_logs)
    df.to_csv("debug_log.csv", index=False, encoding="utf-8-sig")
    print(f"\n>>> {len(df)} 行を書き出しました → debug_log.csv")

    print("\n=== Existing Method (Same Tasks) ===")
    print(f"  - #Tasks : {len(content_tasks)}")
    print(f"  - Cache Hit: {cache_hit}")
    print(f"  - Success Rate: {success_rate:.2f}%")
    print(f"  - Avg Hops: {avg_hops:.2f}")
    for content_idx in range(NUM_CONTENTS_TO_SEARCH):
        print(f"=== [Comparison] Content #{content_idx+1} ===")

        stat_s, stat_an_a = gather_stats_for_content(content_idx, single_all_runs, attrib_noreset_all_adapt)
        plot_metrics_for_content(stat_s, stat_an_a, content_label=content_idx+1)
    # 総合結果の統計量を計算してプロット
    overall_stats = {
        'Single': gather_overall_stats(single_all_runs),
        'NR-Adapt': gather_overall_stats(attrib_noreset_all_adapt),
    }
    plot_overall_metrics(overall_stats)

if __name__ == "__main__":
    main()