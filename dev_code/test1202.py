# 山代法＋比較手法（コンテンツ毎にフェロモン管理）＋提案手法（属性毎にフェロモン管理）
# ICN拡張：探索成功時に要求源ノードにキャッシュを配置＋最短経路でベクトル更新

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
TIME_TO_SIMULATE = 1 # シミュレーション回数
NUM_CONTENTS_TO_SEARCH = 10  # 探索するコンテンツ数
NUM_ANTS = 10
NUM_ITERATIONS = 10 

ALPHA_START = 1.0
BETA_START = 10.0
ALPHA_END = 1.0
BETA_END = 1.0
Q = 100

RHO      = 0.10
RHO_MAX   = 0.9
LAMBDA_R  = 1
BOOST = 5

TIMES_TO_SEARCH_HOP = 100
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1
NUM_MAPS = 5

USE_FIXED_START_NODE = False
USE_EPSILON = True
EPSILON = 0.01
EPSILON_SOM = 0.20

# CSVファイルから属性ベクトルを準備
file_path = "500_movies.csv" 
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

# --- ICN拡張用ヘルパー関数 ---
def apply_icn_update(cache_storage, net_vector_arrays, cid, vect, start_node, path_from_found_to_start):
    """
    探索成功時のICN処理を一括実行する
    path_from_found_to_start: [発見場所 -> ... -> 要求元] の順に並んだ座標リスト
    """
    # 1. キャッシュ格納 (要求元ノードへ)
    if cid not in cache_storage[start_node[0]][start_node[1]]:
        cache_storage[start_node[0]][start_node[1]].append(cid)

    # 2. ベストマップ選択 (要求元ノードのベクトルとコンテンツの類似度で判定)
    best_map_idx = -1
    min_dist = float('inf')
    
    for i, net_vec in enumerate(net_vector_arrays):
        node_vec = net_vec[start_node] 
        dist = np.linalg.norm(vect - node_vec)
        if dist < min_dist:
            min_dist = dist
            best_map_idx = i
            
    # 3. マップの重み付け更新
    target_map = net_vector_arrays[best_map_idx]
    total_hops = len(path_from_found_to_start)
    
    tmp = 0
    for node in path_from_found_to_start:
        tmp += 1
        alpha = LEARNING_RATE * (tmp / total_hops)
        target_map[node] += alpha * (vect - target_map[node])

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

def initialize_single_pheromone_trails(size):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            for neighbor in get_neighbors(x, y, size):
                pheromone_trails[(curr_node, neighbor)] = 1.0
    return pheromone_trails

def initialize_pheromone_trails(size, num_attributes):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            for neighbor in get_neighbors(x, y, size):
                pheromone_trails[(curr_node, neighbor)] = np.ones(num_attributes)
    return pheromone_trails

def cache_prop(size):
    cache_storage = get_init_cache_storage(size)
    net_vector_arrays = [get_init_network_vector(size) for _ in range(NUM_MAPS)]
    placements_per_map = TIME_TO_CACHE_PER_CONTENT // NUM_MAPS
    cache_hops = TIMES_TO_CACHE_HOP
    alpha_zero = LEARNING_RATE

    for map_index in range(NUM_MAPS):
        current_net_vector_array = net_vector_arrays[map_index]
        for _ in range(placements_per_map):
            for cid in range(1, cont_num + 1):
                curr = (np.random.randint(size), np.random.randint(size))
                vect = cont_vector_array[cid - 1]
                hoped_node = []
                for _2 in range(cache_hops):
                    min_dist = float('inf')
                    closest = None
                    for neighbor in get_neighbors(curr[0], curr[1], size):
                        if neighbor not in hoped_node:
                            neighbor_vect = current_net_vector_array[neighbor]
                            dist = np.linalg.norm(vect - neighbor_vect)
                            if dist < min_dist:
                                min_dist = dist
                                closest = neighbor
                    if closest is None: break
                    hoped_node.append(curr)
                    curr = closest
                hoped_node.append(curr)
                cache_storage[curr[0]][curr[1]].append(cid)
                tmp = 0
                total_hops = len(hoped_node)
                for node in hoped_node:
                    tmp += 1
                    alpha = alpha_zero * (tmp / total_hops)
                    current_net_vector_array[node] += alpha * (vect - current_net_vector_array[node])
    return cache_storage, net_vector_arrays

# ----------------------------------------------------------------------------
# コンテンツ探索タスク生成 (Randomのみ使用)
# ----------------------------------------------------------------------------
def generate_multi_contents_tasks_r(size, k=NUM_CONTENTS_TO_SEARCH):
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
    means, success_rates =  [], []
    for costs in iteration_costs_data:
        if costs:
            means.append(np.mean(costs))
            total = len(costs)
            success_count = sum(1 for c in costs if c < TIMES_TO_SEARCH_HOP)
            success_rates.append((success_count / total) * 100)
        else:
            means.append(np.nan)
            success_rates.append(0.0)
    return means, success_rates

# ----------------------------------------------------------------------------
# デバッグ用レコード
# ----------------------------------------------------------------------------
DebugRec = namedtuple("DebugRec", ["method", "cid", "start", "found", "hops", "cost", "path"])

# ----------------------------------------------------------------------------
# 既存手法による探索
# ----------------------------------------------------------------------------
def search_prop(cache_storage, net_vector_arrays, size, content_tasks, log_list):
    cache_hit = 0
    total_hops_used = 0
    for (cid, start_node) in content_tasks:
        hops_per_map = []
        found_on_any_map = False
        best_path_in_task = None
        min_hops_in_task = float('inf')

        for map_index in range(NUM_MAPS):
            current_net_vector_array = net_vector_arrays[map_index]
            searched_node = []
            curr = start_node
            searched_node.append(curr) # スタート地点もパスに含む
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
                        curr = (nx, ny) # 発見場所へ移動
                        searched_node.append(curr)
                        break
                if found: break
                
                min_dist = float('inf')
                closest = None
                vect = cont_vector_array[cid - 1]
                for neighbor in get_neighbors(curr[0], curr[1], size):
                    if neighbor not in searched_node:
                        dist = np.linalg.norm(vect - current_net_vector_array[neighbor])
                        if dist < min_dist:
                            min_dist = dist
                            closest = neighbor
                if closest is None: break
                curr = closest
                searched_node.append(curr)

            if found:
                found_on_any_map = True
                hops_per_map.append(hops_used)
                if hops_used < min_hops_in_task:
                    min_hops_in_task = hops_used
                    best_path_in_task = searched_node[::-1] # 逆順(発見->要求元)
            else:
                hops_per_map.append(TIMES_TO_SEARCH_HOP)

        min_hops = min(hops_per_map)
        if found_on_any_map:
            cache_hit += 1
            if best_path_in_task:
                vect = cont_vector_array[cid - 1]
                apply_icn_update(cache_storage, net_vector_arrays, cid, vect, start_node, best_path_in_task)

        total_hops_used += min_hops

    avg_hops = total_hops_used / len(content_tasks) if content_tasks else 0.0
    return cache_hit, avg_hops

# ----------------------------------------------------------------------------
# ① 提案手法1（コンテンツ毎にフェロモン管理）
# ----------------------------------------------------------------------------
def multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, net_vector_arrays, size, content_tasks, log_list, pheromone_dict=None):
    results = []
    if pheromone_dict is None:
        pheromone_dict = {}

    for (cid, start_node) in content_tasks:
        if cid not in pheromone_dict:
            pheromone_dict[cid] = initialize_single_pheromone_trails(size)
        pheromone_trails = pheromone_dict[cid]
        
        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([])
            continue
            
        vect = cont_vector_array[cid - 1]
        iteration_data = []
        best_cost = TIMES_TO_SEARCH_HOP
        global_best_path_in_task = None
        global_min_hops_in_task = float('inf')

        for t in range(NUM_ITERATIONS):
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
                
                if t == 0:
                    # --- Iter.0 純SOM (ε-greedy) ---
                    searched_node = []
                    curr = start_node
                    # pathには既にstart_nodeが入っている
                    for _ in range(TIMES_TO_SEARCH_HOP):
                        cost += 1
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True
                            break
                        hit_neighbor = False
                        for nx, ny in get_neighbors(curr[0], curr[1], size):
                            if cid in cache_storage[nx][ny]:
                                found = True
                                hit_neighbor = True
                                curr = (nx, ny)
                                path.append(curr)
                                break
                        if found: break

                        allowed_neighbors = [n for n in get_neighbors(curr[0], curr[1], size) if n not in searched_node]
                        if not allowed_neighbors: break

                        min_dist = float('inf')
                        closest = None
                        for neighbor in allowed_neighbors:
                            dist = np.linalg.norm(vect - net_vector_array[neighbor])
                            if dist < min_dist:
                                min_dist = dist
                                closest = neighbor
                        
                        if closest is None: break
                        
                        next_node = None
                        if USE_EPSILON and random.random() < EPSILON_SOM:
                            next_node = random.choice(allowed_neighbors)
                        else:
                            next_node = closest

                        searched_node.append(curr)
                        curr = next_node
                        path.append(curr)                  
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True
                            break
                    current_node = curr
                else:
                    # --- Iter > 0 ACO ---
                    for _ in range(TIMES_TO_SEARCH_HOP):
                        if cid in cache_storage[current_node[0]][current_node[1]]:
                            found = True
                            break
                        neighbors = get_neighbors(current_node[0], current_node[1], size)
                        allowed = [n for n in neighbors if n not in visited]
                        if not allowed: break
                        
                        if  USE_EPSILON and random.random() < EPSILON:
                            next_node = random.choice(allowed)
                        else:
                            if NUM_ITERATIONS >= 1:
                                ALPHA = ALPHA_START + (ALPHA_END - ALPHA_START) * (t / (NUM_ITERATIONS - 1))
                                BETA  = BETA_START  + (BETA_END  - BETA_START)  * (t / (NUM_ITERATIONS - 1))
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
                            if denom == 0: break
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
                    if cost < global_min_hops_in_task:
                        global_min_hops_in_task = cost
                        global_best_path_in_task = path[::-1] # 逆順
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)
            
            # フェロモン更新処理
            best_iter_cost = min(all_costs) if all_costs else TIMES_TO_SEARCH_HOP
            if all_costs:
                iteration_best = min(all_costs)
                if iteration_best < best_cost:
                    best_cost = iteration_best
            iteration_data.append(iter_costs)
            
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)
            
            if best_cost > 0:
                for p, c in zip(all_paths, all_costs):
                    delta_p = max(0.0, (c / best_cost) - 1.0)
                    if delta_p == 0.0: continue
                    rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)
                    for a, b in zip(p[:-1], p[1:]):
                        pheromone_trails[(a, b)] = max(pheromone_trails[(a, b)] * extra_mul, 1e-6)

            for p, c in zip(all_paths, all_costs):
                if c > 0:
                    Q_eff = Q * BOOST if c <= best_cost else Q
                    delta = Q_eff/ c
                    for i in range(len(p) - 1):
                        edge = (p[i], p[i+1])
                        pheromone_trails[edge] += delta

        # タスク完了後、ベストパスでICN更新
        if global_best_path_in_task:
            apply_icn_update(cache_storage, net_vector_arrays, cid, vect, start_node, global_best_path_in_task)
        results.append(iteration_data)
    return results

# ----------------------------------------------------------------------------
# ② 提案手法2（属性毎にフェロモン管理）
# ----------------------------------------------------------------------------
def multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, net_vector_arrays, size, content_tasks, log_list):
    results = []
    global_pheromone_trails = initialize_pheromone_trails(size, N)

    for (cid, start_node) in content_tasks:
        pheromone_trails = global_pheromone_trails # 共有フェロモンを使用
        content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
        if not content_nodes:
            results.append([])
            continue
        vect = cont_vector_array[cid - 1]
        iteration_data = []
        best_cost = TIMES_TO_SEARCH_HOP
        global_best_path_in_task = None
        global_min_hops_in_task = float('inf')

        for t in range(NUM_ITERATIONS):
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

                if t == 0:
                    # --- Iter.0 純SOM (ε-greedy) ---
                    searched_node = []
                    curr = start_node
                    for _ in range(TIMES_TO_SEARCH_HOP):
                        cost += 1
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True; break
                        hit_neighbor = False
                        for nx, ny in get_neighbors(curr[0], curr[1], size):
                            if cid in cache_storage[nx][ny]:
                                found = True; hit_neighbor = True; curr = (nx, ny); path.append(curr); break
                        if found: break

                        allowed_neighbors = [n for n in get_neighbors(curr[0], curr[1], size) if n not in searched_node]
                        if not allowed_neighbors: break

                        min_dist = float('inf'); closest = None
                        for neighbor in allowed_neighbors:
                            dist = np.linalg.norm(vect - net_vector_array[neighbor])
                            if dist < min_dist: min_dist = dist; closest = neighbor
                        
                        if closest is None: break
                        next_node = random.choice(allowed_neighbors) if (USE_EPSILON and random.random() < EPSILON_SOM) else closest
                        searched_node.append(curr)
                        curr = next_node
                        path.append(curr)                  
                        if cid in cache_storage[curr[0]][curr[1]]: found = True; break
                    current_node = curr
                else:
                    # --- Iter > 0 ACO ---
                    for _ in range(TIMES_TO_SEARCH_HOP):
                        if cid in cache_storage[current_node[0]][current_node[1]]: found = True; break
                        neighbors = get_neighbors(current_node[0], current_node[1], size)
                        allowed = [n for n in neighbors if n not in visited]
                        if not allowed: break
                        
                        if USE_EPSILON and random.random() < EPSILON:
                            next_node = random.choice(allowed)
                        else:
                            if NUM_ITERATIONS >= 1:
                                ALPHA = ALPHA_START + (ALPHA_END - ALPHA_START) * (t / (NUM_ITERATIONS - 1))
                                BETA  = BETA_START  + (BETA_END  - BETA_START)  * (t / (NUM_ITERATIONS - 1))
                            probs = []
                            denom = 0
                            for n in allowed:
                                edge = (current_node, n)
                                tau_list = pheromone_trails[edge]
                                dist = np.linalg.norm(vect - net_vector_array[n])
                                eta = 1.0 / (dist + 1e-6)
                                A_ij = np.sum((tau_list ** ALPHA) * vect) / np.sum(vect) if np.sum(vect) > 0 else 0
                                score = A_ij * (eta ** BETA)
                                probs.append(score); denom += score
                            if denom == 0: break
                            probs = [p / denom for p in probs]
                            next_node = random.choices(allowed, weights=probs)[0]
                        path.append(next_node); visited.add(next_node); cost += 1; current_node = next_node
                        if cid in cache_storage[current_node[0]][current_node[1]]: found = True; break
                
                if found:
                    all_paths.append(path); all_costs.append(cost); iter_costs.append(cost)
                    if cost < global_min_hops_in_task:
                        global_min_hops_in_task = cost
                        global_best_path_in_task = path[::-1] # 逆順
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)
            
            # フェロモン更新
            best_iter_cost = min(all_costs) if all_costs else TIMES_TO_SEARCH_HOP
            if all_costs:
                iteration_best = min(all_costs)
                if iteration_best < best_cost: best_cost = iteration_best
            iteration_data.append(iter_costs)
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)
            
            if best_cost > 0:
                for p, c in zip(all_paths, all_costs):
                    delta_p = max(0.0, (c / best_cost) - 1.0)
                    if delta_p == 0.0: continue
                    rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)
                    for i in range(len(p) - 1):
                        edge = (p[i], p[i+1])
                        pheromone_trails[edge] *= extra_mul
                        pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            for p, c in zip(all_paths, all_costs):
                if c > 0:
                    Q_eff = Q * BOOST if c <= best_cost else Q
                    delta = (Q_eff * vect) / (c * np.sum(vect))
                    for i in range(len(p) - 1):
                        edge = (p[i], p[i+1])
                        pheromone_trails[edge] += delta
        
        # タスク完了後更新
        if global_best_path_in_task:
            apply_icn_update(cache_storage, net_vector_arrays, cid, vect, start_node, global_best_path_in_task)
        results.append(iteration_data)
    return results

def multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, net_vector_arrays, size, content_tasks, log_list):
    return multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, net_vector_arrays, size, content_tasks, log_list)

# ----------------------------------------------------------------------------
# 複数マップ並列結果のマージ
# ----------------------------------------------------------------------------
def merge_parallel_results(map_results_list):
    if not map_results_list: return []
    num_maps = len(map_results_list)
    if num_maps == 0: return []
    num_tasks = len(map_results_list[0])
    if num_tasks == 0: return []
    num_iters = len(map_results_list[0][0])
    
    final_results = [ [ [] for _ in range(num_iters)] for _ in range(num_tasks)]
    
    for task_idx in range(num_tasks):
        for iter_idx in range(num_iters):
            merged_costs = []
            for map_idx in range(num_maps):
                if len(map_results_list[map_idx]) > task_idx and len(map_results_list[map_idx][task_idx]) > iter_idx:
                    merged_costs.extend(map_results_list[map_idx][task_idx][iter_idx])
            final_results[task_idx][iter_idx] = merged_costs
    return final_results

# ----------------------------------------------------------------------------
# 統計・グラフ描画
# ----------------------------------------------------------------------------
def gather_overall_stats(all_runs):
    overall_iter_costs = []
    overall_iter_success_rates = []
    for it in range(NUM_ITERATIONS):
        all_costs_for_this_iter = []
        all_success_rates_for_this_iter = []
        for sim_run in all_runs:
            for content_result in sim_run:
                if len(content_result) > it:
                    costs_50_ants = content_result[it]
                    if costs_50_ants: 
                        team_costs_list = [costs_50_ants[i:i + NUM_ANTS] for i in range(0, len(costs_50_ants), NUM_ANTS)]
                        team_avg_costs = []
                        team_success_rates_list = []
                        for team_costs in team_costs_list:
                            if team_costs:
                                team_avg_costs.append(np.mean(team_costs))
                                success_count = sum(1 for c in team_costs if c < TIMES_TO_SEARCH_HOP)
                                team_success_rates_list.append((success_count / len(team_costs)) * 100.0)
                            else:
                                team_avg_costs.append(TIMES_TO_SEARCH_HOP)
                                team_success_rates_list.append(0.0)
                        
                        best_team_index = np.argmin(team_avg_costs)
                        best_team_avg_cost = team_avg_costs[best_team_index]
                        best_team_success_rate = team_success_rates_list[best_team_index]
                        all_costs_for_this_iter.append(best_team_avg_cost)
                        all_success_rates_for_this_iter.append(best_team_success_rate)
                    else:
                        all_costs_for_this_iter.append(TIMES_TO_SEARCH_HOP)
                        all_success_rates_for_this_iter.append(0.0)

        overall_iter_costs.append(all_costs_for_this_iter)
        overall_iter_success_rates.append(all_success_rates_for_this_iter)

    cost_stats = compute_stats_from_costs(overall_iter_costs)
    success_stats = compute_stats_from_costs(overall_iter_success_rates)
    # [0]=mean cost, [1]=mean success rate
    return cost_stats[0], success_stats[0]

def plot_overall_metrics(stats_dict):
    font_size = 18
    its = range(1, NUM_ITERATIONS + 1)

    plt.figure()
    plt.plot(its, stats_dict['Single'][0], label='提案手法1(ID毎)', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][0], label='提案手法2(属性毎)', color='blue', marker='s')
    plt.title("Overall: Average Cost")
    plt.xlabel("Iteration", fontsize=font_size)
    plt.ylabel("Average number of hops", fontsize=font_size)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.legend(fontsize=font_size)
    plt.xlim(left=0); plt.ylim(bottom=0)
    plt.show()

    plt.figure()
    plt.plot(its, stats_dict['Single'][1], label='提案手法1(ID毎)', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][1], label='提案手法2(属性毎)', color='blue', marker='s')
    plt.title("Overall: Success Rate")
    plt.xlabel("Iteration", fontsize=font_size)
    plt.ylabel("Success Rate (%)", fontsize=font_size)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.legend(fontsize=font_size)
    plt.xlim(left=0); plt.ylim(0,100)
    plt.show()

def main():
    size = 50
    cache_storage, net_vector_arrays = cache_prop(size)
    content_tasks = generate_multi_contents_tasks_r(size, k=NUM_CONTENTS_TO_SEARCH)
    print("Generated Content Tasks:", content_tasks)
    
    debug_logs = []
    single_all_runs = []
    attrib_noreset_all_adapt = []

    # visualize_task_distribution(content_tasks, "Task Distribution") # 毎回止めると面倒なのでコメントアウト

    single_pheromone_dict_list = [{} for _ in range(NUM_MAPS)]

    for _sim in range(TIME_TO_SIMULATE):
        print(f"\n=== Simulation #{_sim + 1} ===")
        sim_single_results_list = []
        sim_attrib_results_list = []

        for map_index in range(NUM_MAPS):
            print(f"  --- Map #{map_index} ---")
            current_map_vector = net_vector_arrays[map_index]
            current_pheromone_dict = single_pheromone_dict_list[map_index]
            
            single_res = multi_contents_single_pheromone_with_reset(cache_storage, current_map_vector, net_vector_arrays, size, content_tasks, debug_logs, current_pheromone_dict)
            attrib_res = multi_contents_attrib_pheromone_no_reset(cache_storage, current_map_vector, net_vector_arrays, size, content_tasks, debug_logs)

            sim_single_results_list.append(single_res)
            sim_attrib_results_list.append(attrib_res)

        merged_single_res = merge_parallel_results(sim_single_results_list)
        merged_attrib_res = merge_parallel_results(sim_attrib_results_list)
        single_all_runs.append(merged_single_res)
        attrib_noreset_all_adapt.append(merged_attrib_res)

    cache_hit, avg_hops = search_prop(cache_storage, net_vector_arrays, size, content_tasks, debug_logs)
    success_rate = (cache_hit / len(content_tasks)) * 100

    df = pd.DataFrame(debug_logs)
    df.to_csv("debug_log.csv", index=False, encoding="utf-8-sig")
    print(f"\n>>> {len(df)} 行を書き出しました → debug_log.csv")

    print("\n=== Existing Method (Same Tasks) ===")
    print(f"  - Cache Hit: {cache_hit}")
    print(f"  - Success Rate: {success_rate:.2f}%")
    print(f"  - Avg Hops: {avg_hops:.2f}")

    overall_stats = {
        'Single': gather_overall_stats(single_all_runs),
        'NR-Adapt': gather_overall_stats(attrib_noreset_all_adapt),
    }
    plot_overall_metrics(overall_stats)

if __name__ == "__main__":
    main()