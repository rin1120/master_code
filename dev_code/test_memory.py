import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict

# ----------------------------------------------------------------------------
# パラメータ設定 (変更なし)
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 5
NUM_CONTENTS_TO_SEARCH = 50
NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA = 1.0
BETA = 1.0
Q = 100
RHO = 0.10
BOOST = 5
TIMES_TO_SEARCH_HOP = 50
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
USE_EPSILON = True
EPSILON = 0.1

# ----------------------------------------------------------------------------
# データ準備 (変更なし)
# ----------------------------------------------------------------------------
file_path = "500_movies.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"警告: '{file_path}' が見つかりません。ダミーデータで続行します。")
    data = {'id': range(1, 501)}
    for i in range(14): data[f'genre_{i}'] = np.random.randint(0, 2, 500)
    df = pd.DataFrame(data)
N = len(df.columns) - 1
cont_num = len(df)
cont_vector_array = [np.array(vec) for vec in df.set_index('id').values.tolist()]

# ----------------------------------------------------------------------------
# ネットワーク・キャッシュ関連の関数 (変更なし)
# ----------------------------------------------------------------------------
def get_neighbors(x, y, size):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

def initialize_pheromone_trails(size, vector_size):
    return defaultdict(lambda: np.ones(vector_size))

def initialize_single_pheromone_trails(size):
    return defaultdict(lambda: 1.0)

def cache_prop(size):
    cache_storage = np.empty((size, size), dtype=object)
    for i in range(size):
        for j in range(size): cache_storage[i][j] = []
    net_vector_array = np.random.rand(size, size, N)
    for _ in range(TIME_TO_CACHE_PER_CONTENT):
        for cid in range(1, cont_num + 1):
            curr = (np.random.randint(size), np.random.randint(size))
            vect = cont_vector_array[cid - 1]
            for _ in range(TIMES_TO_CACHE_HOP):
                neighbors = get_neighbors(curr[0], curr[1], size)
                if not neighbors: break
                dists = [np.linalg.norm(vect - net_vector_array[n]) for n in neighbors]
                curr = neighbors[np.argmin(dists)]
            if cid not in cache_storage[curr[0]][curr[1]]:
                cache_storage[curr[0]][curr[1]].append(cid)
            net_vector_array[curr] += LEARNING_RATE * (vect - net_vector_array[curr])
    return cache_storage, net_vector_array

def generate_multi_contents_tasks(size, k):
    # generate_multi_contents_tasksは辞書のリストを返すように変更
    tasks = []
    for _ in range(k):
        cid = np.random.randint(1, cont_num + 1)
        start_node = (np.random.randint(size), np.random.randint(size))
        tasks.append({'cid': cid, 'start_node': start_node})
    return tasks

# ----------------------------------------------------------------------------
# ① 比較手法（並列モデル）
# ----------------------------------------------------------------------------
def conventional_method_mem_limit(cache_storage, net_vector_array, size, content_tasks, memory_limit):
    # --- 状態管理の初期化 ---
    pheromone_dict = {}
    swap_count = 0
    task_states = []
    for task in content_tasks:
        ants = [{'path': [task['start_node']], 'visited': {task['start_node']}, 'cost': 0, 'active': True} for _ in range(NUM_ANTS)]
        task_states.append({'cid': task['cid'], 'ants': ants})

    # --- 【変更】ループの順番を入れ替え ---
    # イテレーションループが一番外側
    for _ in range(NUM_ITERATIONS):
        iter_success_paths = defaultdict(list)
        iter_success_costs = defaultdict(list)

        # コンテンツ（タスク）ループ
        for task in task_states:
            cid = task['cid']
            # メモリ管理
            if cid not in pheromone_dict:
                if len(pheromone_dict) >= memory_limit:
                    min_sum, cid_to_evict = min(((sum(trails.values()), c) for c, trails in pheromone_dict.items()), default=(float('inf'), None))
                    if cid_to_evict:
                        del pheromone_dict[cid_to_evict]
                        swap_count += 1
                pheromone_dict[cid] = initialize_single_pheromone_trails(size)

            # アリのループ（1ステップだけ進める）
            for ant in task['ants']:
                if not ant['active']: continue
                
                current_node = ant['path'][-1]
                if cid in cache_storage[current_node[0]][current_node[1]]:
                    ant['active'] = False
                    iter_success_paths[cid].append(ant['path'])
                    iter_success_costs[cid].append(ant['cost'])
                    continue
                if ant['cost'] >= TIMES_TO_SEARCH_HOP:
                    ant['active'] = False
                    continue
                
                neighbors = [n for n in get_neighbors(current_node[0], current_node[1], size) if n not in ant['visited']]
                if not neighbors:
                    ant['active'] = False
                    continue
                
                vect = cont_vector_array[cid - 1]
                trails = pheromone_dict[cid]
                if USE_EPSILON and random.random() < EPSILON:
                    next_node = random.choice(neighbors)
                else:
                    probs = [(trails[(current_node, n)] ** ALPHA) * (1.0 / (np.linalg.norm(vect - net_vector_array[n]) + 1e-6)) ** BETA for n in neighbors]
                    sum_probs = sum(probs)
                    next_node = random.choices(neighbors, weights=probs, k=1)[0] if sum_probs > 0 else random.choice(neighbors)
                
                ant['path'].append(next_node)
                ant['visited'].add(next_node)
                ant['cost'] += 1

        # フェロモン更新（イテレーションの最後に一括）
        for cid, trails in pheromone_dict.items():
            for edge in list(trails.keys()): trails[edge] *= (1 - RHO)
            if cid in iter_success_costs and iter_success_costs[cid]:
                best_cost_iter = min(iter_success_costs[cid])
                for path, cost in zip(iter_success_paths[cid], iter_success_costs[cid]):
                    if cost > 0:
                        delta = (Q * BOOST if cost <= best_cost_iter else Q) / cost
                        for i in range(len(path) - 1):
                            trails[(path[i], path[i+1])] += delta
    
    # --- 最終結果の集計 ---
    final_costs = []
    for task in task_states:
        for ant in task['ants']:
            final_costs.append(ant['cost'] if not ant['active'] else TIMES_TO_SEARCH_HOP)
            
    return final_costs, swap_count

# ----------------------------------------------------------------------------
# ② 提案手法（並列モデル）
# ----------------------------------------------------------------------------
def proposed_method_mem_limit(cache_storage, net_vector_array, size, content_tasks, memory_limit):
    # --- 状態管理の初期化 ---
    effective_vector_size = min(memory_limit, N)
    pheromone_trails = initialize_pheromone_trails(size, effective_vector_size)
    task_states = []
    for task in content_tasks:
        ants = [{'path': [task['start_node']], 'visited': {task['start_node']}, 'cost': 0, 'active': True} for _ in range(NUM_ANTS)]
        task_states.append({'cid': task['cid'], 'ants': ants})

    # --- 【変更】ループの順番を入れ替え ---
    for _ in range(NUM_ITERATIONS):
        iter_success_paths = defaultdict(list)
        iter_success_costs = defaultdict(list)

        # コンテンツ（タスク）ループ
        for task in task_states:
            cid = task['cid']
            # アリのループ（1ステップだけ進める）
            for ant in task['ants']:
                if not ant['active']: continue
                
                current_node = ant['path'][-1]
                if cid in cache_storage[current_node[0]][current_node[1]]:
                    ant['active'] = False
                    iter_success_paths[cid].append(ant['path'])
                    iter_success_costs[cid].append(ant['cost'])
                    continue
                if ant['cost'] >= TIMES_TO_SEARCH_HOP:
                    ant['active'] = False
                    continue

                neighbors = [n for n in get_neighbors(current_node[0], current_node[1], size) if n not in ant['visited']]
                if not neighbors:
                    ant['active'] = False
                    continue

                vect = cont_vector_array[cid - 1]
                if USE_EPSILON and random.random() < EPSILON:
                    next_node = random.choice(neighbors)
                else:
                    vect_truncated = vect[:effective_vector_size]
                    sum_vect_truncated = np.sum(vect_truncated)
                    probs = []
                    for n in neighbors:
                        tau_vect = pheromone_trails[(current_node, n)]
                        eta = 1.0 / (np.linalg.norm(vect - net_vector_array[n]) + 1e-6)
                        attractiveness = np.sum(tau_vect * vect_truncated) if sum_vect_truncated > 0 else 0
                        probs.append(attractiveness * (eta ** BETA))
                    sum_probs = sum(probs)
                    next_node = random.choices(neighbors, weights=probs, k=1)[0] if sum_probs > 0 else random.choice(neighbors)
                
                ant['path'].append(next_node)
                ant['visited'].add(next_node)
                ant['cost'] += 1

        # フェロモン更新（イテレーションの最後に一括）
        for edge in list(pheromone_trails.keys()):
            pheromone_trails[edge] *= (1 - RHO)
        for cid, paths in iter_success_paths.items():
            if not paths: continue
            vect = cont_vector_array[cid - 1]
            vect_truncated = vect[:effective_vector_size]
            sum_vect_truncated = np.sum(vect_truncated)
            if sum_vect_truncated == 0: continue
            best_cost_iter = min(iter_success_costs[cid])
            for path, cost in zip(paths, iter_success_costs[cid]):
                if cost > 0:
                    delta_vect = ((Q * BOOST if cost <= best_cost_iter else Q) / cost) * (vect_truncated / sum_vect_truncated)
                    for i in range(len(path) - 1):
                        pheromone_trails[(path[i], path[i+1])] += delta_vect

    # --- 最終結果の集計 ---
    final_costs = []
    for task in task_states:
        for ant in task['ants']:
            final_costs.append(ant['cost'] if not ant['active'] else TIMES_TO_SEARCH_HOP)
    
    return final_costs, 0

# ----------------------------------------------------------------------------
# 結果描画用の関数 (変更なし)
# ----------------------------------------------------------------------------
def plot_performance_vs_memory(results):
    font_size = 16
    memory_limits = sorted(results['conventional'].keys())
    conv_hops = [results['conventional'][mem]['avg_hops'] for mem in memory_limits]
    conv_succ = [results['conventional'][mem]['success_rate'] for mem in memory_limits]
    conv_swaps = [results['conventional'][mem]['avg_swaps'] for mem in memory_limits]
    prop_hops = [results['proposed'][mem]['avg_hops'] for mem in memory_limits]
    prop_succ = [results['proposed'][mem]['success_rate'] for mem in memory_limits]
    
    plt.figure(figsize=(8, 6))
    plt.plot(memory_limits, conv_hops, label='比較手法 (ID毎)', color='red', marker='o')
    plt.plot(memory_limits, prop_hops, label='提案手法 (属性毎)', color='blue', marker='s')
    plt.title("メモリ上限と平均ホップ数の関係（並列モデル）", fontsize=font_size)
    plt.xlabel("フェロモン用メモリ上限値", fontsize=font_size)
    plt.ylabel("平均ホップ数", fontsize=font_size)
    plt.xticks(memory_limits)
    plt.legend(fontsize=font_size)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(memory_limits, conv_succ, label='比較手法 (ID毎)', color='red', marker='o')
    plt.plot(memory_limits, prop_succ, label='提案手法 (属性毎)', color='blue', marker='s')
    plt.title("メモリ上限と探索成功率の関係（並列モデル）", fontsize=font_size)
    plt.xlabel("フェロモン用メモリ上限値", fontsize=font_size)
    plt.ylabel("探索成功率 (%)", fontsize=font_size)
    plt.xticks(memory_limits)
    plt.legend(fontsize=font_size)
    plt.ylim(0, 101)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(memory_limits, conv_swaps, label='比較手法 (ID毎)', color='green', marker='^')
    plt.title("メモリ上限とフェロモン入れ替わり数の関係（並列モデル）", fontsize=font_size)
    plt.xlabel("フェロモン用メモリ上限値", fontsize=font_size)
    plt.ylabel("平均入れ替わり数", fontsize=font_size)
    plt.xticks(memory_limits)
    plt.legend(fontsize=font_size)
    plt.ylim(bottom=0)
    plt.show()


# ----------------------------------------------------------------------------
# メイン処理 (変更なし)
# ----------------------------------------------------------------------------
def main():
    size = 50
    MEMORY_LIMITS = [14, 20, 30, 40, 50]
    
    print(f"グリッドサイズ: {size}x{size}, 属性数: {N}")
    print(f"検証メモリ上限: {MEMORY_LIMITS}")

    cache_storage, net_vector_array = cache_prop(size)
    
    results = {'conventional': {}, 'proposed': {}}

    for limit in MEMORY_LIMITS:
        print(f"\n--- メモリ上限: {limit} で検証中... ---")
        
        conv_total_costs = []
        prop_total_costs = []
        conv_total_swaps = 0

        for i in range(TIME_TO_SIMULATE):
            print(f"  シミュレーション回数: {i+1}/{TIME_TO_SIMULATE}")
            content_tasks = generate_multi_contents_tasks(size, k=NUM_CONTENTS_TO_SEARCH)
            
            # 各手法を呼び出し
            conv_costs, conv_swaps = conventional_method_mem_limit(cache_storage, net_vector_array, size, content_tasks, limit)
            prop_costs, _ = proposed_method_mem_limit(cache_storage, net_vector_array, size, content_tasks, limit)

            conv_total_costs.extend(conv_costs)
            prop_total_costs.extend(prop_costs)
            conv_total_swaps += conv_swaps

        # 統計計算
        rate_conv = sum(1 for c in conv_total_costs if c < TIMES_TO_SEARCH_HOP) / len(conv_total_costs) * 100 if conv_total_costs else 0
        hops_conv = np.mean(conv_total_costs) if conv_total_costs else TIMES_TO_SEARCH_HOP
        avg_swaps_conv = conv_total_swaps / TIME_TO_SIMULATE
        results['conventional'][limit] = {'success_rate': rate_conv, 'avg_hops': hops_conv, 'avg_swaps': avg_swaps_conv}
        print(f"  比較手法   -> 成功率: {rate_conv:.2f}%, 平均ホップ: {hops_conv:.2f}, 平均入れ替わり数: {avg_swaps_conv:.2f}")

        rate_prop = sum(1 for c in prop_total_costs if c < TIMES_TO_SEARCH_HOP) / len(prop_total_costs) * 100 if prop_total_costs else 0
        hops_prop = np.mean(prop_total_costs) if prop_total_costs else TIMES_TO_SEARCH_HOP
        results['proposed'][limit] = {'success_rate': rate_prop, 'avg_hops': hops_prop}
        print(f"  提案手法   -> 成功率: {rate_prop:.2f}%, 平均ホップ: {hops_prop:.2f}")

    plot_performance_vs_memory(results)

if __name__ == "__main__":
    main()