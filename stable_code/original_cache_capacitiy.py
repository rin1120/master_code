# 既存手法、提案手法において、キャッシュ容量を持たせたシミュレーション

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import namedtuple  
from collections import deque

# ----------------------------------------------------------------------------
# パラメータ設定
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 3
NUM_CONTENTS_TO_SEARCH = 50  # 探索するコンテンツ数
NUM_ANTS = 10
NUM_ITERATIONS = 100

ALPHA = 1.0   # フェロモンの指数
BETA = 1.0    # SOM類似度の指数
Q = 100

# --- 可変蒸発・付加パラメータ ----------------------------------------
RHO      = 0.10   # ベース蒸発率 (従来の RHO と同値で可)
RHO_MAX   = 0.9   # 蒸発率の上限
LAMBDA_R  = 1   # 拡張係数 λ   (0.1〜0.5 で様子を見ると良い),1だとこの変数がないと同じ
BOOST = 5 # フェロモンの付加量を調整するための係数
# ---------------------------------------------------------------

TIMES_TO_SEARCH_HOP = 50
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10  
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1
CACHE_CAPACITY = 20   

USE_FIXED_START_NODE = False

# ε-greedy戦略用パラメータ（リセットなしの場合のみ使用）
USE_EPSILON = True
EPSILON = 0.01  # 10%の確率でランダム選択


# CSVファイルから属性ベクトルを準備
file_path = "500_movies.csv"  # 適宜修正
#file_path ="1500_wines.csv" # 適宜修正
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
            cache_storage[i][j] = deque(maxlen=CACHE_CAPACITY)
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

def cache_insert(node_list, cid):
    if cid in node_list:        # 既に同コンテンツを保持
        return True
    if len(node_list) < CACHE_CAPACITY:
        node_list.append(cid)   # 空きがあれば追加
        return True
    return False                # 満杯で追放は行わない

def place_with_neighborhood(cache_storage, net_vec, pos, cid, size):
    queue   = [pos]            # FIFO キュー (BFS)
    visited = set()            # 再訪防止
    vect_c  = cont_vector_array[cid - 1]  # コンテンツ属性ベクトル

    while queue:
        v = queue.pop(0)       # 現在ノード
        if v in visited:
            continue
        visited.add(v)

        # (2) 空きがあれば追加して終了
        if cache_insert(cache_storage[v[0]][v[1]], cid):
            return True

        # (3) 隣接に同一 cid があれば重複を避けて終了
        for nb in get_neighbors(v[0], v[1], size):
            if cid in cache_storage[nb[0]][nb[1]]:
                return False

        # (4) 隣接の中で SOM 類似度が最大のノードを次候補に
        best_nb, best_sim = None, -1
        for nb in get_neighbors(v[0], v[1], size):
            if nb in visited:
                continue
            sim = -np.linalg.norm(vect_c - net_vec[nb])  # 距離小 → sim 大
            if sim > best_sim:
                best_nb, best_sim = nb, sim
        if best_nb:                # 候補があればキューへ
            queue.append(best_nb)

    # 探索を尽くしても空きが無い場合
    return False

# ----------------------------------------------------------------------------
# キャッシュ配置
# ----------------------------------------------------------------------------
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
            # キャッシュ容量無限
            #cache_storage[curr[0]][curr[1]].append(cid)
            #　キャッシュ容量有限
            place_with_neighborhood(cache_storage,net_vector_array, curr, cid, size)

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
            #デバッグ用レコードを追加
            log_list.append(DebugRec("BASE", cid, start_node,curr, hops_used, hops_used, []))
        total_hops_used += hops_used
    avg_hops = total_hops_used / len(content_tasks) if content_tasks else 0.0
    return cache_hit, avg_hops

# ----------------------------------------------------------------------------
# ① 単一フェロモン方式 (リセットあり)
# ----------------------------------------------------------------------------
def multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks, log_list):
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
                    #デバック用レコードの追加
                    log_list.append(DebugRec("SINGLE", cid, start_node,current_node, cost, cost, path))
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
# ----------------------------------------------------------------------------
def multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, log_list, reset_pheromone=True):
    results = []
    tag = "ATTRIB_R" if reset_pheromone else "ATTRIB_NR"
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
            if reset_pheromone == False and best_cost >0:
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
                    if reset_pheromone==False and cost <= best_cost:
                        Q_eff = Q * BOOST
                    else:
                        Q_eff = Q
                    #正規化なし
                    #delta = (Q * vect) / cost
                    #正規化あり
                    delta = (Q_eff * vect) / (cost * np.sum(vect))
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta
        results.append(iteration_data)
    return results



def multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks, log_list):
    return multi_contents_attrib_pheromone_common(cache_storage, net_vector_array, size, content_tasks, log_list, reset_pheromone=False)


# --- 新規関数: 総合結果の統計量を計算する ---
def _stat_from_result(result_list):
    """ACO 関数が返す result_list から平均ホップと成功率を抽出"""
    all_costs = []
    succ_cnt  = 0
    for content_result in result_list:          # 各コンテンツ
        for iter_costs in content_result:       # 各イテレーション
            all_costs.extend(iter_costs)
            succ_cnt += sum(c < TIMES_TO_SEARCH_HOP for c in iter_costs)

    if not all_costs:           # キャッシュなしで全くヒットしないケース
        return float('inf'), 0
    avg_hops = np.mean(all_costs)
    succ_rate = succ_cnt / len(all_costs) * 100
    return avg_hops, succ_rate


# --- 性能取得用 ---
def run_once_all_methods(cache_storage, net_vec, size, tasks):
    # --- Previous ---
    hit_p, avg_p = search_prop(cache_storage, net_vec, size, tasks, [])
    succ_p = hit_p / len(tasks) * 100

    # --- Single Pheromone ---
    single_res  = multi_contents_single_pheromone_with_reset(
                      cache_storage, net_vec, size, tasks, [])
    avg_s, succ_s = _stat_from_result(single_res)

    # --- Attrib Pheromone ---
    attrib_res  = multi_contents_attrib_pheromone_no_reset(
                      cache_storage, net_vec, size, tasks, [])
    avg_a, succ_a = _stat_from_result(attrib_res)

    return (avg_p, succ_p, avg_s, succ_s, avg_a, succ_a)


# --- 容量レンジ ---
def simulate_for_capacity(cap_list, size=50):
    hops_p, hops_s, hops_a = [], [], []
    succ_p, succ_s, succ_a = [], [], []
    for cap in cap_list:
        global CACHE_CAPACITY
        CACHE_CAPACITY = cap
        h_p = h_s = h_a = s_p = s_s = s_a = 0
        for _ in range(TIME_TO_SIMULATE):
            cache_storage, net_vec = cache_prop(size)
            tasks = generate_multi_contents_tasks(size, NUM_CONTENTS_TO_SEARCH)
            hp, sp, hs, ss, ha, sa = run_once_all_methods(
                                        cache_storage, net_vec, size, tasks)
            h_p += hp; h_s += hs; h_a += ha
            s_p += sp; s_s += ss; s_a += sa
        hops_p.append(h_p / TIME_TO_SIMULATE)
        hops_s.append(h_s / TIME_TO_SIMULATE)
        hops_a.append(h_a / TIME_TO_SIMULATE)
        succ_p.append(s_p / TIME_TO_SIMULATE)
        succ_s.append(s_s / TIME_TO_SIMULATE)
        succ_a.append(s_a / TIME_TO_SIMULATE)
    return (hops_p, hops_s, hops_a, succ_p, succ_s, succ_a)

# ---------- グラフ描画 ----------
def plot_capacity_tradeoff(cap_list,
                           hops_p, hops_s, hops_a,
                           succ_p, succ_s, succ_a):
    plt.figure()
    plt.plot(cap_list, hops_p, marker='o', label='Previous')
    plt.plot(cap_list, hops_s, marker='s', label='Single Pheromone')
    plt.plot(cap_list, hops_a, marker='^', label='Attrib Pheromone')
    plt.xlabel("Cache Capacity"); plt.ylabel("Average Hops")
    plt.title("Avg Hops"); plt.legend(); plt.grid(False)

    plt.figure()
    plt.plot(cap_list, succ_p, marker='o', label='Previous')
    plt.plot(cap_list, succ_s, marker='s', label='Single Pheromone')
    plt.plot(cap_list, succ_a, marker='^', label='Attrib Pheromone')
    plt.xlabel("Cache Capacity"); plt.ylabel("Success Rate (%)")
    plt.title("Success Rate"); plt.legend(); plt.grid(False)
    plt.show()

# ---------- メイン実行 ----------
def main():
    size = 30
    cap_range = [5, 10, 15]         # 横軸：キャッシュ容量

    # ３手法×容量ごとの平均ホップ数・成功率を取得
    (h_prev, h_sing, h_attr,
     s_prev, s_sing, s_attr) = simulate_for_capacity(cap_range, size)

    # グラフ描画（２枚）
    plot_capacity_tradeoff(cap_range,
                           h_prev, h_sing, h_attr,
                           s_prev, s_sing, s_attr)

if __name__ == "__main__":
    main()