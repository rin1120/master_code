# 山代法＋比較手法（コンテンツ毎にフェロモン管理）＋提案手法（属性毎にフェロモン管理）
#比較手法、提案手法どちらにもローカルベストの探索を実装
#探索性能は全ての探索を反映（成功時と失敗時の両方）
#コンテンツ生成手法をランダム、zipf、類似の3種類実装
#iter=1を山白法＋ランダム探索,iter>1以降でSOMとACOの組み合わせを実施（パラメータ可変）
#パラメータ0は暫定版
#SOMを５つに拡張、フェロモンはマップ毎に保持、共有しない
# ★追加仕様：キャッシュ探索成功時に要求源ノードにキャッシュを配置
# ★重要：ただし「そのタスクの5マップ探索が全部終わってから」1回だけ配置（張り付き防止）

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import namedtuple
from collections import defaultdict
import copy

# ----------------------------------------------------------------------------
# パラメータ設定
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 1
NUM_CONTENTS_TO_SEARCH = 10  # 探索するコンテンツ数
NUM_ANTS = 10
NUM_ITERATIONS = 100

ALPHA_START = 1.0   # 初期フェロモンの指数
BETA_START = 10.0   # 初期SOM類似度の指数
ALPHA_END = 1.0     # 最終フェロモンの指数
BETA_END = 1.0      # 最終SOM類似度の指数
Q = 100

# --- 可変蒸発・付加パラメータ ----------------------------------------
RHO       = 0.10  # ベース蒸発率
RHO_MAX   = 0.9   # 蒸発率の上限
LAMBDA_R  = 1     # 拡張係数 λ
BOOST     = 5     # フェロモンの付加量係数
# ---------------------------------------------------------------

TIMES_TO_SEARCH_HOP = 100
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1
NUM_MAPS = 5

USE_FIXED_START_NODE = False

# ε-greedy戦略用パラメータ
USE_EPSILON = True
EPSILON = 0.01
EPSILON_SOM = 0.20

# CSVファイルから属性ベクトルを準備
file_path = "500_movies.csv"  # 適宜修正
df = pd.read_csv(file_path)
N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec) for vec in cont_vector]

# --- ジャンルとコンテンツIDの対応辞書を作成---
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
    net_vector_arrays = [get_init_network_vector(size) for _ in range(NUM_MAPS)]
    placements_per_map = TIME_TO_CACHE_PER_CONTENT // NUM_MAPS
    cache_hops = TIMES_TO_CACHE_HOP
    alpha_zero = LEARNING_RATE

    for map_index in range(NUM_MAPS):
        current_net_vector_array = net_vector_arrays[map_index]
        # 各マップにキャッシュを配置
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
                    current_net_vector_array[node] += alpha * (vect - current_net_vector_array[node])

    return cache_storage, net_vector_arrays

# ----------------------------------------------------------------------------
# ★★★ ICN拡張：探索成功時に要求源へキャッシュ配置（タスク単位で1回だけ呼ぶ） ★★★
# ----------------------------------------------------------------------------
def apply_icn_update(cache_storage, net_vector_arrays, cid, vect, start_node, path_from_found_to_start):
    """
    探索成功時のICN処理を一括実行する（タスクごとに1回だけ呼び出す想定）
    path_from_found_to_start: [発見場所 -> ... -> 要求元] の順に並んだ座標リスト
    """
    # 1. キャッシュ格納 (要求元ノードへ)
    if cid not in cache_storage[start_node[0]][start_node[1]]:
        cache_storage[start_node[0]][start_node[1]].append(cid)

    # 2. ベストマップ選択（要求元ノードのベクトルとコンテンツの距離で最小のマップ）
    best_map_idx = -1
    min_dist = float('inf')
    for i, net_vec in enumerate(net_vector_arrays):
        node_vec = net_vec[start_node]
        dist = np.linalg.norm(vect - node_vec)
        if dist < min_dist:
            min_dist = dist
            best_map_idx = i

    if best_map_idx < 0:
        return

    # 3. マップの重み更新（発見→要求元の経路上を更新）
    target_map = net_vector_arrays[best_map_idx]
    total_hops = len(path_from_found_to_start)
    if total_hops <= 0:
        return

    tmp = 0
    for node in path_from_found_to_start:
        tmp += 1
        alpha = LEARNING_RATE * (tmp / total_hops)
        target_map[node] += alpha * (vect - target_map[node])

# ----------------------------------------------------------------------------
# コンテンツ探索タスク生成
# ----------------------------------------------------------------------------
# r = random , z = zipf, s = similar

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

def generate_multi_contents_tasks_z(size, k=NUM_CONTENTS_TO_SEARCH, popularity_ratio=0.01, request_concentration=0.8, mode="mix"):
    tasks = []
    num_popular = int(cont_num * popularity_ratio)
    all_cids = np.arange(1, cont_num + 1)
    popular_cids = np.random.choice(all_cids, num_popular, replace=False) if num_popular > 0 else np.array([], dtype=int)
    unpopular_cids = np.setdiff1d(all_cids, popular_cids) if num_popular > 0 else all_cids

    mode = str(mode).lower()

    for _ in range(k):
        if mode == "popular_only":
            if len(popular_cids) > 0:
                cid = int(random.choice(popular_cids))
            else:
                cid = int(random.choice(all_cids))
        elif mode == "unpopular_only":
            if len(unpopular_cids) > 0:
                cid = int(random.choice(unpopular_cids))
            else:
                cid = int(random.choice(all_cids))
        else:
            if random.random() < request_concentration and len(popular_cids) > 0:
                cid = int(random.choice(popular_cids))
            else:
                if len(unpopular_cids) > 0:
                    cid = int(random.choice(unpopular_cids))
                else:
                    cid = int(random.choice(all_cids))

        start_node = (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start_node))
    return tasks

def generate_multi_contents_tasks_s(size, k=NUM_CONTENTS_TO_SEARCH, burst_size=1000):
    """
    アンカーをランダムに1つ選び、ユークリッド距離で近い順に候補を作成。
    候補上位から 1/距離 の重みで k 件サンプリングして要求列を返す。
    """
    V = np.vstack(cont_vector_array).astype(float)

    anchor_idx = np.random.randint(cont_num)
    anchor_vec = V[anchor_idx]

    dist = np.linalg.norm(V - anchor_vec, axis=1)
    dist[anchor_idx] = np.inf

    M = int(max(1, min(burst_size, cont_num - 1)))
    idx_top = np.argpartition(dist, M)[:M]
    cand_cids = (idx_top + 1).astype(int)

    w = 1.0 / (dist[idx_top] + 1e-12)
    p = w / w.sum()

    sampled_cids = np.random.choice(cand_cids, size=k, replace=True, p=p)

    def _start():
        return (size // 2, size // 2) if USE_FIXED_START_NODE else (
            np.random.randint(size), np.random.randint(size)
        )

    tasks = [(int(cid), _start()) for cid in sampled_cids]
    return tasks

# ----------------------------------------------------------------------------
# 統計量計算
# ----------------------------------------------------------------------------
def compute_stats_from_costs(iteration_costs_data):
    means, success_rates = [], []
    for costs in iteration_costs_data:
        means.append(np.mean(costs))
        total = len(costs)
        success_count = sum(1 for c in costs if c < TIMES_TO_SEARCH_HOP)
        success_rates.append((success_count / total) * 100)
    return means, success_rates

# ----------------------------------------------------------------------------
# デバッグ用レコード
# ----------------------------------------------------------------------------
DebugRec = namedtuple("DebugRec",
                      ["method", "cid", "start", "found",
                       "hops", "cost", "path"])

# ----------------------------------------------------------------------------
# 生成タスクの可視化
# ----------------------------------------------------------------------------
def visualize_task_distribution(tasks, title):
    if not tasks:
        print("タスクリストが空です。")
        return

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
# 既存手法（山代法/SOM探索）：複数マップ並列（更新なし）
# ----------------------------------------------------------------------------
def search_prop(cache_storage, net_vector_arrays, size, content_tasks, log_list):
    cache_hit = 0
    total_hops_used = 0
    for (cid, start_node) in content_tasks:
        hops_per_map = []
        found_on_any_map = False

        for map_index in range(NUM_MAPS):
            current_net_vector_array = net_vector_arrays[map_index]

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
                        dist = np.linalg.norm(vect - current_net_vector_array[neighbor])
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
                found_on_any_map = True
                hops_per_map.append(hops_used)
            else:
                hops_per_map.append(TIMES_TO_SEARCH_HOP)

        min_hops = min(hops_per_map)
        if found_on_any_map:
            cache_hit += 1
        total_hops_used += min_hops

    avg_hops = total_hops_used / len(content_tasks) if content_tasks else 0.0
    return cache_hit, avg_hops

# ----------------------------------------------------------------------------
# ★ 1タスク分：提案手法1（ID毎フェロモン）を「1マップ」で実行（更新はしない）
# ----------------------------------------------------------------------------
def run_single_task_single_pheromone(cache_storage, net_vector_array, size, cid, start_node, log_list, pheromone_dict):
    # フェロモン辞書準備
    if cid not in pheromone_dict:
        pheromone_dict[cid] = initialize_single_pheromone_trails(size)
    pheromone_trails = pheromone_dict[cid]

    # コンテンツが存在しないなら空
    content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
    if not content_nodes:
        return [], None, TIMES_TO_SEARCH_HOP

    vect = cont_vector_array[cid - 1]
    iteration_data = []
    best_cost = TIMES_TO_SEARCH_HOP

    global_best_path_from_found_to_start = None
    global_min_hops_in_task = float('inf')

    for t in range(NUM_ITERATIONS):
        iter_costs = []
        all_paths = []
        all_costs = []

        for _ant in range(NUM_ANTS):
            path = []
            visited = set()
            current_node = start_node
            path.append(current_node)
            visited.add(current_node)
            cost = 0
            found = False

            if t == 0:
                # Iter.0：SOM + ε-greedy
                searched_node = []
                curr = start_node
                cost = 0
                path = [curr]

                found_node = None

                for _ in range(TIMES_TO_SEARCH_HOP):
                    cost += 1

                    # 現ノード命中
                    if cid in cache_storage[curr[0]][curr[1]]:
                        found = True
                        found_node = curr
                        break

                    # 近傍命中（「見つかったノード」も経路に入れる）
                    hit_neighbor = False
                    for nx, ny in get_neighbors(curr[0], curr[1], size):
                        if cid in cache_storage[nx][ny]:
                            found = True
                            hit_neighbor = True
                            found_node = (nx, ny)
                            path.append(found_node)
                            break
                    if found and hit_neighbor:
                        break

                    allowed_neighbors = [n for n in get_neighbors(curr[0], curr[1], size) if n not in searched_node]
                    if not allowed_neighbors:
                        break

                    min_dist = float('inf')
                    closest = None
                    for neighbor in allowed_neighbors:
                        dist = np.linalg.norm(vect - net_vector_array[neighbor])
                        if dist < min_dist:
                            min_dist = dist
                            closest = neighbor
                    if closest is None:
                        break

                    if USE_EPSILON and random.random() < EPSILON_SOM:
                        next_node = random.choice(allowed_neighbors)
                    else:
                        next_node = closest

                    searched_node.append(curr)
                    curr = next_node
                    path.append(curr)

                    if cid in cache_storage[curr[0]][curr[1]]:
                        found = True
                        found_node = curr
                        break

                current_node = found_node if found_node is not None else curr

            else:
                # Iter.1以降：ACO
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
                        if NUM_ITERATIONS >= 1:
                            ALPHA = ALPHA_START + (ALPHA_END - ALPHA_START) * (t / (NUM_ITERATIONS - 1))
                            BETA  = BETA_START  + (BETA_END  - BETA_START)  * (t / (NUM_ITERATIONS - 1))

                        probs = []
                        denom = 0.0
                        for n in allowed:
                            edge = (current_node, n)
                            tau = pheromone_trails[edge]
                            dist = np.linalg.norm(vect - net_vector_array[n])
                            eta = 1.0 / (dist + 1e-6)
                            if np.sum(vect) > 0:
                                A_ij = np.sum((tau ** ALPHA) * vect) / np.sum(vect)
                            else:
                                A_ij = 0.0
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

                log_list.append(DebugRec("SINGLE", cid, start_node, current_node, cost, cost, path))

                # タスク内の最良経路（このマップ内）
                if cost < global_min_hops_in_task:
                    global_min_hops_in_task = cost
                    global_best_path_from_found_to_start = path[::-1]  # found -> ... -> start

            else:
                iter_costs.append(TIMES_TO_SEARCH_HOP)

        # そのイテレーションの最良
        if all_costs:
            iteration_best = min(all_costs)
            if iteration_best < best_cost:
                best_cost = iteration_best

        iteration_data.append(iter_costs)

        # 蒸発
        for edge in pheromone_trails:
            pheromone_trails[edge] *= (1 - RHO)
            pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)

        # ローカルベストに応じた追加蒸発
        if best_cost > 0:
            for pth, cst in zip(all_paths, all_costs):
                delta_p = max(0.0, (cst / best_cost) - 1.0)
                if delta_p == 0.0:
                    continue
                rho_eff   = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                extra_mul = 1.0 - (rho_eff - RHO)
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] = max(pheromone_trails[(a, b)] * extra_mul, 1e-6)

        # 付加
        for pth, cst in zip(all_paths, all_costs):
            if cst > 0:
                Q_eff = Q * BOOST if cst <= best_cost else Q
                delta = Q_eff / cst
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] += delta

    best_hops = global_min_hops_in_task if global_best_path_from_found_to_start is not None else TIMES_TO_SEARCH_HOP
    return iteration_data, global_best_path_from_found_to_start, best_hops

# ----------------------------------------------------------------------------
# ★ 1タスク分：提案手法2（属性毎フェロモン）を「1マップ」で実行（更新はしない）
#    ※ pheromone_trails は「マップごとに保持」して、タスク間で使い回す
# ----------------------------------------------------------------------------
def run_single_task_attrib_pheromone(cache_storage, net_vector_array, size, cid, start_node, log_list, pheromone_trails):
    content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
    if not content_nodes:
        return [], None, TIMES_TO_SEARCH_HOP

    vect = cont_vector_array[cid - 1]
    iteration_data = []
    best_cost = TIMES_TO_SEARCH_HOP

    global_best_path_from_found_to_start = None
    global_min_hops_in_task = float('inf')

    for t in range(NUM_ITERATIONS):
        iter_costs = []
        all_paths = []
        all_costs = []

        for _ant in range(NUM_ANTS):
            path = []
            visited = set()
            current_node = start_node
            path.append(current_node)
            visited.add(current_node)
            cost = 0
            found = False

            if t == 0:
                # Iter.0：SOM + ε-greedy
                searched_node = []
                curr = start_node
                cost = 0
                path = [curr]

                found_node = None

                for _ in range(TIMES_TO_SEARCH_HOP):
                    cost += 1

                    if cid in cache_storage[curr[0]][curr[1]]:
                        found = True
                        found_node = curr
                        break

                    hit_neighbor = False
                    for nx, ny in get_neighbors(curr[0], curr[1], size):
                        if cid in cache_storage[nx][ny]:
                            found = True
                            hit_neighbor = True
                            found_node = (nx, ny)
                            path.append(found_node)
                            break
                    if found and hit_neighbor:
                        break

                    allowed_neighbors = [n for n in get_neighbors(curr[0], curr[1], size) if n not in searched_node]
                    if not allowed_neighbors:
                        break

                    min_dist = float('inf')
                    closest = None
                    for neighbor in allowed_neighbors:
                        dist = np.linalg.norm(vect - net_vector_array[neighbor])
                        if dist < min_dist:
                            min_dist = dist
                            closest = neighbor
                    if closest is None:
                        break

                    if USE_EPSILON and random.random() < EPSILON_SOM:
                        next_node = random.choice(allowed_neighbors)
                    else:
                        next_node = closest

                    searched_node.append(curr)
                    curr = next_node
                    path.append(curr)

                    if cid in cache_storage[curr[0]][curr[1]]:
                        found = True
                        found_node = curr
                        break

                current_node = found_node if found_node is not None else curr

            else:
                # Iter.1以降：ACO（属性フェロモン）
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
                        if NUM_ITERATIONS >= 1:
                            ALPHA = ALPHA_START + (ALPHA_END - ALPHA_START) * (t / (NUM_ITERATIONS - 1))
                            BETA  = BETA_START  + (BETA_END  - BETA_START)  * (t / (NUM_ITERATIONS - 1))

                        probs = []
                        denom = 0.0
                        for n in allowed:
                            edge = (current_node, n)
                            tau_list = pheromone_trails[edge]
                            dist = np.linalg.norm(vect - net_vector_array[n])
                            eta = 1.0 / (dist + 1e-6)
                            if np.sum(vect) > 0:
                                A_ij = np.sum((tau_list ** ALPHA) * vect) / np.sum(vect)
                            else:
                                A_ij = 0.0
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

                log_list.append(DebugRec("ATTRIB_NR", cid, start_node, current_node, cost, cost, path))

                if cost < global_min_hops_in_task:
                    global_min_hops_in_task = cost
                    global_best_path_from_found_to_start = path[::-1]

            else:
                iter_costs.append(TIMES_TO_SEARCH_HOP)

        # イテレーション最良
        if all_costs:
            iteration_best = min(all_costs)
            if iteration_best < best_cost:
                best_cost = iteration_best

        iteration_data.append(iter_costs)

        # 蒸発
        for edge in pheromone_trails:
            pheromone_trails[edge] *= (1 - RHO)
            pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

        # 追加蒸発（ローカルベスト）
        if best_cost > 0:
            for pth, cst in zip(all_paths, all_costs):
                delta_p = max(0.0, (cst / best_cost) - 1.0)
                if delta_p == 0.0:
                    continue
                rho_eff   = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                extra_mul = 1.0 - (rho_eff - RHO)
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] *= extra_mul
                    pheromone_trails[(a, b)] = np.maximum(pheromone_trails[(a, b)], 1e-6)

        # 付加
        for pth, cst in zip(all_paths, all_costs):
            if cst > 0:
                Q_eff = Q * BOOST if cst <= best_cost else Q
                delta = (Q_eff * vect) / (cst * np.sum(vect))
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] += delta

    best_hops = global_min_hops_in_task if global_best_path_from_found_to_start is not None else TIMES_TO_SEARCH_HOP
    return iteration_data, global_best_path_from_found_to_start, best_hops

# ----------------------------------------------------------------------------
# 複数マップ並列探索：マージ
# ----------------------------------------------------------------------------
def merge_parallel_results(map_results_list):
    if not map_results_list:
        return []

    num_maps = len(map_results_list)
    if num_maps == 0:
        return []

    num_tasks = len(map_results_list[0])
    if num_tasks == 0:
        return []

    num_iters = len(map_results_list[0][0])
    final_results = [[[] for _ in range(num_iters)] for _ in range(num_tasks)]

    for task_idx in range(num_tasks):
        for iter_idx in range(num_iters):
            merged_costs = []
            for map_idx in range(num_maps):
                if len(map_results_list[map_idx]) > task_idx and len(map_results_list[map_idx][task_idx]) > iter_idx:
                    merged_costs.extend(map_results_list[map_idx][task_idx][iter_idx])
            final_results[task_idx][iter_idx] = merged_costs

    return final_results

# ----------------------------------------------------------------------------
# 解析・可視化系（元のまま）
# ----------------------------------------------------------------------------
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
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=font_size)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

    plt.figure()
    plt.plot(its, stats_dict['Single'][1], label='提案手法1(ID毎)', color='red', marker='o')
    plt.plot(its, stats_dict['NR-Adapt'][1], label='提案手法2(属性毎)', color='blue', marker='s')
    plt.title("Overall: Success Rate")
    plt.xlabel("Iteration", fontsize=font_size)
    plt.ylabel("Success Rate (%)", fontsize=font_size)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=font_size)
    plt.xlim(left=0)
    plt.ylim(0, 100)
    plt.show()

# ----------------------------------------------------------------------------
# main（★ここが本修正：タスクごとに5マップ探索→最後に1回だけICN更新）
# ----------------------------------------------------------------------------
def main():
    size = 50
    init_cache_storage, init_net_vector_arrays = cache_prop(size)
    #コンテンツ生成方法を選択
    content_tasks = generate_multi_contents_tasks_r(size, k=NUM_CONTENTS_TO_SEARCH)
    print("Generated Content Tasks:", content_tasks)
    unique_cids = len(set(cid for cid, _ in content_tasks))
    print(f"生成コンテンツ種類数: {unique_cids}/{cont_num}")

    debug_logs = []
    single_all_runs = []
    attrib_noreset_all_adapt = []

    # 既存手法の集計用
    existing_method_hits = 0
    existing_method_hops = 0

    print("コンテンツ生成の分布を可視化します...")
    visualize_task_distribution(content_tasks, "Task Distribution")

    # マップごとにフェロモン辞書を保持（元の方針を維持）
    single_pheromone_dict_list = [{} for _ in range(NUM_MAPS)]

    for _sim in range(TIME_TO_SIMULATE):
        print(f"\n=== Simulation #{_sim + 1} ===")

        # 【提案手法1用 の環境】
        cache_for_single = copy.deepcopy(init_cache_storage)
        vectors_for_single = copy.deepcopy(init_net_vector_arrays)
        # 【提案手法2用 の環境】
        cache_for_attrib = copy.deepcopy(init_cache_storage)
        vectors_for_attrib = copy.deepcopy(init_net_vector_arrays)
        # 【既存手法用 の環境】
        cache_for_existing = copy.deepcopy(init_cache_storage)
        vectors_for_existing = copy.deepcopy(init_net_vector_arrays)

        # 属性フェロモンは「マップごとに保持」し、タスク間で使い回す（シミュレーション単位で初期化）
        attrib_pheromone_trails_list = [initialize_pheromone_trails(size, N) for _ in range(NUM_MAPS)]

        # （統計用）各マップの結果を格納：最終的に merge_parallel_results でマージ
        sim_single_results_list = [[] for _ in range(NUM_MAPS)]
        sim_attrib_results_list = [[] for _ in range(NUM_MAPS)]

        # ★重要：タスク順で進める（「そのタスクの5マップ探索が全部終わってから」ICN更新を1回だけ実施）
        for task_idx, (cid, start_node) in enumerate(content_tasks):
            vect = cont_vector_array[cid - 1]

            # -----------------------------
            # 提案手法1（ID毎フェロモン）
            # -----------------------------
            best_path_single = None
            best_hops_single = float('inf')

            for map_index in range(NUM_MAPS):
                iter_data, best_path_map, best_hops_map = run_single_task_single_pheromone(
                    cache_for_single,
                    vectors_for_single[map_index],
                    size,
                    cid,
                    start_node,
                    debug_logs,
                    single_pheromone_dict_list[map_index]
                )
                sim_single_results_list[map_index].append(iter_data)

                if best_path_map is not None and best_hops_map < best_hops_single:
                    best_hops_single = best_hops_map
                    best_path_single = best_path_map

            # ★このタスクの5マップ探索が終わってから1回だけ更新（張り付き防止）
            if best_path_single is not None:
                apply_icn_update(
                    cache_for_single,
                    vectors_for_single,
                    cid,
                    vect,
                    start_node,
                    best_path_single
                )

            # -----------------------------
            # 提案手法2（属性毎フェロモン）
            # -----------------------------
            best_path_attrib = None
            best_hops_attrib = float('inf')

            for map_index in range(NUM_MAPS):
                iter_data, best_path_map, best_hops_map = run_single_task_attrib_pheromone(
                    cache_for_attrib,
                    vectors_for_attrib[map_index],
                    size,
                    cid,
                    start_node,
                    debug_logs,
                    attrib_pheromone_trails_list[map_index]  # マップごとに保持
                )
                sim_attrib_results_list[map_index].append(iter_data)

                if best_path_map is not None and best_hops_map < best_hops_attrib:
                    best_hops_attrib = best_hops_map
                    best_path_attrib = best_path_map

            # ★このタスクの5マップ探索が終わってから1回だけ更新（張り付き防止）
            if best_path_attrib is not None:
                apply_icn_update(
                    cache_for_attrib,
                    vectors_for_attrib,
                    cid,
                    vect,
                    start_node,
                    best_path_attrib
                )

        # ---- マップ結果をマージして保存（元の方式を維持） ----
        merged_single_res = merge_parallel_results(sim_single_results_list)
        merged_attrib_res = merge_parallel_results(sim_attrib_results_list)
        single_all_runs.append(merged_single_res)
        attrib_noreset_all_adapt.append(merged_attrib_res)

        # ---- 既存手法（更新なし） ----
        hits, hops = search_prop(cache_for_existing, vectors_for_existing, size, content_tasks, debug_logs)
        existing_method_hits += hits
        existing_method_hops += hops

    # ======= ここから出力（あなたのフォーマットに合わせる） =======
    avg_existing_hits = existing_method_hits / TIME_TO_SIMULATE
    avg_existing_hops = existing_method_hops / TIME_TO_SIMULATE
    existing_success_rate = (avg_existing_hits / len(content_tasks)) * 100

    # CSV 出力（元のまま）
    df = pd.DataFrame(debug_logs)
    df.to_csv("debug_log.csv", index=False, encoding="utf-8-sig")
    print(f"\n>>> {len(df)} 行を書き出しました → debug_log.csv")

    print("\n=== Existing Method (Same Tasks) ===")
    print(f"  - #Tasks : {len(content_tasks)}")
    print(f"  - Cache Hit: {avg_existing_hits}")
    print(f"  - Success Rate: {existing_success_rate:.2f}%")
    print(f"  - Avg Hops: {avg_existing_hops:.2f}")

    for content_idx in range(NUM_CONTENTS_TO_SEARCH):
        print(f"=== [Comparison] Content #{content_idx+1} ===")
        # stat_s, stat_an_a = gather_stats_for_content(content_idx, single_all_runs, attrib_noreset_all_adapt)
        # plot_metrics_for_content(stat_s, stat_an_a, content_label=content_idx+1)

    # 総合結果の統計量を計算してプロット（元のまま）
    overall_stats = {
        'Single': gather_overall_stats(single_all_runs),
        'NR-Adapt': gather_overall_stats(attrib_noreset_all_adapt),
    }
    plot_overall_metrics(overall_stats)



if __name__ == "__main__":
    main()
