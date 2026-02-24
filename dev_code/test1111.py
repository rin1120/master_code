# 山代法＋比較手法（コンテンツ毎にフェロモン管理）＋提案手法（属性毎にフェロモン管理）
#比較手法、提案手法どちらにもローカルベストの探索を実装
#探索性能は全ての探索を反映（成功時と失敗時の両方）
#コンテンツ生成手法をランダム、zipf、類似の3種類実装
#iter=1を山白法＋ランダム探索,iter>1以降でSOMとACOの組み合わせを実施（パラメータ可変）
#パラメータ0は暫定版
#SOMを５つに拡張、フェロモンはマップ毎に保持、共有しない

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
TIME_TO_SIMULATE = 1
NUM_CONTENTS_TO_SEARCH = 10  # 探索するコンテンツ数
NUM_ANTS = 10
NUM_ITERATIONS = 10

ALPHA_START = 1.0   # 初期フェロモンの指数
BETA_START = 10.0    # 初期SOM類似度の指数
ALPHA_END = 1.0   # 最終フェロモンの指数
BETA_END = 1.0    # 最終SOM類似度の指数
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
NUM_MAPS = 5

USE_FIXED_START_NODE = False

# ε-greedy戦略用パラメータ（リセットなしの場合のみ使用）
USE_EPSILON = True
EPSILON = 0.01  # 10%の確率でランダム選択
EPSILON_SOM = 0.20  # 20%の確率でランダム選択


# CSVファイルから属性ベクトルを準備
file_path = "500_movies.csv"  # 適宜修正
#file_path = "10000_movies.csv"  # 適宜修正
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
        #各マップにキャッシュを配置
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
# コンテンツ探索タスク生成
# ----------------------------------------------------------------------------
#r = random , z = zipf, s = similar

######(1)ランダム要求
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

#####(2)zipf要求 modeで人気・不人気・混合を選択可能
def generate_multi_contents_tasks_z(size, k=NUM_CONTENTS_TO_SEARCH, popularity_ratio=0.01, request_concentration=0.8, mode="mix"):
    tasks = []
    
    # 人気コンテンツを決定（元のまま）
    num_popular = int(cont_num * popularity_ratio)
    all_cids = np.arange(1, cont_num + 1)
    popular_cids = np.random.choice(all_cids, num_popular, replace=False) if num_popular > 0 else np.array([], dtype=int)
    unpopular_cids = np.setdiff1d(all_cids, popular_cids) if num_popular > 0 else all_cids

    # 互換のため：modeは "mix" / "popular_only" / "unpopular_only"
    mode = str(mode).lower()

    for _ in range(k):
        # request_concentration の確率で人気コンテンツから選ぶ（"mix" のときのみ）
        if mode == "popular_only":
            # 人気のみから選択（空なら全体にフォールバック）
            if len(popular_cids) > 0:
                cid = int(random.choice(popular_cids))
            else:
                cid = int(random.choice(all_cids))
        elif mode == "unpopular_only":
            # 不人気のみから選択（空なら全体にフォールバック）
            if len(unpopular_cids) > 0:
                cid = int(random.choice(unpopular_cids))
            else:
                cid = int(random.choice(all_cids))
        else:
            # 従来の挙動（mix）
            if random.random() < request_concentration and len(popular_cids) > 0:
                cid = int(random.choice(popular_cids))
            else:
                # unpopular が空のケースを安全に処理
                if len(unpopular_cids) > 0:
                    cid = int(random.choice(unpopular_cids))
                else:
                    cid = int(random.choice(all_cids))
            
        start_node = (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start_node))
    return tasks


####(3)類似要求
# def generate_multi_contents_tasks_s(size, k=NUM_CONTENTS_TO_SEARCH, burst_size=1000):
#     tasks = []
#     available_genres = list(genre_to_cids.keys())
#     if not available_genres:
#         raise ValueError("ジャンル情報が見つかりませんでした。")
    
#     while len(tasks) < k:
#         current_genre = random.choice(available_genres)
#         cids_in_genre = genre_to_cids[current_genre]
#         for _ in range(burst_size):
#             if len(tasks) >= k: break
#             cid = random.choice(cids_in_genre)
#             start_node = (np.random.randint(size), np.random.randint(size))
#             tasks.append((cid, start_node))
#     return tasks

def generate_multi_contents_tasks_s(size, k=NUM_CONTENTS_TO_SEARCH, burst_size=1000):
    """
    アンカーをランダムに1つ選び、ユークリッド距離（SOM距離）で近い順に候補を作成。
    候補上位から 1/距離 の重みで k 件サンプリングして要求列を返す。
    """
    # 属性行列
    V = np.vstack(cont_vector_array).astype(float)  # (cont_num, N)

    # アンカーをランダムに選択
    anchor_idx = np.random.randint(cont_num)  # 0-based
    anchor_vec = V[anchor_idx]

    # 全件とのユークリッド距離（小さいほど類似）
    dist = np.linalg.norm(V - anchor_vec, axis=1)
    dist[anchor_idx] = np.inf  # 自分自身は除外

    # 上位M(=近い方から)を候補に
    M = int(max(1, min(burst_size, cont_num - 1)))
    idx_top = np.argpartition(dist, M)[:M]  # 最小M件
    cand_cids = (idx_top + 1).astype(int)

    # 重み = 1/距離（0割防止）。総和0にはならない想定だが念のため正規化
    w = 1.0 / (dist[idx_top] + 1e-12)
    p = w / w.sum()

    # k件サンプリング（重複可）
    sampled_cids = np.random.choice(cand_cids, size=k, replace=True, p=p)

    # 既存の start_node ロジックを踏襲
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
    means, success_rates =  [], []
    for costs in iteration_costs_data:
        means.append(np.mean(costs))
        total = len(costs)
        success_count = sum(1 for c in costs if c < TIMES_TO_SEARCH_HOP)
        success_rates.append((success_count / total) * 100)
    return means, success_rates

# ----------------------------------------------------------------------------
# デバッグ用レコード（CSV 行にそのまま使う）        ★
# ----------------------------------------------------------------------------
DebugRec = namedtuple("DebugRec",
                      ["method", "cid", "start", "found",
                       "hops", "cost", "path"])


# ----------------------------------------------------------------------------
#生成タスクの可視化
# ----------------------------------------------------------------------------
def visualize_task_distribution(tasks, title):
    """生成されたタスクのコンテンツIDの分布を可視化する"""
    if not tasks:
        print("タスクリストが空です。")
        return
        
    cids = [task[0] for task in tasks]
    counts = pd.Series(cids).value_counts()
    
    plt.figure(figsize=(15, 6))
    counts.plot(kind='bar', width=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel("コンテンツID")
    
    # 【変更】Y軸のラベルを「要求回数」に変更
    plt.ylabel("要求回数")
    
    # x軸のラベルが多すぎると見にくいので、表示を調整
    if len(counts) > 50:
        plt.xticks([]) # 50個以上ならラベルを非表示に
    else:
        plt.xticks(rotation=90, fontsize=8)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7) # y軸にグリッドを追加して見やすく
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# 既存手法による探索
# ----------------------------------------------------------------------------
def search_prop(cache_storage, net_vector_arrays, size, content_tasks, log_list):
    cache_hit = 0
    total_hops_used = 0
    for (cid, start_node) in content_tasks:
        #5マップ並列探索
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
            # #デバッグ用レコードを追加
            # log_list.append(DebugRec("BASE", cid, start_node,curr, hops_used, hops_used, []))

        total_hops_used += min_hops

    avg_hops = total_hops_used / len(content_tasks) if content_tasks else 0.0
    return cache_hit, avg_hops


# ----------------------------------------------------------------------------
# ① 提案手法1（コンテンツ毎にフェロモン管理）
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
                    # === Iter.0 は純SOM（既存方式）に ε-greedy を導入 ===
                    searched_node = []
                    curr = start_node
                    cost = 0
                    path = [curr]                  

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
                                break
                        if found and hit_neighbor:
                            break

                        # --- ε-greedy 選択ロジック ---
                        
                        # 1. 訪問可能（未訪問）な隣接ノードのリストを取得
                        allowed_neighbors = [
                            n for n in get_neighbors(curr[0], curr[1], size)
                            if n not in searched_node
                        ]

                        if not allowed_neighbors:
                            # 行き先がない（袋小路）
                            break

                        # 2. 貪欲法（exploitation）のための「最も近い」ノードを探す
                        min_dist = float('inf')
                        closest = None
                        for neighbor in allowed_neighbors:
                            dist = np.linalg.norm(vect - net_vector_array[neighbor])
                            if dist < min_dist:
                                min_dist = dist
                                closest = neighbor
                        
                        if closest is None:
                            # 念のため（allowed_neighborsが空でなければ通常ここには来ない）
                            break
                        
                        # 3. ε-greedy による最終決定
                        next_node = None
                        if USE_EPSILON and random.random() < EPSILON_SOM:
                            # (A) 探索 (Exploration) ： 確率 ε でランダムに選ぶ
                            next_node = random.choice(allowed_neighbors)
                        else:
                            # (B) 搾取 (Exploitation)： 確率 1-ε で最も近いノードを選ぶ
                            next_node = closest
                        # --- ロジックここまで ---

                        searched_node.append(curr)
                        
                        # 決定したノードへ移動
                        curr = next_node  # <--- `closest` ではなく `next_node` に変更
                        path.append(curr)                  
                        
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True
                            break
                    current_node = curr
                else:
                    # === 以降は現行の確率選択(ACO)そのまま ===                
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
                        # フェロモンパラメータの調整
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
# ② 提案手法2（属性毎にフェロモン管理）
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
                    # === Iter.0 は純SOM（既存方式）に ε-greedy を導入 ===
                    searched_node = []
                    curr = start_node
                    cost = 0
                    path = [curr]                  

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
                                break
                        if found and hit_neighbor:
                            break

                        # --- ε-greedy 選択ロジック ---
                        
                        # 1. 訪問可能（未訪問）な隣接ノードのリストを取得
                        allowed_neighbors = [
                            n for n in get_neighbors(curr[0], curr[1], size)
                            if n not in searched_node
                        ]

                        if not allowed_neighbors:
                            # 行き先がない（袋小路）
                            break

                        # 2. 貪欲法（exploitation）のための「最も近い」ノードを探す
                        min_dist = float('inf')
                        closest = None
                        for neighbor in allowed_neighbors:
                            dist = np.linalg.norm(vect - net_vector_array[neighbor])
                            if dist < min_dist:
                                min_dist = dist
                                closest = neighbor
                        
                        if closest is None:
                            # 念のため（allowed_neighborsが空でなければ通常ここには来ない）
                            break
                        
                        # 3. ε-greedy による最終決定
                        next_node = None
                        if USE_EPSILON and random.random() < EPSILON_SOM:
                            # (A) 探索 (Exploration) ： 確率 ε でランダムに選ぶ
                            next_node = random.choice(allowed_neighbors)
                        else:
                            # (B) 搾取 (Exploitation)： 確率 1-ε で最も近いノードを選ぶ
                            next_node = closest
                        # --- ロジックここまで ---

                        searched_node.append(curr)
                        
                        # 決定したノードへ移動
                        curr = next_node  # <--- `closest` ではなく `next_node` に変更
                        path.append(curr)                  
                        
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True
                            break
                    current_node = curr
                else:
                    # === 以降は現行の確率選択(ACO)そのまま ===
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
                        # フェロモンパラメータの調整
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

# ----------------------------------------------------------------------------
# 複数マップ並列探索
# ----------------------------------------------------------------------------
# ★★★ 新規追加関数 ★★★
def merge_parallel_results(map_results_list):
    """
    5つの並列探索結果（[map][task][iter][costs]）を受け取り、
    統計処理用の単一の結果（[task][iter][costs_50_ants]）に合算する。
    """
    if not map_results_list:
        return []

    num_maps = len(map_results_list)
    if num_maps == 0:
        return []
        
    # 
    num_tasks = len(map_results_list[0])
    if num_tasks == 0:
        return []
        
    num_iters = len(map_results_list[0][0])
    
    # 最終的な結果を格納するリストを初期化
    final_results = [ [ [] for _ in range(num_iters)] for _ in range(num_tasks)]
    
    for task_idx in range(num_tasks):
        for iter_idx in range(num_iters):
            # このタスク/イテレーションの全マップ（50匹）のコストを合算
            merged_costs = []
            for map_idx in range(num_maps):
                # [map_idx][task_idx][iter_idx] のコストリストを取得して合算
                if len(map_results_list[map_idx]) > task_idx and \
                   len(map_results_list[map_idx][task_idx]) > iter_idx:
                    merged_costs.extend(map_results_list[map_idx][task_idx][iter_idx])
            
            final_results[task_idx][iter_idx] = merged_costs
            
    return final_results

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
    s_mean, s_succ = stat_single
    an_a_mean, an_a_succ = stat_attrib_noreset_adpat
    its = range(1, NUM_ITERATIONS + 1)
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
    overall_iter_costs = []          # 「最優秀チームの平均コスト」を格納
    overall_iter_success_rates = []  # 「最優秀チームの成功率(%)」を格納
    for it in range(NUM_ITERATIONS):
        all_costs_for_this_iter = []
        all_success_rates_for_this_iter = []
        for sim_run in all_runs:
            # sim_runは各コンテンツ結果のリスト
            for content_result in sim_run:
                if len(content_result) > it:
                    costs_50_ants = content_result[it]
                    if costs_50_ants: 
                        team_costs_list = [
                            costs_50_ants[i:i + NUM_ANTS] 
                            for i in range(0, len(costs_50_ants), NUM_ANTS)
                        ]
                        # 各チーム（マップ）の平均コストを計算
                        team_avg_costs = []
                        team_success_rates_list = []
                        for team_costs in team_costs_list:
                            if team_costs: # チームのコストリストが空でない
                                team_avg_costs.append(np.mean(team_costs))
                                success_count = sum(1 for c in team_costs if c < TIMES_TO_SEARCH_HOP)
                                team_success_rates_list.append((success_count / len(team_costs)) * 100.0)
                            else:
                                team_avg_costs.append(TIMES_TO_SEARCH_HOP) # 失敗扱い
                                team_success_rates_list.append(0.0)
                        # 5チームの平均のうち、最良（min）のものを採用
                        # ★「最優秀チーム（平均コストmin）」のインデックスを取得
                        best_team_index = np.argmin(team_avg_costs)
                        # ★「最優秀チームの平均コスト」を取得
                        best_team_avg_cost = team_avg_costs[best_team_index]
                        # ★「最優秀チームの成功率(%)」を取得
                        best_team_success_rate = team_success_rates_list[best_team_index]
                        
                        all_costs_for_this_iter.append(best_team_avg_cost)
                        all_success_rates_for_this_iter.append(best_team_success_rate)
                    else: # costs_50_ants が空の場合（＝集計漏れ防止）
                        all_costs_for_this_iter.append(TIMES_TO_SEARCH_HOP) # 失敗扱い
                        all_success_rates_for_this_iter.append(0.0) # 失敗扱い


        overall_iter_costs.append(all_costs_for_this_iter)
        overall_iter_success_rates.append(all_success_rates_for_this_iter)

    cost_stats = compute_stats_from_costs(overall_iter_costs)
    success_stats = compute_stats_from_costs(overall_iter_success_rates)

    return cost_stats[0], success_stats[0]

def plot_overall_metrics(stats_dict):
    # stats_dict: {'Single': stats, 'AttribReset': stats, 'NR-Adapt': stats, 'NR-Base': stats}
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
    plt.ylim(0,100)
    plt.show()


def main():
    size = 50
    cache_storage, net_vector_arrays = cache_prop(size)
    #コンテンツ生成方法を選択
    content_tasks = generate_multi_contents_tasks_r(size, k=NUM_CONTENTS_TO_SEARCH)
    print("Generated Content Tasks:", content_tasks)
    unique_cids = len(set(cid for cid, _ in content_tasks))
    print(f"生成コンテンツ種類数: {unique_cids}/{cont_num}")

    debug_logs = []

    single_all_runs = []
    attrib_noreset_all_adapt = []

    print("コンテンツ生成の分布を可視化します...")
    visualize_task_distribution(content_tasks, "Task Distribution")

    single_pheromone_dict_list = [{} for _ in range(NUM_MAPS)]

    for _sim in range(TIME_TO_SIMULATE):
        print(f"\n=== Simulation #{_sim + 1} ===")
        sim_single_results_list = []
        sim_attrib_results_list = []

        for map_index in range(NUM_MAPS):
            print(f"  --- Map #{map_index} ---")
            current_map_vector = net_vector_arrays[map_index]
            current_pheromone_dict = single_pheromone_dict_list[map_index] # マップ固有の辞書
            
            single_res = multi_contents_single_pheromone_with_reset(cache_storage, current_map_vector, size, content_tasks, debug_logs, current_pheromone_dict)
            attrib_res = multi_contents_attrib_pheromone_no_reset(cache_storage, current_map_vector, size, content_tasks, debug_logs)
            
            sim_single_results_list.append(single_res)
            sim_attrib_results_list.append(attrib_res)

        merged_single_res = merge_parallel_results(sim_single_results_list)
        merged_attrib_res = merge_parallel_results(sim_attrib_results_list)
        single_all_runs.append(merged_single_res)
        attrib_noreset_all_adapt.append(merged_attrib_res)


    cache_hit, avg_hops = search_prop(cache_storage, net_vector_arrays, size, content_tasks, debug_logs)
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

        # stat_s, stat_an_a = gather_stats_for_content(content_idx, single_all_runs, attrib_noreset_all_adapt)
        # plot_metrics_for_content(stat_s, stat_an_a, content_label=content_idx+1)
    #総合結果の統計量を計算してプロット
    overall_stats = {
        'Single': gather_overall_stats(single_all_runs),
        'NR-Adapt': gather_overall_stats(attrib_noreset_all_adapt),
    }
    plot_overall_metrics(overall_stats)



if __name__ == "__main__":
    main()