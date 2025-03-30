####### compare_search_single_multi_pheromone_aco.py #######
###　既存手法、多次元フェロモン、単一フェロモンの性能比較

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# 共通のパラメータ設定
TIME_TO_SIMULATE = 5  # シミュレーション回数
TIMES_TO_SEARCH = 50  # 検索回数
TIMES_TO_SEARCH_HOP = 50  # 検索時の最大ホップ数
TIMES_TO_CACHE_HOP = 10  # キャッシュ時の最大ホップ数
TIME_TO_CACHE_PER_CONTENT = 10  # 各コンテンツごとのキャッシュ回数
LEARNING_RATE = 0.5  # 学習率

CACHE_CAPACITY = 20  # キャッシュ容量（未使用）
VECTOR_INCREMENT = 0.1  # ベクトルの増分

###   CSVデータのインポート   ###
file_path = "/Users/asaken-n51/Documents/master_code/test/500_movies.csv"
df = pd.read_csv(file_path)

# ベクトル要素の数（'id'列を除く）
N = len(df.columns) - 1  # Nは属性の数を表す
# コンテンツの総数
cont_num = len(df)
# コンテンツのベクトルデータを定義
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = []
for i in range(cont_num):
    cont_vector_array.append(np.array(cont_vector[i]))

# 隣接ノードの定義（斜めを含む8方向）
def get_neighbors(x, y, size):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

# キャッシュストレージの初期化
def get_init_cache_storage(size):
    cache_storage = np.empty((size, size), dtype=object)
    for i in range(size):
        for j in range(size):
            cache_storage[i][j] = []
    return cache_storage

# ネットワークベクトルの初期化
def get_init_network_vector(size):
    incr = VECTOR_INCREMENT
    net_vector = np.array([np.arange(0, 1 + incr, incr) for _ in range(N)])
    net_vector_array = np.random.choice(net_vector.flatten(), (size, size, N))
    return net_vector_array

# フェロモンレベルの初期化（多次元フェロモンマップ）
def initialize_pheromone_trails(size, num_attributes):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            neighbors = get_neighbors(x, y, size)
            for neighbor in neighbors:
                edge = (curr_node, neighbor)
                pheromone_trails[edge] = np.ones(num_attributes)  # 属性ごとのフェロモン値を1.0で初期化
    return pheromone_trails

# フェロモンレベルの初期化（単一フェロモン）
def initialize_single_pheromone_trails(size):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            neighbors = get_neighbors(x, y, size)
            for neighbor in neighbors:
                edge = (curr_node, neighbor)
                pheromone_trails[edge] = 1.0  # 単一フェロモン値を1.0で初期化
    return pheromone_trails

###   キャッシュ配置   ###
def cache_prop(size):
    cache_storage = get_init_cache_storage(size)
    net_vector_array = get_init_network_vector(size)
    cache_num = TIME_TO_CACHE_PER_CONTENT
    cache_hops = TIMES_TO_CACHE_HOP
    alpha_zero = LEARNING_RATE

    for _ in range(cache_num):
        for id in range(1, cont_num + 1):
            curr = (np.random.randint(size), np.random.randint(size))
            # vectは選択されたコンテンツのベクトル
            vect = cont_vector_array[id - 1]
            hoped_node = []

            for _ in range(cache_hops):
                min_dist = float('inf')
                closest = None
                neighbors = get_neighbors(curr[0], curr[1], size)
                for neighbor in neighbors:
                    if neighbor not in hoped_node:
                        neighbor_vect = net_vector_array[neighbor]
                        dist = np.linalg.norm(vect - neighbor_vect)
                        if dist < min_dist:
                            min_dist = dist
                            closest = neighbor
                if closest is None:
                    break

                # 最も近いノードに移動
                hoped_node.append(curr)
                curr = closest

            # 最後のノードにキャッシュを保存
            hoped_node.append(curr)
            # キャッシュをストレージに保存
            cache_storage[curr[0]][curr[1]].append(id)

            tmp = 0
            total_hops = len(hoped_node)
            # ネットワークベクトルを更新
            for node in hoped_node:
                tmp += 1
                alpha = alpha_zero * (tmp / total_hops)
                net_vector_array[node] += alpha * (vect - net_vector_array[node])

    return cache_storage, net_vector_array

###   検索タスクの事前生成   ###
def generate_search_tasks(size):
    search_tasks = []
    for _ in range(TIMES_TO_SEARCH):
        id = np.random.randint(1, cont_num + 1)  # 検索するコンテンツID
        start_node = (np.random.randint(size), np.random.randint(size))
        search_tasks.append((id, start_node))
    return search_tasks

###   蟻コロニー最適化を用いたコンテンツの探索（多次元フェロモン）   ###
def search_prop_with_aco(cache_storage, net_vector_array, size, search_tasks):
    # ACOのパラメータ
    num_ants = 10          # 蟻の数
    num_iterations = 100   # イテレーション数
    alpha = 1.0           # フェロモンの重要度
    beta = 2.0            # ヒューリスティック情報の重要度
    rho = 0.1             # フェロモンの蒸発率
    Q = 100               # フェロモンの増加量

    num_attributes = N    # 属性の数

    cache_hit = 0
    max_hops = TIMES_TO_SEARCH_HOP

    total_hops_list = []
    theoretical_hops_list = []

    # フェロモンレベルを初期化（多次元フェロモン）
    pheromone_trails = initialize_pheromone_trails(size, num_attributes)

    for search_iter, (id, start_node) in enumerate(search_tasks):
        vect = cont_vector_array[id - 1]

        # コンテンツがキャッシュされているノードを取得
        content_nodes = []
        for x in range(size):
            for y in range(size):
                if id in cache_storage[x][y]:
                    content_nodes.append((x, y))

        if not content_nodes:
            continue  # コンテンツがどこにもキャッシュされていない場合はスキップ

        # 最も近いコンテンツノードを選択（理論最短経路計算用）
        min_dist = float('inf')
        closest_content_node = None
        for node in content_nodes:
            dx = abs(node[0] - start_node[0])
            dy = abs(node[1] - start_node[1])
            dist = max(dx, dy)  # チェビシェフ距離
            if dist < min_dist:
                min_dist = dist
                closest_content_node = node
        theoretical_hops_list.append(min_dist)

        all_paths = []
        all_costs = []
        success = False

        for iteration in range(num_iterations):
            for ant in range(num_ants):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                total_cost = 0
                hops = 0

                while hops < max_hops:
                    if id in cache_storage[current_node[0]][current_node[1]]:
                        # コンテンツを発見
                        success = True
                        break
                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
                    if not allowed_neighbors:
                        break  # 訪問可能なノードがない場合

                    # 移動確率を計算
                    probabilities = []
                    denominator = 0.0
                    for neighbor in allowed_neighbors:
                        edge = (current_node, neighbor)
                        tau_list = pheromone_trails[edge]  # 属性ごとのフェロモン値
                        neighbor_vect = net_vector_array[neighbor]
                        # ヒューリスティック情報 η_{ij} を計算（ベクトルの類似度）
                        distance = np.linalg.norm(vect - neighbor_vect)
                        eta = 1.0 / (distance + 1e-6)  # ゼロ除算を防ぐ

                        # 属性ごとの評価値の計算
                        attribute_scores = (tau_list ** alpha) * vect  # vect = 属性ベクトル

                        # 属性ごとの評価値の総和を計算
                        sum_attribute_scores = np.sum(attribute_scores)

                        # ノードの総合評価値を計算
                        score = sum_attribute_scores * (eta ** beta)

                        probabilities.append(score)
                        denominator += score

                    if denominator == 0:
                        break  # これ以上移動できない場合
                    probabilities = [prob / denominator for prob in probabilities]
                    # 次のノードを選択
                    next_node = random.choices(allowed_neighbors, weights=probabilities)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    total_cost += 1  # 移動あたりのコストは1
                    hops += 1
                    current_node = next_node

                    # 現在のノードでコンテンツをチェック
                    if id in cache_storage[current_node[0]][current_node[1]]:
                        # コンテンツを発見
                        success = True
                        break

                if success:
                    # コンテンツを発見
                    all_paths.append(path)
                    all_costs.append(total_cost)
                    success = False  # 次の蟻の探索のためリセット

        # フェロモンの蒸発と更新
        if all_paths:
            # フェロモンの蒸発
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - rho)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)  # 最小値を設定

            # フェロモンの更新（成功した全ての経路に対して）
            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    delta_tau = (Q * vect) / cost  # コストに反比例し、属性ごとに増加量を計算
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        pheromone_trails[edge] += delta_tau

            cache_hit += 1
            min_cost = min(all_costs)
            total_hops_list.append(min_cost)
        else:
            total_hops_list.append(max_hops)  # コンテンツが見つからなかった場合

    average_hops = sum(total_hops_list) / len(total_hops_list) if total_hops_list else 0
    average_theoretical_hops = sum(theoretical_hops_list) / len(theoretical_hops_list) if theoretical_hops_list else 0
    cache_hit_rate = 100 * (cache_hit / TIMES_TO_SEARCH)
    return cache_hit, cache_hit_rate, average_hops, average_theoretical_hops

###   単一フェロモンを用いたACOを用いたコンテンツ探索   ###
def search_prop_with_single_aco(cache_storage, net_vector_array, size, search_tasks):
    # ACOのパラメータ
    num_ants = 10          # 蟻の数
    num_iterations = 100   # イテレーション数
    alpha = 1.0           # フェロモンの重要度
    beta = 2.0            # ヒューリスティック情報の重要度
    rho = 0.1             # フェロモンの蒸発率
    Q = 100               # フェロモンの増加量

    cache_hit = 0
    max_hops = TIMES_TO_SEARCH_HOP

    total_hops_list = []
    theoretical_hops_list = []

    for search_iter, (id, start_node) in enumerate(search_tasks):
        vect = cont_vector_array[id - 1]

        # フェロモンレベルを初期化（単一フェロモン）
        pheromone_trails = initialize_single_pheromone_trails(size)

        # コンテンツがキャッシュされているノードを取得
        content_nodes = []
        for x in range(size):
            for y in range(size):
                if id in cache_storage[x][y]:
                    content_nodes.append((x, y))

        if not content_nodes:
            continue  # コンテンツがどこにもキャッシュされていない場合はスキップ

        # 最も近いコンテンツノードを選択（理論最短経路計算用）
        min_dist = float('inf')
        closest_content_node = None
        for node in content_nodes:
            dx = abs(node[0] - start_node[0])
            dy = abs(node[1] - start_node[1])
            dist = max(dx, dy)  # チェビシェフ距離
            if dist < min_dist:
                min_dist = dist
                closest_content_node = node
        theoretical_hops_list.append(min_dist)

        all_paths = []
        all_costs = []
        success = False

        for iteration in range(num_iterations):
            for ant in range(num_ants):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                total_cost = 0
                hops = 0

                while hops < max_hops:
                    if id in cache_storage[current_node[0]][current_node[1]]:
                        # コンテンツを発見
                        success = True
                        break
                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
                    if not allowed_neighbors:
                        break  # 訪問可能なノードがない場合

                    # 移動確率を計算
                    probabilities = []
                    denominator = 0.0
                    for neighbor in allowed_neighbors:
                        edge = (current_node, neighbor)
                        tau = pheromone_trails[edge]  # 単一フェロモン値
                        neighbor_vect = net_vector_array[neighbor]
                        # ヒューリスティック情報 η_{ij} を計算（ベクトルの類似度）
                        distance = np.linalg.norm(vect - neighbor_vect)
                        eta = 1.0 / (distance + 1e-6)  # ゼロ除算を防ぐ

                        # ノードの総合評価値を計算
                        score = (tau ** alpha) * (eta ** beta)

                        probabilities.append(score)
                        denominator += score

                    if denominator == 0:
                        break  # これ以上移動できない場合
                    probabilities = [prob / denominator for prob in probabilities]
                    # 次のノードを選択
                    next_node = random.choices(allowed_neighbors, weights=probabilities)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    total_cost += 1  # 移動あたりのコストは1
                    hops += 1
                    current_node = next_node

                    # 現在のノードでコンテンツをチェック
                    if id in cache_storage[current_node[0]][current_node[1]]:
                        # コンテンツを発見
                        success = True
                        break

                if success:
                    # コンテンツを発見
                    all_paths.append(path)
                    all_costs.append(total_cost)
                    success = False  # 次の蟻の探索のためリセット

        # フェロモンの蒸発と更新
        if all_paths:
            # フェロモンの蒸発
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - rho)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)  # 最小値を設定

            # フェロモンの更新（成功した全ての経路に対して）
            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    delta_tau = Q / cost  # コストに反比例し、単一フェロモンの増加量を計算
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        pheromone_trails[edge] += delta_tau

            cache_hit += 1
            min_cost = min(all_costs)
            total_hops_list.append(min_cost)
        else:
            total_hops_list.append(max_hops)  # コンテンツが見つからなかった場合

    average_hops = sum(total_hops_list) / len(total_hops_list) if total_hops_list else 0
    average_theoretical_hops = sum(theoretical_hops_list) / len(theoretical_hops_list) if theoretical_hops_list else 0
    cache_hit_rate = 100 * (cache_hit / TIMES_TO_SEARCH)
    return cache_hit, cache_hit_rate, average_hops, average_theoretical_hops

###   オリジナルのコンテンツ探索関数（比較用）   ###
def search_prop(cache_storage, net_vector_array, size, search_tasks):
    cache_hit = 0
    search_hops = TIMES_TO_SEARCH_HOP
    total_hops_list = []
    theoretical_hops_list = []

    for id, start_node in search_tasks:
        searched_node = []
        curr = start_node
        searched = False
        hops = 0

        # コンテンツがキャッシュされているノードを取得
        content_nodes = []
        for x in range(size):
            for y in range(size):
                if id in cache_storage[x][y]:
                    content_nodes.append((x, y))

        if not content_nodes:
            continue  # コンテンツがどこにもキャッシュされていない場合はスキップ

        # 最も近いコンテンツノードを選択（理論最短経路計算用）
        min_dist = float('inf')
        closest_content_node = None
        for node in content_nodes:
            dx = abs(node[0] - curr[0])
            dy = abs(node[1] - curr[1])
            dist = max(dx, dy)  # チェビシェフ距離
            if dist < min_dist:
                min_dist = dist
                closest_content_node = node
        theoretical_hops_list.append(min_dist)

        for _ in range(search_hops):
            hops += 1
            neighbors = get_neighbors(curr[0], curr[1], size)
            if id in cache_storage[curr[0]][curr[1]]:
                searched = True
                break

            for neighbor in neighbors:
                if id in cache_storage[neighbor[0]][neighbor[1]]:
                    searched = True
                    break
            if searched:
                break

            min_dist = float('inf')
            closest = None
            vect = cont_vector_array[id - 1]

            for neighbor in neighbors:
                if neighbor not in searched_node:
                    neighbor_vect = net_vector_array[neighbor]
                    dist = np.linalg.norm(vect - neighbor_vect)
                    if dist < min_dist:
                        min_dist = dist
                        closest = neighbor

            searched_node.append(curr)
            if all(element in searched_node for element in neighbors):
                break
            curr = closest

        if searched:
            cache_hit += 1
            total_hops_list.append(hops)
        else:
            total_hops_list.append(search_hops)  # コンテンツが見つからなかった場合

    average_hops = sum(total_hops_list) / len(total_hops_list) if total_hops_list else 0
    average_theoretical_hops = sum(theoretical_hops_list) / len(theoretical_hops_list) if theoretical_hops_list else 0
    cache_hit_rate = 100 * (cache_hit / TIMES_TO_SEARCH)
    return cache_hit, cache_hit_rate, average_hops, average_theoretical_hops

###   メイン関数   ###
def main():
    network_sizes = [50]
    original_average_hops_list = []
    aco_average_hops_list = []
    single_aco_average_hops_list = []  # 単一フェロモン用
    theoretical_hops_list = []

    for size in network_sizes:
        temp_original_hops = []
        temp_aco_hops = []
        temp_single_aco_hops = []  # 単一フェロモン用
        temp_original_cache_hits = []
        temp_aco_cache_hits = []
        temp_single_aco_cache_hits = []  # 単一フェロモン用
        temp_original_cache_hit_rates = []
        temp_aco_cache_hit_rates = []
        temp_single_aco_cache_hit_rates = []  # 単一フェロモン用
        temp_original_theoretical_hops = []
        temp_aco_theoretical_hops = []
        temp_single_aco_theoretical_hops = []  # 単一フェロモン用

        print(f"\n===== ネットワークサイズ: {size} =====")
        for i in range(TIME_TO_SIMULATE):
            cache_storage, net_vector_array = cache_prop(size)
            # 検索タスクの生成
            search_tasks = generate_search_tasks(size)

            # オリジナルの探索
            cache_hit, cache_hit_rate, average_hops, average_theoretical_hops = search_prop(
                cache_storage, net_vector_array, size, search_tasks)
            temp_original_hops.append(average_hops)
            temp_original_cache_hits.append(cache_hit)
            temp_original_cache_hit_rates.append(cache_hit_rate)
            temp_original_theoretical_hops.append(average_theoretical_hops)
            print(f"-----     {i+1} 回目のシミュレーション（オリジナル探索）     -----")
            print(f"キャッシュヒット数: {cache_hit}")
            print(f"キャッシュヒット率: {cache_hit_rate} %")
            print(f"平均ホップ数: {average_hops}")
            print(f"理論上の平均最短ホップ数: {average_theoretical_hops}")

            # ACOアルゴリズムによる探索（多次元フェロモン）
            cache_hit_aco, cache_hit_rate_aco, average_hops_aco, average_theoretical_hops_aco = search_prop_with_aco(
                cache_storage, net_vector_array, size, search_tasks)
            temp_aco_hops.append(average_hops_aco)
            temp_aco_cache_hits.append(cache_hit_aco)
            temp_aco_cache_hit_rates.append(cache_hit_rate_aco)
            temp_aco_theoretical_hops.append(average_theoretical_hops_aco)
            print(f"-----     {i+1} 回目のシミュレーション（多次元フェロモンACO探索）     -----")
            print(f"キャッシュヒット数: {cache_hit_aco}")
            print(f"キャッシュヒット率: {cache_hit_rate_aco} %")
            print(f"平均ホップ数: {average_hops_aco}")
            print(f"理論上の平均最短ホップ数: {average_theoretical_hops_aco}")

            # 単一フェロモンACOによる探索
            cache_hit_single_aco, cache_hit_rate_single_aco, average_hops_single_aco, average_theoretical_hops_single_aco = search_prop_with_single_aco(
                cache_storage, net_vector_array, size, search_tasks)
            temp_single_aco_hops.append(average_hops_single_aco)
            temp_single_aco_cache_hits.append(cache_hit_single_aco)
            temp_single_aco_cache_hit_rates.append(cache_hit_rate_single_aco)
            temp_single_aco_theoretical_hops.append(average_theoretical_hops_single_aco)
            print(f"-----     {i+1} 回目のシミュレーション（単一フェロモンACO探索）     -----")
            print(f"キャッシュヒット数: {cache_hit_single_aco}")
            print(f"キャッシュヒット率: {cache_hit_rate_single_aco} %")
            print(f"平均ホップ数: {average_hops_single_aco}")
            print(f"理論上の平均最短ホップ数: {average_theoretical_hops_single_aco}")

        # 各ネットワークサイズでの平均を計算
        original_average_hops = sum(temp_original_hops) / len(temp_original_hops)
        aco_average_hops = sum(temp_aco_hops) / len(temp_aco_hops)
        single_aco_average_hops = sum(temp_single_aco_hops) / len(temp_single_aco_hops)  # 単一フェロモン用
        original_average_cache_hits = sum(temp_original_cache_hits) / len(temp_original_cache_hits)
        aco_average_cache_hits = sum(temp_aco_cache_hits) / len(temp_aco_cache_hits)
        single_aco_average_cache_hits = sum(temp_single_aco_cache_hits) / len(temp_single_aco_cache_hits)  # 単一フェロモン用
        original_average_cache_hit_rates = sum(temp_original_cache_hit_rates) / len(temp_original_cache_hit_rates)
        aco_average_cache_hit_rates = sum(temp_aco_cache_hit_rates) / len(temp_aco_cache_hit_rates)
        single_aco_average_cache_hit_rates = sum(temp_single_aco_cache_hit_rates) / len(temp_single_aco_cache_hit_rates)  # 単一フェロモン用
        average_theoretical_hops = sum(temp_original_theoretical_hops) / len(temp_original_theoretical_hops)

        # 結果の表示
        print(f"\n----- オリジナル探索の結果 -----")
        print(f"平均キャッシュヒット数: {original_average_cache_hits}")
        print(f"平均キャッシュヒット率: {original_average_cache_hit_rates} %")
        print(f"平均ホップ数: {original_average_hops}")
        print(f"平均理論最短ホップ数: {average_theoretical_hops}")
        print(f"----- ACO探索（多次元フェロモン）の結果 -----")
        print(f"平均キャッシュヒット数: {aco_average_cache_hits}")
        print(f"平均キャッシュヒット率: {aco_average_cache_hit_rates} %")
        print(f"平均ホップ数: {aco_average_hops}")
        print(f"平均理論最短ホップ数: {average_theoretical_hops}")
        print(f"----- ACO探索（単一フェロモン）の結果 -----")
        print(f"平均キャッシュヒット数: {single_aco_average_cache_hits}")
        print(f"平均キャッシュヒット率: {single_aco_average_cache_hit_rates} %")
        print(f"平均ホップ数: {single_aco_average_hops}")
        print(f"平均理論最短ホップ数: {average_theoretical_hops}")

        # グラフ作成用のリストに追加
        original_average_hops_list.append(original_average_hops)
        aco_average_hops_list.append(aco_average_hops)
        single_aco_average_hops_list.append(single_aco_average_hops)  # 単一フェロモン用
        theoretical_hops_list.append(average_theoretical_hops)

    # グラフの作成
    plt.figure(figsize=(12, 8))
    plt.plot(network_sizes, original_average_hops_list, label='Original Search', marker='o')
    plt.plot(network_sizes, aco_average_hops_list, label='ACO Search with Attribute Pheromones', marker='s')
    plt.plot(network_sizes, single_aco_average_hops_list, label='ACO Search with Single Pheromone', marker='d')  # 単一フェロモン用
    plt.plot(network_sizes, theoretical_hops_list, label='Theoretical Minimum', marker='^')
    plt.xlabel('Network Size (Grid Length)')
    plt.ylabel('Average Hops')
    plt.title('Comparison of Average Hops: Original, ACO (Attribute Pheromones), ACO (Single Pheromone), and Theoretical Minimum')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
