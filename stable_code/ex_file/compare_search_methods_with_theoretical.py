####### compare_search_methods_with_theoretical.py #######
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# 共通のパラメータ設定
TIME_TO_SIMULATE = 3  # シミュレーション回数
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
N = len(df.columns) - 1
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

###   蟻コロニー最適化を用いたコンテンツの探索   ###
def search_prop_with_aco(cache_storage, net_vector_array, size, search_tasks):
    # ACOのパラメータ
    num_ants = 5  # 蟻の数
    num_iterations = 10  # イテレーション数
    alpha_initial = 0.1  # フェロモンの初期重要度
    alpha_final = 1.0    # フェロモンの最終重要度
    beta_initial = 5.0   # ヒューリスティック情報の初期重要度
    beta_final = 1.0     # ヒューリスティック情報の最終重要度
    rho = 0.5    # フェロモンの蒸発率
    Q = 100      # フェロモンの増加量

    cache_hit = 0
    max_hops = TIMES_TO_SEARCH_HOP

    total_hops_list = []
    theoretical_hops_list = []

    for search_iter, (id, start_node) in enumerate(search_tasks):
        vect = cont_vector_array[id - 1]

        # フェロモンレベルを初期化（各コンテンツの探索ごとに初期化）
        pheromone_trails = {}  # 各エッジのフェロモンレベルを格納
        for x in range(size):
            for y in range(size):
                curr_node = (x, y)
                neighbors = get_neighbors(x, y, size)
                for neighbor in neighbors:
                    edge = (curr_node, neighbor)
                    pheromone_trails[edge] = 1.0  # 初期フェロモンレベルを1.0に設定

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

        # alphaとbetaを時間とともに調整
        alpha = alpha_initial + (alpha_final - alpha_initial) * (search_iter / TIMES_TO_SEARCH)
        beta = beta_initial - (beta_initial - beta_final) * (search_iter / TIMES_TO_SEARCH)

        # ACOアルゴリズムを実行
        best_path = None
        best_cost = float('inf')

        for iteration in range(num_iterations):
            all_paths = []
            all_costs = []
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
                        tau = (pheromone_trails[edge] + 1e-6) ** alpha  # フェロモンがゼロの場合を避ける
                        neighbor_vect = net_vector_array[neighbor]
                        dist = np.linalg.norm(vect - neighbor_vect)
                        eta = (1.0 / (dist + 1e-6)) ** beta  # ゼロ除算を避ける
                        probabilities.append(tau * eta)
                        denominator += tau * eta

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
                        break

                if id in cache_storage[current_node[0]][current_node[1]]:
                    # コンテンツを発見
                    all_paths.append(path)
                    all_costs.append(total_cost)
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path

        # フェロモンの蒸発
        for edge in pheromone_trails:
            pheromone_trails[edge] *= (1 - rho)
            if pheromone_trails[edge] < 1e-6:
                pheromone_trails[edge] = 1e-6  # フェロモンレベルがゼロにならないように

        # フェロモンの更新
        for path, cost in zip(all_paths, all_costs):
            if cost == 0:
                pheromone_delta = Q  # 最大のフェロモン増加を割り当て
            else:
                pheromone_delta = Q / cost
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                pheromone_trails[edge] += pheromone_delta

        if best_path is not None:
            cache_hit += 1
            total_hops_list.append(len(best_path) - 1)
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
    network_sizes = [10, 20, 30, 40, 50]
    original_average_hops_list = []
    aco_average_hops_list = []
    theoretical_hops_list = []

    for size in network_sizes:
        temp_original_hops = []
        temp_aco_hops = []
        temp_original_cache_hits = []
        temp_aco_cache_hits = []
        temp_original_cache_hit_rates = []
        temp_aco_cache_hit_rates = []
        temp_original_theoretical_hops = []
        temp_aco_theoretical_hops = []

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

            # ACOアルゴリズムによる探索
            cache_hit_aco, cache_hit_rate_aco, average_hops_aco, average_theoretical_hops_aco = search_prop_with_aco(
                cache_storage, net_vector_array, size, search_tasks)
            temp_aco_hops.append(average_hops_aco)
            temp_aco_cache_hits.append(cache_hit_aco)
            temp_aco_cache_hit_rates.append(cache_hit_rate_aco)
            temp_aco_theoretical_hops.append(average_theoretical_hops_aco)
            print(f"-----     {i+1} 回目のシミュレーション（ACO探索）     -----")
            print(f"キャッシュヒット数: {cache_hit_aco}")
            print(f"キャッシュヒット率: {cache_hit_rate_aco} %")
            print(f"平均ホップ数: {average_hops_aco}")
            print(f"理論上の平均最短ホップ数: {average_theoretical_hops_aco}")

        # 各ネットワークサイズでの平均を計算
        original_average_hops = sum(temp_original_hops) / len(temp_original_hops)
        aco_average_hops = sum(temp_aco_hops) / len(temp_aco_hops)
        original_average_cache_hits = sum(temp_original_cache_hits) / len(temp_original_cache_hits)
        aco_average_cache_hits = sum(temp_aco_cache_hits) / len(temp_aco_cache_hits)
        original_average_cache_hit_rates = sum(temp_original_cache_hit_rates) / len(temp_original_cache_hit_rates)
        aco_average_cache_hit_rates = sum(temp_aco_cache_hit_rates) / len(temp_aco_cache_hit_rates)
        average_theoretical_hops = sum(temp_original_theoretical_hops) / len(temp_original_theoretical_hops)

        # 結果の表示
        print(f"\n----- オリジナル探索の結果 -----")
        print(f"平均キャッシュヒット数: {original_average_cache_hits}")
        print(f"平均キャッシュヒット率: {original_average_cache_hit_rates} %")
        print(f"平均ホップ数: {original_average_hops}")
        print(f"平均理論最短ホップ数: {average_theoretical_hops}")
        print(f"----- ACO探索の結果 -----")
        print(f"平均キャッシュヒット数: {aco_average_cache_hits}")
        print(f"平均キャッシュヒット率: {aco_average_cache_hit_rates} %")
        print(f"平均ホップ数: {aco_average_hops}")
        print(f"平均理論最短ホップ数: {average_theoretical_hops}")

        # グラフ作成用のリストに追加
        original_average_hops_list.append(original_average_hops)
        aco_average_hops_list.append(aco_average_hops)
        theoretical_hops_list.append(average_theoretical_hops)

    # グラフの作成
    plt.figure(figsize=(10, 6))
    plt.plot(network_sizes, original_average_hops_list, label='Original Search', marker='o')
    plt.plot(network_sizes, aco_average_hops_list, label='ACO Search', marker='s')
    plt.plot(network_sizes, theoretical_hops_list, label='Theoretical Minimum', marker='^')
    plt.xlabel('Network Size (Grid Length)')
    plt.ylabel('Average Hops')
    plt.title('Comparison of Average Hops between Original, ACO Searches, and Theoretical Minimum by Network Size')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
