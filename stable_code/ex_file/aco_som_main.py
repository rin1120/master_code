####### compare_iterations_and_network_sizes_with_shared_tasks_fixed.py #######
import numpy as np
import pandas as pd
import random
import copy  # ディープコピー用
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
file_path = "500_movies.csv"  # CSVファイルのパスを適切に設定してください
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
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
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
def search_prop_with_aco(cache_storage, net_vector_array, size, search_tasks, num_iterations):
    # ACOのパラメータ
    num_ants = 10  # 蟻の数
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
        for node in content_nodes:
            dx = abs(node[0] - start_node[0])
            dy = abs(node[1] - start_node[1])
            dist = max(dx, dy)  # チェビシェフ距離
            if dist < min_dist:
                min_dist = dist
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
    return average_hops, average_theoretical_hops

###   メイン関数   ###
def main():
    network_sizes = [10, 20, 30, 40, 50]
    iteration_counts = [10, 20, 30]
    results = {}  # {(size, iterations): average_hops}
    theoretical_hops_dict = {}

    # 事前に検索タスクを生成し、各シミュレーションで共有
    shared_search_tasks = {}
    shared_cache_storage = {}
    shared_net_vector_array = {}

    for size in network_sizes:
        # キャッシュ配置とネットワークベクトルも同一条件にする場合
        cache_storage, net_vector_array = cache_prop(size)
        shared_cache_storage[size] = cache_storage
        shared_net_vector_array[size] = net_vector_array

        # 検索タスクを生成
        search_tasks = generate_search_tasks(size)
        shared_search_tasks[size] = search_tasks

    for size in network_sizes:
        print(f"\n===== ネットワークサイズ: {size} =====")
        theoretical_hops_list = []
        for num_iterations in iteration_counts:
            temp_aco_hops = []
            temp_theoretical_hops = []
            print(f"\n--- イテレーション数: {num_iterations} ---")
            for i in range(TIME_TO_SIMULATE):
                # 同一のキャッシュ配置とネットワークベクトル、検索タスクをディープコピー
                cache_storage = copy.deepcopy(shared_cache_storage[size])
                net_vector_array = copy.deepcopy(shared_net_vector_array[size])
                search_tasks = copy.deepcopy(shared_search_tasks[size])
                # ACOアルゴリズムによる探索
                average_hops_aco, average_theoretical_hops = search_prop_with_aco(
                    cache_storage, net_vector_array, size, search_tasks, num_iterations)
                temp_aco_hops.append(average_hops_aco)
                temp_theoretical_hops.append(average_theoretical_hops)
                print(f"シミュレーション {i+1}: 平均ホップ数 = {average_hops_aco}, 理論最短ホップ数 = {average_theoretical_hops}")

            # 平均値を計算
            aco_average_hops = sum(temp_aco_hops) / len(temp_aco_hops)
            average_theoretical_hops = sum(temp_theoretical_hops) / len(temp_theoretical_hops)
            results[(size, num_iterations)] = aco_average_hops

            # 理論ホップ数はネットワークサイズにのみ依存するため、一度だけ保存
            if num_iterations == iteration_counts[0]:
                theoretical_hops_dict[size] = average_theoretical_hops

            print(f"\nネットワークサイズ {size}, イテレーション数 {num_iterations} の結果:")
            print(f"平均ホップ数: {aco_average_hops}")
            print(f"理論上の平均最短ホップ数: {average_theoretical_hops}")

    # グラフの作成
    plt.figure(figsize=(10, 6))
    for num_iterations in iteration_counts:
        average_hops_list = []
        for size in network_sizes:
            average_hops_list.append(results[(size, num_iterations)])
        plt.plot(network_sizes, average_hops_list, label=f'Iterations={num_iterations}', marker='o')
    # 理論ホップ数のプロット
    theoretical_hops_list = [theoretical_hops_dict[size] for size in network_sizes]
    plt.plot(network_sizes, theoretical_hops_list, label='Theoretical Minimum', marker='^', linestyle='--')
    plt.xlabel('Network Size (Grid Length)')
    plt.ylabel('Average Hops')
    plt.title('Effect of Iterations on Average Hops for Different Network Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
