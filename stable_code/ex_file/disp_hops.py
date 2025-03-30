import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math  # 追加

### 定数の定義 ###
TIME_TO_SIMULATE = 10
TIMES_TO_SEARCH = 50
TIMES_TO_SEARCH_HOP = 50
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

### データの読み込み ###
file_path = "/Users/asaken-n51/Documents/master_code/test/500_movies.csv"
df = pd.read_csv(file_path)

N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(cont_vector[i]) for i in range(cont_num)]

### キャッシュストレージの初期化 ###
def get_init_cache_storage(size):
    cache_storage = np.empty((size, size), dtype=object)
    for i in range(size):
        for j in range(size):
            cache_storage[i][j] = []
    return cache_storage

### 初期ベクトルデータの設定 ###
def get_init_network_vector(size):
    incr = VECTOR_INCREMENT
    net_vector = np.array([np.arange(0, 1 + incr, incr) for _ in range(N)])
    net_vector_array = np.random.choice(net_vector.flatten(), (size, size, N))
    return net_vector_array

### 隣接ノードの取得 ###
def get_neighbors(x, y, size):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

### キャッシュプロセス ###
def cache_prop(size):
    cache_storage = get_init_cache_storage(size)
    net_vector_array = get_init_network_vector(size)
    cache_num = TIME_TO_CACHE_PER_CONTENT
    cache_hops = TIMES_TO_CACHE_HOP
    alpha_zero = LEARNING_RATE

    for _ in range(cache_num):
        for id in range(1, cont_num + 1):
            curr = (np.random.randint(size), np.random.randint(size))
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
                hoped_node.append(curr)
                curr = closest

            hoped_node.append(curr)
            cache_storage[curr[0]][curr[1]].append(id)

            tmp = 0
            total_hops = len(hoped_node)
            for node in hoped_node:
                tmp += 1
                alpha = alpha_zero * (tmp / total_hops)
                net_vector_array[node] += alpha * (vect - net_vector_array[node])

    return cache_storage, net_vector_array

### 検索プロセス ###
def search_prop(cache_storage, net_vector_array, size):
    cache_hit = 0
    search_num = TIMES_TO_SEARCH
    search_hops = TIMES_TO_SEARCH_HOP
    hops_to_hit_list = []
    theoretical_hops_to_hit_list = []

    for _ in range(1, search_num + 1):
        id = np.random.randint(1, cont_num)
        searched_node = []
        curr = (np.random.randint(size), np.random.randint(size))
        start = curr
        searched = False
        hops = 0

        for _ in range(search_hops):
            neighbors = get_neighbors(curr[0], curr[1], size)
            hops += 1
            if id in cache_storage[curr[0]][curr[1]]:
                searched = True
                break

            for neighbor in neighbors:
                if id in cache_storage[neighbor[0]][neighbor[1]]:
                    curr = neighbor
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
            hops_to_hit_list.append(hops)
            # オクタイル距離を使用して理論的なホップ数を計算
            dx = abs(curr[0] - start[0])
            dy = abs(curr[1] - start[1])
            theoretical_hops = max(dx, dy)
            theoretical_hops_to_hit_list.append(theoretical_hops)

    return cache_hit, hops_to_hit_list, theoretical_hops_to_hit_list

### メインプロセス ###
def main():
    network_sizes = [10, 20, 30, 40, 50]
    avg_hops_to_hit_results = []
    avg_theoretical_hops_to_hit_results = []

    for size in network_sizes:
        hops_to_hit_index = []
        theoretical_hops_to_hit_index = []

        for _ in range(TIME_TO_SIMULATE):
            cache_storage, net_vector_array = cache_prop(size)
            _, hops_to_hit_list, theoretical_hops_to_hit_list = search_prop(cache_storage, net_vector_array, size)
            avg_hops_to_hit = sum(hops_to_hit_list) / len(hops_to_hit_list) if hops_to_hit_list else 0
            avg_theoretical_hops_to_hit = sum(theoretical_hops_to_hit_list) / len(theoretical_hops_to_hit_list) if theoretical_hops_to_hit_list else 0
            hops_to_hit_index.append(avg_hops_to_hit)
            theoretical_hops_to_hit_index.append(avg_theoretical_hops_to_hit)

        avg_hops_to_hit_results.append(sum(hops_to_hit_index) / len(hops_to_hit_index))
        avg_theoretical_hops_to_hit_results.append(sum(theoretical_hops_to_hit_index) / len(theoretical_hops_to_hit_index))

    # フォントサイズの設定
    X_AXIS_FONT_SIZE = 33  # x軸ラベルのフォントサイズ
    Y_AXIS_FONT_SIZE = 33  # y軸ラベルのフォントサイズ
    TICKS_FONT_SIZE = 30   # 軸目盛りのフォントサイズ
    TITLE_FONT_SIZE = 36   # タイトルのフォントサイズ
    LEGEND_FONT_SIZE = 33  # 凡例のフォントサイズ

    plt.plot(network_sizes, avg_hops_to_hit_results, label='Avg Hops to Hit')
    plt.plot(network_sizes, avg_theoretical_hops_to_hit_results, label='Avg Theoretical Hops to Hit', linestyle='--')
    plt.xlabel('Network Size', fontsize=X_AXIS_FONT_SIZE)
    plt.ylabel('Hops', fontsize=Y_AXIS_FONT_SIZE)
    #plt.title('Average Hops to Hit - Average Theoretical Hops to Hit', fontsize=TITLE_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)  # 凡例のフォントサイズを設定

    # 軸の目盛りとラベルをカスタマイズ
    # x軸の目盛りを設定
    x_ticks = network_sizes  # 表示したいx軸の値を指定
    plt.xticks(x_ticks, fontsize=TICKS_FONT_SIZE)  # フォントサイズを設定

    # y軸の目盛りを設定
    max_y_value = max(avg_hops_to_hit_results + avg_theoretical_hops_to_hit_results)
    step_size = 5
    max_y_value_rounded = math.ceil(max_y_value / step_size) * step_size
    y_ticks = np.arange(0, max_y_value_rounded + step_size, step_size)
    plt.yticks(y_ticks, fontsize=TICKS_FONT_SIZE)  # フォントサイズを設定

    # グリッドを削除
    # plt.grid(True)  # この行をコメントアウトまたは削除

    plt.show()

if __name__ == "__main__":
    main()
