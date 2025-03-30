import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# パラメータ設定
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 3
NUM_CONTENTS_TO_SEARCH = 2
NUM_ANTS = 10
NUM_ITERATIONS = 100

ALPHA = 1.0
BETA = 1.0
RHO = 0.1
Q = 100

TIMES_TO_SEARCH_HOP = 50
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

USE_FIXED_START_NODE = False

# CSVファイルから属性ベクトルを準備
file_path = "500_movies.csv"  # 適宜修正
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
    dist = np.linalg.norm(vect_a - vect_b)
    return dist

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
    net_vector_array = np.random.choice(net_vector.flatten(), (size, size, N))
    return net_vector_array

def initialize_pheromone_trails(size, num_attributes):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            neighbors = get_neighbors(x, y, size)
            for neighbor in neighbors:
                edge = (curr_node, neighbor)
                pheromone_trails[edge] = np.ones(num_attributes)
    return pheromone_trails

def initialize_single_pheromone_trails(size):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            neighbors = get_neighbors(x, y, size)
            for neighbor in neighbors:
                edge = (curr_node, neighbor)
                pheromone_trails[edge] = 1.0
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
            cache_storage[curr[0]][curr[1]].append(cid)

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
    medians = []
    q1s = []
    q3s = []
    means = []
    success_rates = []

    for costs in iteration_costs_data:
        median_val = np.median(costs)
        q1_val = np.percentile(costs, 25)
        q3_val = np.percentile(costs, 75)
        mean_val = np.mean(costs)

        success_count = sum(1 for c in costs if c < TIMES_TO_SEARCH_HOP)
        total_ants = len(costs)
        success_rate = (success_count / total_ants) * 100
        medians.append(median_val)
        q1s.append(q1_val)
        q3s.append(q3_val)
        means.append(mean_val)
        success_rates.append(success_rate)

    return medians, q1s, q3s, means, success_rates

# ----------------------------------------------------------------------------
# 既存手法による探索
# ----------------------------------------------------------------------------
def search_prop(cache_storage, net_vector_array, size, content_tasks):
    """
    既存手法による探索を、提案手法と同じタスク群 (content_tasks) で実行する版。
    content_tasks: [ (cid, start_node), (cid, start_node), ... ]
    戻り値: (cache_hit, avg_hops)
       - cache_hit: 見つかった回数
       - avg_hops : 1要求あたりの平均ホップ数
    """
    cache_hit = 0
    total_hops_used = 0
    search_hops = TIMES_TO_SEARCH_HOP

    # 全タスク数 ( = len(content_tasks) ) 回だけ探索する
    for (cid, start_node) in content_tasks:
        # 1つの要求につき探索開始
        searched_node = []
        curr = start_node
        searched = False
        hops_used = 0

        for _hop in range(search_hops):
            hops_used += 1

            # 1) 現在ノードあるいは近傍ノードにキャッシュがあれば探索成功
            if cid in cache_storage[curr[0]][curr[1]]:
                searched = True
                break

            neighbors = get_neighbors(curr[0], curr[1], size)
            found_in_neighbor = any(
                (cid in cache_storage[nx][ny]) for (nx, ny) in neighbors
            )
            if found_in_neighbor:
                searched = True
                break

            # 2) 見つからない場合 → 属性ベクトルが最も近い neighbor を選ぶ
            min_dist = float('inf')
            closest = None
            vect = cont_vector_array[cid - 1]

            for neighbor in neighbors:
                if neighbor not in searched_node:
                    neighbor_vect = net_vector_array[neighbor]
                    dist = np.linalg.norm(vect - neighbor_vect)
                    if dist < min_dist:
                        min_dist = dist
                        closest = neighbor

            searched_node.append(curr)
            if all(nb in searched_node for nb in neighbors):
                # 近傍がすべて訪問済みならこれ以上探索不可
                break

            # 次ノードへ移動
            if closest is not None:
                curr = closest
            else:
                # 一応Noneチェック。closestが見つからなかったら探索打ち切り
                break

        if searched:
            cache_hit += 1

        total_hops_used += hops_used

    # 統計量
    num_tasks = len(content_tasks)
    avg_hops = total_hops_used / num_tasks if num_tasks > 0 else 0.0

    return cache_hit, avg_hops


# ----------------------------------------------------------------------------
# 1) 単一フェロモン方式 (リセットあり)
# ----------------------------------------------------------------------------
def multi_contents_single_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks):
    results_for_each_content = []

    for (cid, start_node) in content_tasks:
        pheromone_trails = initialize_single_pheromone_trails(size)

        content_nodes = []
        for x in range(size):
            for y in range(size):
                if cid in cache_storage[x][y]:
                    content_nodes.append((x, y))
        if not content_nodes:
            results_for_each_content.append([])
            continue

        vect = cont_vector_array[cid - 1]
        iteration_costs_data = []

        for iteration in range(NUM_ITERATIONS):
            iteration_all_costs = []
            all_paths = []
            all_costs = []
            for ant in range(NUM_ANTS):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                total_cost = 0
                hops = 0
                found = False

                while hops < TIMES_TO_SEARCH_HOP:
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break

                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed_neighbors = [n for n in neighbors if n not in visited]
                    if not allowed_neighbors:
                        break

                    probabilities = []
                    denominator = 0.0
                    for neighbor in allowed_neighbors:
                        edge = (current_node, neighbor)
                        tau = pheromone_trails[edge]
                        neighbor_vect = net_vector_array[neighbor]
                        distance = np.linalg.norm(vect - neighbor_vect)
                        eta = 1.0 / (distance + 1e-6)
                        score = (tau ** ALPHA) * (eta ** BETA)
                        probabilities.append(score)
                        denominator += score

                    if denominator == 0:
                        break
                    probabilities = [p / denominator for p in probabilities]
                    next_node = random.choices(allowed_neighbors, weights=probabilities)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    total_cost += 1
                    hops += 1
                    current_node = next_node

                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break

                if found:
                    all_paths.append(path)
                    all_costs.append(total_cost)
                    iteration_all_costs.append(total_cost)
                else:
                    iteration_all_costs.append(TIMES_TO_SEARCH_HOP)

            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)

            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    delta_tau = Q / cost
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta_tau

            iteration_costs_data.append(iteration_all_costs)

        results_for_each_content.append(iteration_costs_data)
    return results_for_each_content

# ----------------------------------------------------------------------------
# 2) 属性フェロモン方式 (リセットあり)
# ----------------------------------------------------------------------------
def multi_contents_attrib_pheromone_with_reset(cache_storage, net_vector_array, size, content_tasks):
    results_for_each_content = []

    for (cid, start_node) in content_tasks:
        pheromone_trails = initialize_pheromone_trails(size, N)

        content_nodes = []
        for x in range(size):
            for y in range(size):
                if cid in cache_storage[x][y]:
                    content_nodes.append((x, y))
        if not content_nodes:
            results_for_each_content.append([])
            continue

        vect = cont_vector_array[cid - 1]
        iteration_costs_data = []

        for iteration in range(NUM_ITERATIONS):
            iteration_all_costs = []
            all_paths = []
            all_costs = []
            for ant in range(NUM_ANTS):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                total_cost = 0
                hops = 0
                found = False

                while hops < TIMES_TO_SEARCH_HOP:
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break

                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed_neighbors = [n for n in neighbors if n not in visited]
                    if not allowed_neighbors:
                        break

                    probabilities = []
                    denominator = 0.0
                    for neighbor in allowed_neighbors:
                        edge = (current_node, neighbor)
                        tau_list = pheromone_trails[edge]
                        neighbor_vect = net_vector_array[neighbor]
                        distance = np.linalg.norm(vect - neighbor_vect)
                        eta = 1.0 / (distance + 1e-6)

                        attribute_scores = (tau_list ** ALPHA) * vect
                        sum_attribute_scores = np.sum(attribute_scores)
                        score = sum_attribute_scores * (eta ** BETA)

                        probabilities.append(score)
                        denominator += score

                    if denominator == 0:
                        break

                    probabilities = [p / denominator for p in probabilities]
                    next_node = random.choices(allowed_neighbors, weights=probabilities)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    total_cost += 1
                    hops += 1
                    current_node = next_node

                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break

                if found:
                    all_paths.append(path)
                    all_costs.append(total_cost)
                    iteration_all_costs.append(total_cost)
                else:
                    iteration_all_costs.append(TIMES_TO_SEARCH_HOP)

            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    delta_tau = (Q * vect) / cost
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta_tau

            iteration_costs_data.append(iteration_all_costs)

        results_for_each_content.append(iteration_costs_data)
    return results_for_each_content

# ----------------------------------------------------------------------------
# 3) 属性フェロモン方式 (リセットなし)
# ----------------------------------------------------------------------------
def multi_contents_attrib_pheromone_no_reset(cache_storage, net_vector_array, size, content_tasks):
    pheromone_trails = initialize_pheromone_trails(size, N)
    results_for_each_content = []

    for (cid, start_node) in content_tasks:
        content_nodes = []
        for x in range(size):
            for y in range(size):
                if cid in cache_storage[x][y]:
                    content_nodes.append((x, y))
        if not content_nodes:
            results_for_each_content.append([])
            continue

        vect = cont_vector_array[cid - 1]
        iteration_costs_data = []

        for iteration in range(NUM_ITERATIONS):
            iteration_all_costs = []
            all_paths = []
            all_costs = []
            for ant in range(NUM_ANTS):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                total_cost = 0
                hops = 0
                found = False

                while hops < TIMES_TO_SEARCH_HOP:
                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break

                    neighbors = get_neighbors(current_node[0], current_node[1], size)
                    allowed_neighbors = [n for n in neighbors if n not in visited]
                    if not allowed_neighbors:
                        break

                    probabilities = []
                    denominator = 0.0
                    for neighbor in allowed_neighbors:
                        edge = (current_node, neighbor)
                        tau_list = pheromone_trails[edge]
                        neighbor_vect = net_vector_array[neighbor]
                        distance = np.linalg.norm(vect - neighbor_vect)
                        eta = 1.0 / (distance + 1e-6)

                        attribute_scores = (tau_list ** ALPHA) * vect
                        sum_attribute_scores = np.sum(attribute_scores)
                        score = sum_attribute_scores * (eta ** BETA)

                        probabilities.append(score)
                        denominator += score

                    if denominator == 0:
                        break
                    probabilities = [p / denominator for p in probabilities]
                    next_node = random.choices(allowed_neighbors, weights=probabilities)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    total_cost += 1
                    hops += 1
                    current_node = next_node

                    if cid in cache_storage[current_node[0]][current_node[1]]:
                        found = True
                        break

                if found:
                    all_paths.append(path)
                    all_costs.append(total_cost)
                    iteration_all_costs.append(total_cost)
                else:
                    iteration_all_costs.append(TIMES_TO_SEARCH_HOP)

            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            for path, cost in zip(all_paths, all_costs):
                if cost > 0:
                    delta_tau = (Q * vect) / cost
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta_tau

            iteration_costs_data.append(iteration_all_costs)

        results_for_each_content.append(iteration_costs_data)

    return results_for_each_content

# ----------------------------------------------------------------------------
# メイン関数
# ----------------------------------------------------------------------------
def main():
    size = 50
    cache_storage, net_vector_array = cache_prop(size)

    # 例として2つのコンテンツを探索するタスクを生成
    content_tasks = generate_multi_contents_tasks(size, k=NUM_CONTENTS_TO_SEARCH)
    # 例: [(272, (8, 49)), (109, (49, 40))] のような形で2つ生成される想定
    print("Generated Content Tasks:", content_tasks)

    # ここで A, B を取り出して距離を計算
    # ※ 2つ以上生成されることを前提に、最初の2要素だけ取り出す
    if len(content_tasks) >= 2:
        A = content_tasks[0][0]  # 1つ目のコンテンツID
        B = content_tasks[1][0]  # 2つ目のコンテンツID
        distance = compute_content_distance(A, B)
        print(f"Distance between content {A} and {B} = {distance:.3f}")

    # 3方式の結果を格納
    single_all_runs = []
    attrib_reset_all_runs = []
    attrib_noreset_all_runs = []

    TIME_TO_SIMULATE = 3
    for _sim in range(TIME_TO_SIMULATE):
        single_res = multi_contents_single_pheromone_with_reset(
            cache_storage, net_vector_array, size, content_tasks
        )
        attrib_reset_res = multi_contents_attrib_pheromone_with_reset(
            cache_storage, net_vector_array, size, content_tasks
        )
        attrib_noreset_res = multi_contents_attrib_pheromone_no_reset(
            cache_storage, net_vector_array, size, content_tasks
        )

        single_all_runs.append(single_res)
        attrib_reset_all_runs.append(attrib_reset_res)
        attrib_noreset_all_runs.append(attrib_noreset_res)
        

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
            merged_costs = []
            for one_run in iteration_data_runs:
                merged_costs.extend(one_run[it])
            combined_iteration_data.append(merged_costs)

        return compute_stats_from_costs(combined_iteration_data)

    def gather_stats_for_content(content_index):
        s_stat = average_iteration_data_across_runs(single_all_runs, content_index)
        ar_stat = average_iteration_data_across_runs(attrib_reset_all_runs, content_index)
        an_stat = average_iteration_data_across_runs(attrib_noreset_all_runs, content_index)
        return (s_stat, ar_stat, an_stat)

    def plot_three_metrics_for_content(stat_single, stat_attrib_reset, stat_attrib_noreset, content_label):
        if (stat_single is None) or (stat_attrib_reset is None) or (stat_attrib_noreset is None):
            print(f"No data for content{content_label}")
            return

        s_med, s_q1, s_q3, s_mean, s_succ = stat_single
        ar_med, ar_q1, ar_q3, ar_mean, ar_succ = stat_attrib_reset
        an_med, an_q1, an_q3, an_mean, an_succ = stat_attrib_noreset

        its = range(1, NUM_ITERATIONS + 1)

        # 比較1: 中央値と四分位範囲
        plt.figure()
        plt.plot(its, s_med, label='Single(Reset)', color='red', marker='o')
        plt.fill_between(its, s_q1, s_q3, color='red', alpha=0.2)
        plt.plot(its, ar_med, label='Attrib(Reset)', color='blue', marker='s')
        plt.fill_between(its, ar_q1, ar_q3, color='blue', alpha=0.2)
        plt.plot(its, an_med, label='Attrib(NoReset)', color='green', marker='^')
        plt.fill_between(its, an_q1, an_q3, color='green', alpha=0.2)
        plt.title(f"Content{content_label}: Median & Quartiles")
        plt.xlabel("Iteration")
        plt.ylabel("Hops")
        plt.legend()
        plt.show()

        # 比較2: 成功率
        plt.figure()
        plt.plot(its, s_succ, label='Single(Reset)', color='red', marker='o')
        plt.plot(its, ar_succ, label='Attrib(Reset)', color='blue', marker='s')
        plt.plot(its, an_succ, label='Attrib(NoReset)', color='green', marker='^')
        plt.title(f"Content{content_label}: Success Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Success Rate (%)")
        plt.legend()
        plt.show()

        # 比較3: 平均コスト
        plt.figure()
        plt.plot(its, s_mean, label='Single(Reset)', color='red', marker='o')
        plt.plot(its, ar_mean, label='Attrib(Reset)', color='blue', marker='s')
        plt.plot(its, an_mean, label='Attrib(NoReset)', color='green', marker='^')
        plt.title(f"Content{content_label}: Average Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Average Hops")
        plt.legend()
        plt.show()
    
    # 既存手法で同じタスクを探索
    cache_hit, avg_hops = search_prop(cache_storage, net_vector_array, size, content_tasks)
    success_rate = (cache_hit / len(content_tasks)) * 100
    print("\n=== Existing Method (Same Tasks) ===")
    print(f"  - #Tasks : {len(content_tasks)}")
    print(f"  - Cache Hit: {cache_hit}")
    print(f"  - Success Rate: {success_rate:.2f}%")
    print(f"  - Avg Hops: {avg_hops:.2f}")


    # === [Comparison] Content #1 ===
    print("=== [Comparison] Content #1 ===")
    stat_s_c1, stat_ar_c1, stat_an_c1 = gather_stats_for_content(0)
    plot_three_metrics_for_content(stat_s_c1, stat_ar_c1, stat_an_c1, content_label=1)

    # === [Comparison] Content #2 ===
    if NUM_CONTENTS_TO_SEARCH > 1:
        print("=== [Comparison] Content #2 ===")
        stat_s_c2, stat_ar_c2, stat_an_c2 = gather_stats_for_content(1)
        plot_three_metrics_for_content(stat_s_c2, stat_ar_c2, stat_an_c2, content_label=2)

if __name__ == "__main__":
    main()
