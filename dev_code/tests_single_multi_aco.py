import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# -----------------------
# パラメータ設定
# -----------------------
TIME_TO_SIMULATE = 10       # シミュレーション回数
TIMES_TO_SEARCH = 1        # 検索回数
TIMES_TO_SEARCH_HOP = 50
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

# ACOパラメータ（共通）
NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
Q = 100

file_path = "500_movies.csv"  # 適宜修正
df = pd.read_csv(file_path)

N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec) for vec in cont_vector]

USE_FIXED_START_NODE = True  # Trueなら要求元ノード固定、Falseなら要求元ノード変動

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
            cache_storage[curr[0]][curr[1]].append(cid)

            tmp = 0
            total_hops = len(hoped_node)
            for node in hoped_node:
                tmp += 1
                alpha = alpha_zero * (tmp / total_hops)
                net_vector_array[node] += alpha * (vect - net_vector_array[node])

    return cache_storage, net_vector_array

def generate_search_tasks_fixed(size):
    # 要求元ノードを固定（例：ネットワーク中央付近）
    start_node = (size // 2, size // 2)
    cid = np.random.randint(1, cont_num + 1)
    return [(cid, start_node)]

def generate_search_tasks_variable(size):
    # 要求元ノードを毎回変動（従来と同じ）
    search_tasks = []
    for _ in range(TIMES_TO_SEARCH):
        cid = np.random.randint(1, cont_num + 1)
        start_node = (np.random.randint(size), np.random.randint(size))
        search_tasks.append((cid, start_node))
    return search_tasks

def search_prop_with_aco(cache_storage, net_vector_array, size, search_tasks):
    num_attributes = N
    max_hops = TIMES_TO_SEARCH_HOP
    iteration_costs_data = []

    for (cid, start_node) in search_tasks:
        vect = cont_vector_array[cid - 1]

        content_nodes = []
        for x in range(size):
            for y in range(size):
                if cid in cache_storage[x][y]:
                    content_nodes.append((x, y))
        if not content_nodes:
            continue

        pheromone_trails = initialize_pheromone_trails(size, num_attributes)

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

                while hops < max_hops:
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
                    iteration_all_costs.append(max_hops)

            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            if all_paths:
                for path, cost in zip(all_paths, all_costs):
                    if cost > 0:
                        delta_tau = (Q * vect) / cost
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i + 1])
                            pheromone_trails[edge] += delta_tau

            iteration_costs_data.append(iteration_all_costs)

    return iteration_costs_data

def search_prop_with_single_aco(cache_storage, net_vector_array, size, search_tasks):
    max_hops = TIMES_TO_SEARCH_HOP
    iteration_costs_data = []

    for (cid, start_node) in search_tasks:
        vect = cont_vector_array[cid - 1]

        content_nodes = []
        for x in range(size):
            for y in range(size):
                if cid in cache_storage[x][y]:
                    content_nodes.append((x, y))
        if not content_nodes:
            continue

        pheromone_trails = initialize_single_pheromone_trails(size)

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

                while hops < max_hops:
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
                    iteration_all_costs.append(max_hops)

            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)

            if all_paths:
                for path, cost in zip(all_paths, all_costs):
                    if cost > 0:
                        delta_tau = Q / cost
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i + 1])
                            pheromone_trails[edge] += delta_tau

            iteration_costs_data.append(iteration_all_costs)

    return iteration_costs_data

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
        success_rate = (success_count / NUM_ANTS) * 100
        medians.append(median_val)
        q1s.append(q1_val)
        q3s.append(q3_val)
        means.append(mean_val)
        success_rates.append(success_rate)
    return medians, q1s, q3s, means, success_rates

def main():
    size = 50
    cache_storage, net_vector_array = cache_prop(size)

    # 要求元ノード固定 or 変動
    if USE_FIXED_START_NODE:
        search_tasks = generate_search_tasks_fixed(size)
    else:
        search_tasks = generate_search_tasks_variable(size)

    attrib_all_runs_medians = []
    attrib_all_runs_q1s = []
    attrib_all_runs_q3s = []
    attrib_all_runs_means = []
    attrib_all_runs_success = []

    single_all_runs_medians = []
    single_all_runs_q1s = []
    single_all_runs_q3s = []
    single_all_runs_means = []
    single_all_runs_success = []

    for _ in range(TIME_TO_SIMULATE):
        attrib_aco_data = search_prop_with_aco(cache_storage, net_vector_array, size, search_tasks)
        single_aco_data = search_prop_with_single_aco(cache_storage, net_vector_array, size, search_tasks)

        attrib_median, attrib_q1, attrib_q3, attrib_mean, attrib_succ = compute_stats_from_costs(attrib_aco_data)
        single_median, single_q1, single_q3, single_mean, single_succ = compute_stats_from_costs(single_aco_data)

        attrib_all_runs_medians.append(attrib_median)
        attrib_all_runs_q1s.append(attrib_q1)
        attrib_all_runs_q3s.append(attrib_q3)
        attrib_all_runs_means.append(attrib_mean)
        attrib_all_runs_success.append(attrib_succ)

        single_all_runs_medians.append(single_median)
        single_all_runs_q1s.append(single_q1)
        single_all_runs_q3s.append(single_q3)
        single_all_runs_means.append(single_mean)
        single_all_runs_success.append(single_succ)

    num_iterations = NUM_ITERATIONS

    def average_over_runs(data_list):
        data_array = np.array(data_list)
        return np.mean(data_array, axis=0)

    attrib_final_median = average_over_runs(attrib_all_runs_medians)
    attrib_final_q1 = average_over_runs(attrib_all_runs_q1s)
    attrib_final_q3 = average_over_runs(attrib_all_runs_q3s)
    attrib_final_mean = average_over_runs(attrib_all_runs_means)
    attrib_final_success = average_over_runs(attrib_all_runs_success)

    single_final_median = average_over_runs(single_all_runs_medians)
    single_final_q1 = average_over_runs(single_all_runs_q1s)
    single_final_q3 = average_over_runs(single_all_runs_q3s)
    single_final_mean = average_over_runs(single_all_runs_means)
    single_final_success = average_over_runs(single_all_runs_success)

    iterations = range(1, num_iterations + 1)

    # グラフ表示
    plt.figure()
    plt.plot(iterations, attrib_final_median, label='Attr ACO Median', color='blue', marker='o')
    plt.fill_between(iterations, attrib_final_q1, attrib_final_q3, color='blue', alpha=0.2, label='Attr ACO Q1-Q3')
    plt.plot(iterations, single_final_median, label='Single ACO Median', color='red', marker='s')
    plt.fill_between(iterations, single_final_q1, single_final_q3, color='red', alpha=0.2, label='Single ACO Q1-Q3')
    plt.title("Median and Quartiles per Iteration (Averaged over Runs)")
    plt.xlabel("Iteration")
    plt.ylabel("Hops")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(iterations, attrib_final_success, label="Attr ACO Success Rate", marker='o')
    plt.plot(iterations, single_final_success, label="Single ACO Success Rate", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate per Iteration (Averaged over Runs)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iterations, attrib_final_mean, label="Attr ACO Avg Cost", marker='o')
    plt.plot(iterations, single_final_mean, label="Single ACO Avg Cost", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Average Cost (Hops)")
    plt.title("Convergence over Iterations (Averaged over Runs)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 関数定義を追加
    def generate_search_tasks_fixed(size):
        # 開始ノード固定の例：ネットワーク中央
        start_node = (size // 2, size // 2)
        cid = np.random.randint(1, cont_num + 1)
        return [(cid, start_node)]

    def generate_search_tasks_variable(size):
        # 要求元ノード変動（従来）
        search_tasks = []
        for _ in range(TIMES_TO_SEARCH):
            cid = np.random.randint(1, cont_num + 1)
            start_node = (np.random.randint(size), np.random.randint(size))
            search_tasks.append((cid, start_node))
        return search_tasks

    main()
