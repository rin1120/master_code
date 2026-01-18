# ----------------------------------------------------------------------------
# Proposed methods only (NO existing method)
# - Map count sweep: num_maps = 1..5
# - ICN update: after all maps finish for the task, apply once (stickiness prevention)
# - Output: avg best_hops and success rate per num_maps
# - Plot: x=num_maps, y=avg_best_hops / success_rate
# - キャッシュ移動が探索終了毎に逐次行われる
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import japanize_matplotlib  # optional: remove if you don't need Japanese fonts
from collections import namedtuple, defaultdict
import copy

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
TIME_TO_SIMULATE = 100
NUM_CONTENTS_TO_SEARCH = 100
NUM_ANTS = 10
NUM_ITERATIONS = 10

ALPHA_START = 1.0
BETA_START = 10.0
ALPHA_END = 1.0
BETA_END = 1.0
Q = 100

RHO       = 0.10
RHO_MAX   = 0.9
LAMBDA_R  = 1
BOOST     = 5

TIMES_TO_SEARCH_HOP = 100
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

USE_FIXED_START_NODE = False

USE_EPSILON = True
EPSILON = 0.01
EPSILON_SOM = 0.20

# ----------------------------------------------------------------------------
# Load content vectors
# ----------------------------------------------------------------------------
file_path = "500_movies.csv"
df = pd.read_csv(file_path)
N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec) for vec in cont_vector]

# (Optional) genre->cids (kept if you later want genre-based tasks)
non_genre_cols = ['id', 'release_date', 'revenue', 'runtime']
genre_cols = [col for col in df.columns if col not in non_genre_cols]
genre_to_cids = defaultdict(list)
for genre_name in genre_cols:
    cids_in_genre = df[df[genre_name] == 1]['id'].tolist()
    if cids_in_genre:
        genre_to_cids[genre_name] = cids_in_genre

# ----------------------------------------------------------------------------
# Debug record (lightweight)
# ----------------------------------------------------------------------------
DebugRec = namedtuple("DebugRec",
                      ["method", "cid", "start", "found",
                       "hops", "cost", "path"])

# ----------------------------------------------------------------------------
# Network helpers
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

# ----------------------------------------------------------------------------
# Cache placement (per num_maps)
# ----------------------------------------------------------------------------
def cache_prop(size, num_maps):
    cache_storage = get_init_cache_storage(size)
    net_vector_arrays = [get_init_network_vector(size) for _ in range(num_maps)]

    placements_per_map = max(1, TIME_TO_CACHE_PER_CONTENT // max(1, num_maps))
    cache_hops = TIMES_TO_CACHE_HOP
    alpha_zero = LEARNING_RATE

    for map_index in range(num_maps):
        current_net_vector_array = net_vector_arrays[map_index]
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
# ICN update (apply once per task after all maps finish)
# ----------------------------------------------------------------------------
def apply_icn_update(cache_storage, net_vector_arrays, cid, vect, start_node, path_from_found_to_start):
    if cid not in cache_storage[start_node[0]][start_node[1]]:
        cache_storage[start_node[0]][start_node[1]].append(cid)

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
# Task generation (random)
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Proposed method 1: single pheromone per content id
# ----------------------------------------------------------------------------
def run_single_task_single_pheromone(cache_storage, net_vector_array, size, cid, start_node, log_list, pheromone_dict):
    if cid not in pheromone_dict:
        pheromone_dict[cid] = initialize_single_pheromone_trails(size)
    pheromone_trails = pheromone_dict[cid]

    content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
    if not content_nodes:
        return None, TIMES_TO_SEARCH_HOP

    vect = cont_vector_array[cid - 1]

    global_best_path_from_found_to_start = None
    global_min_hops_in_task = float('inf')
    best_cost = TIMES_TO_SEARCH_HOP

    for t in range(NUM_ITERATIONS):
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
                log_list.append(DebugRec("SINGLE", cid, start_node, current_node, cost, cost, path))

                if cost < global_min_hops_in_task:
                    global_min_hops_in_task = cost
                    global_best_path_from_found_to_start = path[::-1]

        if all_costs:
            iteration_best = min(all_costs)
            if iteration_best < best_cost:
                best_cost = iteration_best

        for edge in pheromone_trails:
            pheromone_trails[edge] *= (1 - RHO)
            pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)

        if best_cost > 0:
            for pth, cst in zip(all_paths, all_costs):
                delta_p = max(0.0, (cst / best_cost) - 1.0)
                if delta_p == 0.0:
                    continue
                rho_eff   = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                extra_mul = 1.0 - (rho_eff - RHO)
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] = max(pheromone_trails[(a, b)] * extra_mul, 1e-6)

        for pth, cst in zip(all_paths, all_costs):
            if cst > 0:
                Q_eff = Q * BOOST if cst <= best_cost else Q
                delta = Q_eff / cst
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] += delta

    best_hops = global_min_hops_in_task if global_best_path_from_found_to_start is not None else TIMES_TO_SEARCH_HOP
    return global_best_path_from_found_to_start, best_hops

# ----------------------------------------------------------------------------
# Proposed method 2: attribute pheromone per edge (kept across tasks, per map)
# ----------------------------------------------------------------------------
def run_single_task_attrib_pheromone(cache_storage, net_vector_array, size, cid, start_node, log_list, pheromone_trails):
    content_nodes = [(x, y) for x in range(size) for y in range(size) if cid in cache_storage[x][y]]
    if not content_nodes:
        return None, TIMES_TO_SEARCH_HOP

    vect = cont_vector_array[cid - 1]

    global_best_path_from_found_to_start = None
    global_min_hops_in_task = float('inf')
    best_cost = TIMES_TO_SEARCH_HOP

    for t in range(NUM_ITERATIONS):
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
                            tau_vec = pheromone_trails[edge]
                            dist = np.linalg.norm(vect - net_vector_array[n])
                            eta = 1.0 / (dist + 1e-6)

                            if np.sum(vect) > 0:
                                A_ij = np.sum((tau_vec ** ALPHA) * vect) / np.sum(vect)
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
                log_list.append(DebugRec("ATTRIB_NR", cid, start_node, current_node, cost, cost, path))

                if cost < global_min_hops_in_task:
                    global_min_hops_in_task = cost
                    global_best_path_from_found_to_start = path[::-1]

        if all_costs:
            iteration_best = min(all_costs)
            if iteration_best < best_cost:
                best_cost = iteration_best

        for edge in pheromone_trails:
            pheromone_trails[edge] *= (1 - RHO)
            pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

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

        for pth, cst in zip(all_paths, all_costs):
            if cst > 0 and np.sum(vect) > 0:
                Q_eff = Q * BOOST if cst <= best_cost else Q
                delta = (Q_eff * vect) / (cst * np.sum(vect))
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] += delta

    best_hops = global_min_hops_in_task if global_best_path_from_found_to_start is not None else TIMES_TO_SEARCH_HOP
    return global_best_path_from_found_to_start, best_hops

# ----------------------------------------------------------------------------
# One run for a given num_maps
# ----------------------------------------------------------------------------
def run_experiment(num_maps, size=50, seed=None, save_debug_csv=False):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    init_cache_storage, init_net_vector_arrays = cache_prop(size, num_maps)
    content_tasks = generate_multi_contents_tasks_r(size, k=NUM_CONTENTS_TO_SEARCH)

    debug_logs = []

    cache_for_single = copy.deepcopy(init_cache_storage)
    vectors_for_single = copy.deepcopy(init_net_vector_arrays)

    cache_for_attrib = copy.deepcopy(init_cache_storage)
    vectors_for_attrib = copy.deepcopy(init_net_vector_arrays)

    single_pheromone_dict_list = [{} for _ in range(num_maps)]
    attrib_pheromone_trails_list = [initialize_pheromone_trails(size, N) for _ in range(num_maps)]

    best_hops_single_list = []
    best_hops_attrib_list = []

    for (cid, start_node) in content_tasks:
        vect = cont_vector_array[cid - 1]

        # method 1
        best_path_single = None
        best_hops_single = float('inf')
        for map_index in range(num_maps):
            path_map, hops_map = run_single_task_single_pheromone(
                cache_for_single,
                vectors_for_single[map_index],
                size,
                cid,
                start_node,
                debug_logs,
                single_pheromone_dict_list[map_index]
            )
            if path_map is not None and hops_map < best_hops_single:
                best_hops_single = hops_map
                best_path_single = path_map

        if best_path_single is not None:
            apply_icn_update(cache_for_single, vectors_for_single, cid, vect, start_node, best_path_single)
        best_hops_single_list.append(best_hops_single)

        # method 2
        best_path_attrib = None
        best_hops_attrib = float('inf')
        for map_index in range(num_maps):
            path_map, hops_map = run_single_task_attrib_pheromone(
                cache_for_attrib,
                vectors_for_attrib[map_index],
                size,
                cid,
                start_node,
                debug_logs,
                attrib_pheromone_trails_list[map_index]
            )
            if path_map is not None and hops_map < best_hops_attrib:
                best_hops_attrib = hops_map
                best_path_attrib = path_map

        if best_path_attrib is not None:
            apply_icn_update(cache_for_attrib, vectors_for_attrib, cid, vect, start_node, best_path_attrib)
        if not np.isfinite(best_hops_attrib):
            best_hops_attrib = TIMES_TO_SEARCH_HOP
        best_hops_attrib_list.append(best_hops_attrib)

    def _avg(xs):
        return float(np.mean(xs)) if len(xs) > 0 else float('nan')

    def _succ_rate(xs):
        if len(xs) == 0:
            return 0.0
        succ = sum(1 for h in xs if h < TIMES_TO_SEARCH_HOP)
        return 100.0 * succ / len(xs)

    result = {
        "num_maps": num_maps,
        "single_avg_best_hops": _avg(best_hops_single_list),
        "single_success_rate": _succ_rate(best_hops_single_list),
        "attrib_avg_best_hops": _avg(best_hops_attrib_list),
        "attrib_success_rate": _succ_rate(best_hops_attrib_list),
    }

    if save_debug_csv:
        df_log = pd.DataFrame(debug_logs)
        df_log.to_csv(f"debug_log_maps{num_maps}.csv", index=False, encoding="utf-8-sig")
        print(f">>> debug_log_maps{num_maps}.csv saved: {len(df_log)} rows")

    return result

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_maps_results(df_sum):
    x = df_sum["num_maps"].tolist()
    font_size = 18

    # 1) Avg best hops
    plt.figure()
    plt.plot(x, df_sum["single_avg_best_hops"].tolist(), marker="o", label="提案手法1(ID毎)", color='red')
    plt.plot(x, df_sum["attrib_avg_best_hops"].tolist(), marker="s", label="提案手法2(属性毎)", color='blue')
    plt.xlabel("Number of Maps", fontsize=font_size)
    plt.ylabel("Average Number of Hops", fontsize=font_size)
    plt.xticks(x, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.show()

    # 2) Success rate
    plt.figure()
    plt.plot(x, df_sum["single_success_rate"].tolist(), marker="o", label="提案手法1(ID毎)", color='red')
    plt.plot(x, df_sum["attrib_success_rate"].tolist(), marker="s", label="提案手法2(属性毎)", color='blue')
    plt.xlabel("Number of Maps", fontsize=font_size)
    plt.ylabel("Success Rate (%)", fontsize=font_size)
    plt.xticks(x, fontsize=font_size)
    plt.ylim(0, 100)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# main: sweep num_maps = 1..5
# ----------------------------------------------------------------------------
def main():
    size = 50
    save_debug_csv = False

    all_results = []
    for num_maps in range(1, 6):
        r = run_experiment(num_maps=num_maps, size=size, seed=None, save_debug_csv=save_debug_csv)
        all_results.append(r)

        print(f"\n=== maps={num_maps} ===")
        print(f"[SINGLE] avg best hops: {r['single_avg_best_hops']:.2f}, success: {r['single_success_rate']:.2f}%")
        print(f"[ATTRIB] avg best hops: {r['attrib_avg_best_hops']:.2f}, success: {r['attrib_success_rate']:.2f}%")

    df_sum = pd.DataFrame(all_results)
    df_sum.to_csv("summary_maps_1to5.csv", index=False, encoding="utf-8-sig")
    print("\n>>> summary saved: summary_maps_1to5.csv")

    plot_maps_results(df_sum)

if __name__ == "__main__":
    main()
