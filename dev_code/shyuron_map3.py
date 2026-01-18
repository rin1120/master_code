# ----------------------------------------------------------------------------
# Experiment 2 (Final Design):
# - FIXED num_maps (e.g., 3)
# - Sweep network size
# - Cache is SHARED among maps (common cache storage)
# - Search is ALWAYS performed ONLY on the COMPATIBLE (best-match) map.
#   (No search on non-selected maps)
# - Compare UPDATE policy:
#     (1) COMPAT-UPDATE: update the same best-match map
#     (2) RANDOM-UPDATE: update one randomly chosen map (may equal best-match)
# - After SUCCESS:
#     * cache copy to start node (shared cache)
#     * SOM weight update along the found path (ONLY on selected update map)
# - Proposed search methods:
#     A) ID-pheromone ACO
#     B) Attribute-pheromone ACO
# - Metric: avg hops (failure treated as TIMES_TO_SEARCH_HOP)
#　探索マップは常に相性マップのみ、並列探索無し、更新マップを変える実験
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import copy
import japanize_matplotlib
from collections import namedtuple

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
NUM_MAPS_FIXED = 3
TIME_TO_SIMULATE = 10
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
cont_vector_array = [np.array(v, dtype=float) for v in df.set_index("id").values.tolist()]

DebugRec = namedtuple("DebugRec", ["method", "policy", "size", "cid", "hops", "search_map_idx", "update_map_idx"])

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_neighbors(x, y, size):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    out = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            out.append((nx, ny))
    return out

def get_init_cache_storage(size):
    cache_storage = np.empty((size, size), dtype=object)
    for i in range(size):
        for j in range(size):
            cache_storage[i][j] = []
    return cache_storage

def get_init_network_vector(size):
    incr = VECTOR_INCREMENT
    net_vector = np.array([np.arange(0, 1 + incr, incr) for _ in range(N)])
    return np.random.choice(net_vector.flatten(), (size, size, N)).astype(float)

def initialize_pheromone_trails(size, num_attributes):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr = (x, y)
            for nb in get_neighbors(x, y, size):
                pheromone_trails[(curr, nb)] = np.ones(num_attributes, dtype=float)
    return pheromone_trails

def initialize_single_pheromone_trails(size):
    pheromone_trails = {}
    for x in range(size):
        for y in range(size):
            curr = (x, y)
            for nb in get_neighbors(x, y, size):
                pheromone_trails[(curr, nb)] = 1.0
    return pheromone_trails

def cache_prop(size, num_maps):
    cache_storage = get_init_cache_storage(size)
    net_vector_arrays = [get_init_network_vector(size) for _ in range(num_maps)]

    placements_per_map = max(1, TIME_TO_CACHE_PER_CONTENT // max(1, num_maps))
    alpha_zero = LEARNING_RATE

    for mi in range(num_maps):
        net_vec = net_vector_arrays[mi]
        for _ in range(placements_per_map):
            for cid in range(1, cont_num + 1):
                curr = (np.random.randint(size), np.random.randint(size))
                vect = cont_vector_array[cid - 1]
                route = []

                for _2 in range(TIMES_TO_CACHE_HOP):
                    allowed = [nb for nb in get_neighbors(curr[0], curr[1], size) if nb not in route]
                    if not allowed:
                        break
                    closest = min(allowed, key=lambda nb: np.linalg.norm(vect - net_vec[nb]))
                    route.append(curr)
                    curr = closest

                route.append(curr)
                cache_storage[curr[0]][curr[1]].append(cid)

                total = len(route)
                for i, node in enumerate(route, start=1):
                    alpha = alpha_zero * (i / total)
                    net_vec[node] += alpha * (vect - net_vec[node])

    return cache_storage, net_vector_arrays

def generate_tasks(size, k):
    tasks = []
    for _ in range(k):
        cid = np.random.randint(1, cont_num + 1)
        if USE_FIXED_START_NODE:
            start = (size // 2, size // 2)
        else:
            start = (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start))
    return tasks

def _alpha_beta(t):
    if NUM_ITERATIONS <= 1:
        return ALPHA_START, BETA_START
    a = ALPHA_START + (ALPHA_END - ALPHA_START) * (t / (NUM_ITERATIONS - 1))
    b = BETA_START  + (BETA_END  - BETA_START)  * (t / (NUM_ITERATIONS - 1))
    return a, b

def _som_greedy_search(cache_storage, net_vector_array, size, cid, start_node, vect):
    searched = []
    curr = start_node
    path = [curr]
    hops = 0

    for _ in range(TIMES_TO_SEARCH_HOP):
        hops += 1

        if cid in cache_storage[curr[0]][curr[1]]:
            return True, curr, path, hops

        for nx, ny in get_neighbors(curr[0], curr[1], size):
            if cid in cache_storage[nx][ny]:
                found = (nx, ny)
                path.append(found)
                return True, found, path, hops

        allowed = [n for n in get_neighbors(curr[0], curr[1], size) if n not in searched]
        if not allowed:
            break

        closest = min(allowed, key=lambda n: np.linalg.norm(vect - net_vector_array[n]))
        if USE_EPSILON and random.random() < EPSILON_SOM:
            nxt = random.choice(allowed)
        else:
            nxt = closest

        searched.append(curr)
        curr = nxt
        path.append(curr)

    return False, curr, path, TIMES_TO_SEARCH_HOP

# ----------------------------------------------------------------------------
# Search methods (Proposed)
# ----------------------------------------------------------------------------
def run_single_task_single_pheromone(cache_storage, net_vector_array, size, cid, start_node, pheromone_dict):
    if cid not in pheromone_dict:
        pheromone_dict[cid] = initialize_single_pheromone_trails(size)
    pheromone_trails = pheromone_dict[cid]
    vect = cont_vector_array[cid - 1]

    best_path_found_to_start = None
    best_hops_in_task = TIMES_TO_SEARCH_HOP
    best_cost_so_far = TIMES_TO_SEARCH_HOP

    for t in range(NUM_ITERATIONS):
        all_paths, all_costs = [], []

        for _ in range(NUM_ANTS):
            if t == 0:
                found, _, path, hops = _som_greedy_search(
                    cache_storage, net_vector_array, size, cid, start_node, vect
                )
                if found:
                    all_paths.append(path)
                    all_costs.append(hops)
                    if hops < best_hops_in_task:
                        best_hops_in_task = hops
                        best_path_found_to_start = path[::-1]
                continue

            visited = {start_node}
            current = start_node
            path = [current]
            cost = 0
            found = False

            for _step in range(TIMES_TO_SEARCH_HOP):
                if cid in cache_storage[current[0]][current[1]]:
                    found = True
                    break
                allowed = [n for n in get_neighbors(current[0], current[1], size) if n not in visited]
                if not allowed:
                    break

                if USE_EPSILON and random.random() < EPSILON:
                    nxt = random.choice(allowed)
                else:
                    ALPHA, BETA = _alpha_beta(t)
                    scores, denom = [], 0.0
                    for n in allowed:
                        tau = pheromone_trails[(current, n)]
                        eta = 1.0 / (np.linalg.norm(vect - net_vector_array[n]) + 1e-6)
                        s = (tau ** ALPHA) * (eta ** BETA)
                        scores.append(s)
                        denom += s
                    if denom == 0.0:
                        break
                    probs = [s / denom for s in scores]
                    nxt = random.choices(allowed, weights=probs)[0]

                visited.add(nxt)
                path.append(nxt)
                cost += 1
                current = nxt

                if cid in cache_storage[current[0]][current[1]]:
                    found = True
                    break

            if found:
                all_paths.append(path)
                all_costs.append(cost)
                if cost < best_hops_in_task:
                    best_hops_in_task = cost
                    best_path_found_to_start = path[::-1]

        if all_costs:
            best_cost_so_far = min(best_cost_so_far, min(all_costs))

        for edge in pheromone_trails:
            pheromone_trails[edge] *= (1 - RHO)
            pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)

        if best_cost_so_far > 0 and all_costs:
            for pth, cst in zip(all_paths, all_costs):
                delta_p = max(0.0, (cst / best_cost_so_far) - 1.0)
                if delta_p == 0.0:
                    continue
                rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                extra_mul = 1.0 - (rho_eff - RHO)
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] = max(pheromone_trails[(a, b)] * extra_mul, 1e-6)

        for pth, cst in zip(all_paths, all_costs):
            if cst <= 0:
                continue
            Q_eff = Q * BOOST if cst <= best_cost_so_far else Q
            delta = Q_eff / cst
            for a, b in zip(pth[:-1], pth[1:]):
                pheromone_trails[(a, b)] += delta

    return best_path_found_to_start, best_hops_in_task

def run_single_task_attrib_pheromone(cache_storage, net_vector_array, size, cid, start_node, pheromone_trails):
    vect = cont_vector_array[cid - 1]
    sum_vect = float(np.sum(vect)) if np.sum(vect) > 0 else 0.0

    best_path_found_to_start = None
    best_hops_in_task = TIMES_TO_SEARCH_HOP
    best_cost_so_far = TIMES_TO_SEARCH_HOP

    for t in range(NUM_ITERATIONS):
        all_paths, all_costs = [], []

        for _ in range(NUM_ANTS):
            if t == 0:
                found, _, path, hops = _som_greedy_search(
                    cache_storage, net_vector_array, size, cid, start_node, vect
                )
                if found:
                    all_paths.append(path)
                    all_costs.append(hops)
                    if hops < best_hops_in_task:
                        best_hops_in_task = hops
                        best_path_found_to_start = path[::-1]
                continue

            visited = {start_node}
            current = start_node
            path = [current]
            cost = 0
            found = False

            for _step in range(TIMES_TO_SEARCH_HOP):
                if cid in cache_storage[current[0]][current[1]]:
                    found = True
                    break
                allowed = [n for n in get_neighbors(current[0], current[1], size) if n not in visited]
                if not allowed:
                    break

                if USE_EPSILON and random.random() < EPSILON:
                    nxt = random.choice(allowed)
                else:
                    ALPHA, BETA = _alpha_beta(t)
                    scores, denom = [], 0.0
                    for n in allowed:
                        tau_vec = pheromone_trails[(current, n)]
                        eta = 1.0 / (np.linalg.norm(vect - net_vector_array[n]) + 1e-6)
                        if sum_vect > 0:
                            Aij = float(np.sum((tau_vec ** ALPHA) * vect) / sum_vect)
                        else:
                            Aij = 0.0
                        s = Aij * (eta ** BETA)
                        scores.append(s)
                        denom += s
                    if denom == 0.0:
                        break
                    probs = [s / denom for s in scores]
                    nxt = random.choices(allowed, weights=probs)[0]

                visited.add(nxt)
                path.append(nxt)
                cost += 1
                current = nxt

                if cid in cache_storage[current[0]][current[1]]:
                    found = True
                    break

            if found:
                all_paths.append(path)
                all_costs.append(cost)
                if cost < best_hops_in_task:
                    best_hops_in_task = cost
                    best_path_found_to_start = path[::-1]

        if all_costs:
            best_cost_so_far = min(best_cost_so_far, min(all_costs))

        for edge in pheromone_trails:
            pheromone_trails[edge] *= (1 - RHO)
            pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

        if best_cost_so_far > 0 and all_costs:
            for pth, cst in zip(all_paths, all_costs):
                delta_p = max(0.0, (cst / best_cost_so_far) - 1.0)
                if delta_p == 0.0:
                    continue
                rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                extra_mul = 1.0 - (rho_eff - RHO)
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] *= extra_mul
                    pheromone_trails[(a, b)] = np.maximum(pheromone_trails[(a, b)], 1e-6)

        if sum_vect > 0:
            for pth, cst in zip(all_paths, all_costs):
                if cst <= 0:
                    continue
                Q_eff = Q * BOOST if cst <= best_cost_so_far else Q
                delta_vec = (Q_eff * vect) / (cst * sum_vect)
                for a, b in zip(pth[:-1], pth[1:]):
                    pheromone_trails[(a, b)] += delta_vec

    return best_path_found_to_start, best_hops_in_task

# ----------------------------------------------------------------------------
# Map selection
# ----------------------------------------------------------------------------
def select_map_compatible(net_vector_arrays, start_node, vect):
    return int(min(
        range(len(net_vector_arrays)),
        key=lambda i: np.linalg.norm(vect - net_vector_arrays[i][start_node])
    ))

def select_map_random(num_maps):
    return int(random.randrange(num_maps))

# ----------------------------------------------------------------------------
# ICN update (shared cache + SOM update on selected map only)
# ----------------------------------------------------------------------------
def apply_icn_update_to_selected_map(cache_storage, net_vector_arrays, selected_map_idx,
                                     cid, vect, start_node, path_found_to_start):
    if cid not in cache_storage[start_node[0]][start_node[1]]:
        cache_storage[start_node[0]][start_node[1]].append(cid)

    target = net_vector_arrays[selected_map_idx]
    total = len(path_found_to_start)
    if total <= 0:
        return
    for i, node in enumerate(path_found_to_start, start=1):
        alpha = LEARNING_RATE * (i / total)
        target[node] += alpha * (vect - target[node])

# ----------------------------------------------------------------------------
# One experiment for a fixed size and fixed num_maps, compare UPDATE policies
# ----------------------------------------------------------------------------
def run_one_setting(size, num_maps=NUM_MAPS_FIXED, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    acc = {
        ("COMPAT", "ID"): [],
        ("RANDOM", "ID"): [],
        ("COMPAT", "ATTR"): [],
        ("RANDOM", "ATTR"): [],
    }

    for _sim in range(TIME_TO_SIMULATE):
        cache0, maps0 = cache_prop(size, num_maps)
        tasks = generate_tasks(size, NUM_CONTENTS_TO_SEARCH)

        envs = {}
        for policy in ["COMPAT", "RANDOM"]:
            for method in ["ID", "ATTR"]:
                envs[(policy, method)] = {
                    "cache": copy.deepcopy(cache0),
                    "maps": copy.deepcopy(maps0),
                    "single_pher": [{} for _ in range(num_maps)],
                    "attrib_pher": [initialize_pheromone_trails(size, N) for _ in range(num_maps)],
                    "hops": [],
                }

        for cid, start in tasks:
            vect = cont_vector_array[cid - 1]

            for policy in ["COMPAT", "RANDOM"]:
                for method in ["ID", "ATTR"]:
                    e = envs[(policy, method)]

                    # (Common) Search map is ALWAYS the compatible best map.
                    search_map_idx = select_map_compatible(e["maps"], start, vect)

                    # Search ONLY on that selected map
                    if method == "ID":
                        best_path, best_h = run_single_task_single_pheromone(
                            e["cache"], e["maps"][search_map_idx], size, cid, start, e["single_pher"][search_map_idx]
                        )
                    else:
                        best_path, best_h = run_single_task_attrib_pheromone(
                            e["cache"], e["maps"][search_map_idx], size, cid, start, e["attrib_pher"][search_map_idx]
                        )

                    # Update policy differs ONLY here
                    if best_path is not None:
                        if policy == "COMPAT":
                            update_map_idx = search_map_idx  # best map
                        else:
                            update_map_idx = select_map_random(num_maps)

                        apply_icn_update_to_selected_map(
                            e["cache"], e["maps"], update_map_idx, cid, vect, start, best_path
                        )

                    hop_val = best_h if np.isfinite(best_h) else TIMES_TO_SEARCH_HOP
                    e["hops"].append(hop_val)

        for key in acc.keys():
            hops = envs[key]["hops"]
            acc[key].append(float(np.mean(hops)) if hops else float("nan"))

    out = {"size": size}
    out["compat_id"] = float(np.mean(acc[("COMPAT", "ID")]))
    out["random_id"] = float(np.mean(acc[("RANDOM", "ID")]))
    out["compat_attr"] = float(np.mean(acc[("COMPAT", "ATTR")]))
    out["random_attr"] = float(np.mean(acc[("RANDOM", "ATTR")]))
    return out

# ----------------------------------------------------------------------------
# Sweep network size (X axis)
# ----------------------------------------------------------------------------
def main():
    size_list = [10, 20, 30, 40, 50]
    results = []

    for size in size_list:
        r = run_one_setting(size=size, num_maps=NUM_MAPS_FIXED, seed=None)
        results.append(r)
        print(f"size={size} | ID: compat-update={r['compat_id']:.2f}, random-update={r['random_id']:.2f} "
              f"| ATTR: compat-update={r['compat_attr']:.2f}, random-update={r['random_attr']:.2f}")

    df_sum = pd.DataFrame(results)
    out_csv = f"summary_update_policy_maps{NUM_MAPS_FIXED}_searchBest_onlyUpdateDiff.csv"
    df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f">>> saved: {out_csv}")

    x = df_sum["size"].tolist()

    plt.figure()
    plt.plot(x, df_sum["compat_id"].tolist(), marker="o", label="ID毎: 相性更新")
    plt.plot(x, df_sum["random_id"].tolist(), marker="s", label="ID毎: ランダム更新")
    plt.xlabel("Network Size")
    plt.ylabel("Average Number of Hops")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(x, df_sum["compat_attr"].tolist(), marker="o", label="属性毎: 相性更新")
    plt.plot(x, df_sum["random_attr"].tolist(), marker="s", label="属性毎: ランダム更新")
    plt.xlabel("Network Size")
    plt.ylabel("Average Number of Hops")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
