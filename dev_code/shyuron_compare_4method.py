## ----------------------------------------------------------------------------
# Integrated Simulation Code: Comparison of Existing vs Proposed Methods
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict

# ----------------------------------------------------------------------------
# 1. Parameter Settings
# ----------------------------------------------------------------------------
# Simulation Control
TIME_TO_SIMULATE = 5          # Number of simulations per network size to average
NETWORK_SIZES = [10, 20, 30, 40, 50] # X-axis: Network sizes to test
#NETWORK_SIZES = [10, 20]    # Debug mode (faster)

# Content & Network
NUM_CONTENTS_TO_SEARCH = 100   # Number of search tasks per simulation
TIMES_TO_CACHE_HOP = 10
TIMES_TO_SEARCH_HOP = 50      # TTL (Time To Live) for search
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5
VECTOR_INCREMENT = 0.1

# ACO Parameters (Proposed Methods)
NUM_ANTS = 10                  # Ants per iteration
NUM_ITERATIONS = 100           # Total iterations (Use last iter for result)

ALPHA_START = 1.0
BETA_START = 10.0
ALPHA_END = 1.0
BETA_END = 1.0
Q = 100
RHO = 0.10
RHO_MAX = 0.9
LAMBDA_R = 1
BOOST = 5

USE_EPSILON = True
EPSILON = 0.01
EPSILON_SOM = 0.20

# CSV Load
# CSVファイルから属性ベクトルを準備
#file_path = "500_movies.csv"  # 適宜修正
file_path = "1500_wines.csv"  # 適宜修正
df = pd.read_csv(file_path)
N = len(df.columns) - 1
cont_num = len(df)
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = [np.array(vec) for vec in cont_vector]

# ----------------------------------------------------------------------------
# 2. Common Helper Functions (Network, Cache, Neighbors)
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
    # Create simple grid vectors or random choice
    net_vector = np.array([np.arange(0, 1 + incr, incr) for _ in range(N)])
    # Flatten and choice to match shape (size, size, N)
    # Using random choice from generated ranges as in Code 1
    flat_options = net_vector.flatten()
    return np.random.choice(flat_options, (size, size, N))

def cache_prop(size):
    """Builds the network environment (Cache and SOM Vectors)"""
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
            
            # Update vectors (SOM learning)
            tmp = 0
            total_hops = len(hoped_node)
            for node in hoped_node:
                tmp += 1
                alpha = alpha_zero * (tmp / total_hops)
                net_vector_array[node] += alpha * (vect - net_vector_array[node])
                
    return cache_storage, net_vector_array

# ----------------------------------------------------------------------------
# 3. Task Generation (Common for all methods)
# ----------------------------------------------------------------------------
def generate_common_tasks_r(size, num_tasks=NUM_CONTENTS_TO_SEARCH):
    """Generates a fixed list of (content_id, start_node) tuples"""
    tasks = []
    for _ in range(num_tasks):
        cid = np.random.randint(1, cont_num + 1)
        start_node = (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start_node))
    return tasks


def generate_common_tasks_z(size, num_tasks=NUM_CONTENTS_TO_SEARCH, popularity_ratio=0.05, request_concentration=0.8):
    """(Zipf-like) Generates tasks with popularity bias (Mix mode only)"""
    tasks = []
    
    # 人気コンテンツのプールを決定
    num_popular = int(cont_num * popularity_ratio)
    all_cids = np.arange(1, cont_num + 1)
    
    # 人気/不人気IDのセットを作成
    if num_popular > 0:
        popular_cids = np.random.choice(all_cids, num_popular, replace=False)
        unpopular_cids = np.setdiff1d(all_cids, popular_cids)
    else:
        popular_cids = np.array([], dtype=int)
        unpopular_cids = all_cids

    for _ in range(num_tasks):
        # request_concentration の確率で人気コンテンツから選ぶ
        if len(popular_cids) > 0 and random.random() < request_concentration:
            cid = int(random.choice(popular_cids))
        else:
            # 不人気（または人気プールが空の場合）から選択
            if len(unpopular_cids) > 0:
                cid = int(random.choice(unpopular_cids))
            else:
                cid = int(random.choice(all_cids)) # フォールバック
            
        start_node = (np.random.randint(size), np.random.randint(size))
        tasks.append((cid, start_node))
    return tasks

# ----------------------------------------------------------------------------
# 4. Methods to Compare
# ----------------------------------------------------------------------------

# --- A. Existing Method (Deterministic) ---
def search_prop_existing(cache_storage, net_vector_array, size, content_tasks):
    total_hops_used = 0
    success_count = 0
    
    for (cid, start_node) in content_tasks:
        searched_node = []
        curr = start_node
        found = False
        hops_used = 0
        vect = cont_vector_array[cid - 1]

        for _ in range(TIMES_TO_SEARCH_HOP):
            hops_used += 1
            
            # Check current node
            if cid in cache_storage[curr[0]][curr[1]]:
                found = True
                break
            
            # Check neighbors (1-hop lookahead for cache)
            neighbors = get_neighbors(curr[0], curr[1], size)
            for nx, ny in neighbors:
                if cid in cache_storage[nx][ny]:
                    found = True
                    break
            if found:
                break

            # Move to closest neighbor (SOM-based Gradient)
            min_dist = float('inf')
            closest = None
            
            for neighbor in neighbors:
                if neighbor not in searched_node:
                    dist = np.linalg.norm(vect - net_vector_array[neighbor])
                    if dist < min_dist:
                        min_dist = dist
                        closest = neighbor
            
            searched_node.append(curr)
            
            # Dead end check
            if closest is None:
                break
                
            curr = closest
        
        if found:
            success_count += 1
            total_hops_used += hops_used
        else:
            total_hops_used += TIMES_TO_SEARCH_HOP # Penalty for failure
            
    avg_hops = total_hops_used / len(content_tasks) if content_tasks else 0.0
    return avg_hops

# --- B. Theoretical Shortest Path (Chebyshev Distance) ---
def theoretical_hops(cache_storage, size, content_tasks):
    dists = []
    for (cid, start_node) in content_tasks:
        # Find all locations of this content
        locs = [(x, y) for x in range(size) for y in range(size)
                if cid in cache_storage[x][y]]
        
        if not locs:
            # If not in cache anywhere, technically infinite, 
            # but for comparison we treat it as max hops or skip.
            # Here we use TIMES_TO_SEARCH_HOP as penalty or 0 if strictly 'distance'
            dists.append(TIMES_TO_SEARCH_HOP) 
            continue
            
        # Chebyshev distance: max(|x1-x2|, |y1-y2|)
        d = min(max(abs(x - start_node[0]), abs(y - start_node[1])) for x, y in locs)
        dists.append(d)

    return sum(dists)/len(dists) if dists else 0.0

# --- Helper for ACO: Initialize Pheromones ---
def initialize_pheromone_trails(size, num_attributes=None):
    pheromone_trails = {}
    is_vector = (num_attributes is not None)
    
    for x in range(size):
        for y in range(size):
            curr_node = (x, y)
            for neighbor in get_neighbors(x, y, size):
                if is_vector:
                    pheromone_trails[(curr_node, neighbor)] = np.ones(num_attributes)
                else:
                    pheromone_trails[(curr_node, neighbor)] = 1.0
    return pheromone_trails

# --- C. Proposal 1: Content-based Pheromone ---
def run_proposal_1(cache_storage, net_vector_array, size, content_tasks):
    # This wraps 'multi_contents_single_pheromone_with_reset' logic
    # Returns: Average hops at the FINAL iteration
    
    pheromone_dict = {} # Content ID -> Pheromone Map
    total_final_hops = 0
    
    for (cid, start_node) in content_tasks:
        # Initialize or Get Pheromone for this Content
        if cid not in pheromone_dict:
            pheromone_dict[cid] = initialize_pheromone_trails(size, num_attributes=None) # Scalar pheromone
        
        pheromone_trails = pheromone_dict[cid]
        vect = cont_vector_array[cid - 1]
        
        # Determine if content exists (for optimization, though ACO handles it)
        # In ACO logic provided, if not found, cost is max.
        
        # -- Iteration Loop --
        final_iter_costs = []
        
        # Variables for learning
        best_cost_global = TIMES_TO_SEARCH_HOP
        
        for t in range(NUM_ITERATIONS):
            iter_costs = []
            all_paths = []
            all_costs = []
            
            # Ant Loop
            for _ in range(NUM_ANTS):
                path = []
                visited = set()
                current_node = start_node
                path.append(current_node)
                visited.add(current_node)
                cost = 0
                found = False
                
                # ... [Logic for Movement: Iter 0 (Greedy) vs Iter >0 (Probabilistic)] ...
                # Simplyfing integration by using the core logic from Code 2 directly here
                # to ensure 't' is handled correctly.
                
                if t == 0:
                    # Iter 0: Greedy / Epsilon-Greedy
                    curr = start_node
                    path = [curr]
                    visited.add(curr)
                    cost = 0
                    
                    for _step in range(TIMES_TO_SEARCH_HOP):
                        cost += 1
                        # Check cache (current + neighbors)
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True; break
                        hit_neighbor = False
                        for nx, ny in get_neighbors(curr[0], curr[1], size):
                            if cid in cache_storage[nx][ny]:
                                found = True; hit_neighbor = True; break
                        if found and hit_neighbor: break
                        
                        # Move
                        allowed = [n for n in get_neighbors(curr[0], curr[1], size) if n not in visited] # Simplified 'searched_node'
                        if not allowed: break
                        
                        # Epsilon Greedy
                        if USE_EPSILON and random.random() < EPSILON_SOM:
                             next_node = random.choice(allowed)
                        else:
                            # Closest
                            min_dist_val = float('inf')
                            next_node = None
                            for nb in allowed:
                                dist = np.linalg.norm(vect - net_vector_array[nb])
                                if dist < min_dist_val:
                                    min_dist_val = dist
                                    next_node = nb
                            if next_node is None: next_node = allowed[0] # Fallback

                        curr = next_node
                        path.append(curr)
                        visited.add(curr)
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True; break
                
                else:
                    # Iter > 0: ACO Probabilistic
                    curr = start_node
                    
                    # Update Alpha/Beta
                    ALPHA = ALPHA_START + (ALPHA_END - ALPHA_START) * (t / (NUM_ITERATIONS - 1))
                    BETA  = BETA_START  + (BETA_END  - BETA_START)  * (t / (NUM_ITERATIONS - 1))
                    
                    for _step in range(TIMES_TO_SEARCH_HOP):
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True; break
                        
                        neighbors = get_neighbors(curr[0], curr[1], size)
                        allowed = [n for n in neighbors if n not in visited]
                        if not allowed: break
                        
                        # Epsilon
                        if USE_EPSILON and random.random() < EPSILON:
                            next_node = random.choice(allowed)
                        else:
                            probs = []
                            denom = 0
                            for n in allowed:
                                edge = (curr, n)
                                tau = pheromone_trails[edge]
                                dist = np.linalg.norm(vect - net_vector_array[n])
                                eta = 1.0 / (dist + 1e-6)
                                
                                # Scalar pheromone formula
                                score = (tau ** ALPHA) * (eta ** BETA)
                                probs.append(score)
                                denom += score
                            
                            if denom == 0:
                                next_node = random.choice(allowed)
                            else:
                                probs = [p/denom for p in probs]
                                next_node = random.choices(allowed, weights=probs)[0]
                        
                        path.append(next_node)
                        visited.add(next_node)
                        cost += 1
                        curr = next_node
                        if cid in cache_storage[curr[0]][curr[1]]:
                            found = True; break

                # Result recording
                if found:
                    all_paths.append(path)
                    all_costs.append(cost)
                    iter_costs.append(cost)
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)
            
            # --- End of Ant Loop for Iteration t ---
            
            # Update best cost
            if all_costs:
                min_c = min(all_costs)
                if min_c < best_cost_global:
                    best_cost_global = min_c
            
            # Pheromone Update (Evaporation)
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = max(pheromone_trails[edge], 1e-6)
                
            # Adaptive Evaporation based on local best
            current_iter_best = min(all_costs) if all_costs else TIMES_TO_SEARCH_HOP
            if current_iter_best > 0:
                for path, c in zip(all_paths, all_costs):
                    delta_p = max(0.0, (c / current_iter_best) - 1.0) # Using iter best for local adaptation
                    if delta_p == 0: continue
                    rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)
                    for a, b in zip(path[:-1], path[1:]):
                         pheromone_trails[(a, b)] = max(pheromone_trails[(a, b)] * extra_mul, 1e-6)

            # Pheromone Deposit
            for path, c in zip(all_paths, all_costs):
                if c > 0:
                    Q_eff = Q * BOOST if c <= best_cost_global else Q
                    delta = Q_eff / c
                    for i in range(len(path)-1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta
            
            # Store costs if this is the last iteration
            if t == NUM_ITERATIONS - 1:
                final_iter_costs = iter_costs

        # End of Iterations for this task
        if final_iter_costs:
            total_final_hops += np.mean(final_iter_costs)
        else:
            total_final_hops += TIMES_TO_SEARCH_HOP
            
    return total_final_hops / len(content_tasks)

# --- D. Proposal 2: Attribute-based Pheromone ---
def run_proposal_2(cache_storage, net_vector_array, size, content_tasks):
    # This wraps 'multi_contents_attrib_pheromone_common' logic
    # Global Pheromone map for attributes
    global_pheromone_trails = initialize_pheromone_trails(size, num_attributes=N)
    
    total_final_hops = 0
    
    for (cid, start_node) in content_tasks:
        pheromone_trails = global_pheromone_trails # Shared!
        vect = cont_vector_array[cid - 1]
        
        final_iter_costs = []
        best_cost_global = TIMES_TO_SEARCH_HOP
        
        for t in range(NUM_ITERATIONS):
            iter_costs = []
            all_paths = []
            all_costs = []
            
            for _ in range(NUM_ANTS):
                path = []
                visited = set()
                curr = start_node
                path.append(curr)
                visited.add(curr)
                cost = 0
                found = False
                
                # Iter 0: Same as Prop 1 (Greedy/Epsilon)
                if t == 0:
                    # (Reusing logic for brevity: essentially standard greedy walk)
                    # For strict correctness, copying the "Iter 0" block from Prop 1 is needed
                    # ... [Insert Iter 0 Logic Here - same as Prop 1] ...
                    # To keep code clean, assuming same implementation:
                    for _step in range(TIMES_TO_SEARCH_HOP):
                        cost += 1
                        if cid in cache_storage[curr[0]][curr[1]]: found=True; break
                        hit_n = False
                        for nx, ny in get_neighbors(curr[0], curr[1], size):
                            if cid in cache_storage[nx][ny]: found=True; hit_n=True; break
                        if found and hit_n: break
                        
                        allowed = [n for n in get_neighbors(curr[0], curr[1], size) if n not in visited]
                        if not allowed: break
                        
                        if USE_EPSILON and random.random() < EPSILON_SOM:
                             next_node = random.choice(allowed)
                        else:
                            min_d = float('inf'); next_node = None
                            for nb in allowed:
                                d = np.linalg.norm(vect - net_vector_array[nb])
                                if d < min_d: min_d=d; next_node=nb
                            if next_node is None: next_node = allowed[0]
                        
                        curr = next_node
                        path.append(curr); visited.add(curr)
                        if cid in cache_storage[curr[0]][curr[1]]: found=True; break
                
                else:
                    # Iter > 0: Attribute Pheromone Logic
                    ALPHA = ALPHA_START + (ALPHA_END - ALPHA_START) * (t / (NUM_ITERATIONS - 1))
                    BETA  = BETA_START  + (BETA_END  - BETA_START)  * (t / (NUM_ITERATIONS - 1))
                    
                    for _step in range(TIMES_TO_SEARCH_HOP):
                        if cid in cache_storage[curr[0]][curr[1]]: found=True; break
                        
                        neighbors = get_neighbors(curr[0], curr[1], size)
                        allowed = [n for n in neighbors if n not in visited]
                        if not allowed: break
                        
                        if USE_EPSILON and random.random() < EPSILON:
                            next_node = random.choice(allowed)
                        else:
                            probs = []
                            denom = 0
                            for n in allowed:
                                edge = (curr, n)
                                tau_vec = pheromone_trails[edge] # Vector
                                dist = np.linalg.norm(vect - net_vector_array[n])
                                eta = 1.0 / (dist + 1e-6)
                                
                                # Vector Pheromone Calculation
                                if np.sum(vect) > 0:
                                    # Weighted sum of pheromones based on content attributes
                                    A_ij = np.sum((tau_vec ** ALPHA) * vect) / np.sum(vect)
                                else:
                                    A_ij = 0
                                
                                score = A_ij * (eta ** BETA)
                                probs.append(score)
                                denom += score
                            
                            if denom == 0: next_node = random.choice(allowed)
                            else:
                                probs = [p/denom for p in probs]
                                next_node = random.choices(allowed, weights=probs)[0]
                        
                        curr = next_node
                        path.append(curr); visited.add(curr)
                        cost += 1
                        if cid in cache_storage[curr[0]][curr[1]]: found=True; break

                if found:
                    all_paths.append(path)
                    all_costs.append(cost)
                    iter_costs.append(cost)
                else:
                    iter_costs.append(TIMES_TO_SEARCH_HOP)

            # Update Global Best
            if all_costs:
                if min(all_costs) < best_cost_global: best_cost_global = min(all_costs)
            
            # Pheromone Update (Evaporation)
            for edge in pheromone_trails:
                pheromone_trails[edge] *= (1 - RHO)
                pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            # Adaptive extra evaporation
            curr_iter_best = min(all_costs) if all_costs else TIMES_TO_SEARCH_HOP
            if curr_iter_best > 0:
                for path, c in zip(all_paths, all_costs):
                    delta_p = max(0.0, (c / curr_iter_best) - 1.0)
                    if delta_p == 0: continue
                    rho_eff = min(RHO_MAX, RHO * (1 + LAMBDA_R * delta_p))
                    extra_mul = 1.0 - (rho_eff - RHO)
                    for i in range(len(path)-1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] *= extra_mul
                        pheromone_trails[edge] = np.maximum(pheromone_trails[edge], 1e-6)

            # Deposit
            for path, c in zip(all_paths, all_costs):
                if c > 0:
                    Q_eff = Q * BOOST if c <= best_cost_global else Q
                    # Vector deposit
                    if np.sum(vect) > 0:
                        delta = (Q_eff * vect) / (c * np.sum(vect))
                    else:
                        delta = 0 # Should not happen with valid data
                    
                    for i in range(len(path)-1):
                        edge = (path[i], path[i+1])
                        pheromone_trails[edge] += delta
            
            if t == NUM_ITERATIONS - 1:
                final_iter_costs = iter_costs

        if final_iter_costs:
            total_final_hops += np.mean(final_iter_costs)
        else:
            total_final_hops += TIMES_TO_SEARCH_HOP

    return total_final_hops / len(content_tasks)

# ----------------------------------------------------------------------------
# 5. Main Experiment Loop
# ----------------------------------------------------------------------------
def run_experiment():
    results = {
        'Existing': [],
        'Theoretical': [],
        'Prop1 (Content)': [],
        'Prop2 (Attrib)': []
    }
    
    print(f"Starting Experiment across sizes: {NETWORK_SIZES}")
    print(f"Simulations per size: {TIME_TO_SIMULATE}, Tasks per sim: {NUM_CONTENTS_TO_SEARCH}")
    
    for size in NETWORK_SIZES:
        print(f"\n[Network Size: {size}x{size}]")
        
        # Temporary lists for this size
        res_exist = []
        res_theo = []
        res_prop1 = []
        res_prop2 = []
        
        for sim_idx in range(TIME_TO_SIMULATE):
            print(f"  > Sim {sim_idx+1}/{TIME_TO_SIMULATE}...", end=" ", flush=True)
            
            # 1. Build Environment (Common)
            cache_storage, net_vector_array = cache_prop(size)
            
            # 2. Generate Tasks (Common)
            tasks = generate_common_tasks_z(size)
            
            # 3. Run Methods
            # Existing
            h_exist = search_prop_existing(cache_storage, net_vector_array, size, tasks)
            res_exist.append(h_exist)
            
            # Theoretical
            h_theo = theoretical_hops(cache_storage, size, tasks)
            res_theo.append(h_theo)
            
            # Prop 1
            h_p1 = run_proposal_1(cache_storage, net_vector_array, size, tasks)
            res_prop1.append(h_p1)
            
            # Prop 2
            h_p2 = run_proposal_2(cache_storage, net_vector_array, size, tasks)
            res_prop2.append(h_p2)
            
            print(f"Done. (Ext:{h_exist:.2f}, P1:{h_p1:.2f}, P2:{h_p2:.2f})")
        
        # Average across simulations
        results['Existing'].append(np.mean(res_exist))
        results['Theoretical'].append(np.mean(res_theo))
        results['Prop1 (Content)'].append(np.mean(res_prop1))
        results['Prop2 (Attrib)'].append(np.mean(res_prop2))
        
    return results

def plot_results(results):
    sizes = NETWORK_SIZES
    fs = 22
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(sizes, results['Existing'], marker='^', label='既存手法', linewidth=2, color='#1f77b4')
    plt.plot(sizes, results['Prop1 (Content)'], marker='o', label='提案手法1(ID毎)', linewidth=2, color='red')
    plt.plot(sizes, results['Prop2 (Attrib)'], marker='s', label='提案手法2(属性毎)', linewidth=2, color='blue')
    plt.plot(sizes, results['Theoretical'], marker='x', linestyle='--', label='理論的最短経路', linewidth=2, color='orange')
    
    plt.ylim(bottom=0)
    plt.xticks(sizes, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Network size (N×N)', fontsize=fs)
    plt.ylabel('Average hops', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_experiment()
    print("\nFinal Results:", results)
    plot_results(results)