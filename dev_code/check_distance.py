######      Proporsal Yamashiro      ######
TIME_TO_SIMULATE = 3
NETWORK_SIZE = 50
TIMES_TO_SEARCH = 50
TIMES_TO_SEARCH_HOP = 50
TIMES_TO_CACHE_HOP = 10
TIME_TO_CACHE_PER_CONTENT = 10
LEARNING_RATE = 0.5

CACHE_CAPACITY = 20
VECTOR_INCREMENT = 0.1

###   import module   ###
import numpy as np
import pandas as pd
import random
import importlib
import matplotlib.pyplot as plt
import japanize_matplotlib

###   import csv   ###
file_path = "/Users/asaken-n51/Documents/master_code/test/500_movies.csv"
df = pd.read_csv(file_path)

###   define variables   ###
# NetworkSize (size x size)
size = NETWORK_SIZE
# Number of vector elements (except id number)
N = len(df.columns) -1
# Number of contents
cont_num = len(df)
# Define contents vector data
cont_vector = df.set_index('id').values.tolist()
cont_vector_array = []
for i in range(cont_num):
		cont_vector_array.append(np.array(cont_vector[i]))

# Define caching storage
def get_init_cache_storage():
    cache_storage = np.empty((size,size), dtype=object)
    for i in range(size):
        for j in range(size):
            cache_storage[i][j] = []
    return cache_storage

# Define initial vector data
def get_init_network_vector():
    incr = VECTOR_INCREMENT
    net_vector = np.array([np.arange(0, 1+incr, incr) for _ in range(N)])
    net_vector_array = np.random.choice(net_vector.flatten(), (size, size, N))
    return net_vector_array

# Define neighbors
def get_neighbors(x, y):
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0,  -1),           (0, 1),
                  (1,  -1),  (1, 0), (1,  1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors

###   caching   ###
def cache_prop():
    cache_storage = get_init_cache_storage()
    net_vector_array = get_init_network_vector()
    cache_num = TIME_TO_CACHE_PER_CONTENT
    cache_hops = TIMES_TO_CACHE_HOP
    alpha_zero = LEARNING_RATE
    #print(f"{net_vector_array}")

    for _ in range(cache_num):
        for id in range(1, cont_num+1):
            curr = (np.random.randint(size), np.random.randint(size))
            # vect is Vector of selected content
            vect = cont_vector_array[id-1]
            hoped_node = []

            for _ in range(cache_hops):
                min_dist = float('inf')
                closest = None
                neighbors = get_neighbors(curr[0], curr[1])
                for neighbor in neighbors:
                    if neighbor not in hoped_node:
                        neighbor_vect = net_vector_array[neighbor]
                        dist = np.linalg.norm(vect - neighbor_vect)
                        if dist < min_dist:
                            min_dist = dist
                            closest = neighbor
                if closest is None:
                    break
                # hop nearest node
                hoped_node.append(curr)
                curr = closest

            # cache latest node       
            hoped_node.append(curr)
            # storage cache
            cache_storage[curr[0]][curr[1]].append(id)
            ##  Show dead end rooting  ##
            #if len(hoped_node) < cache_hops:
                #print(f"dead end rooting: {hoped_node}")
            
            tmp = 0
            total_hops = len(hoped_node)
            # update network vector
            for node in hoped_node:
                tmp += 1
                alpha = alpha_zero * (tmp/total_hops)
                net_vector_array[node] += alpha * (vect - net_vector_array[node])
 
    # Check for changes in network vector
    #print(f"{np.round(net_vector_array,2)}")
    return cache_storage, net_vector_array


###   search content   ###
def search_prop(cache_storage, net_vector_array):
    cache_hit = 0
    search_num = TIMES_TO_SEARCH
    search_hops = TIMES_TO_SEARCH_HOP
    total_hops_list = []

    for _ in range(1, search_num+1):
        id = np.random.randint(1, cont_num)
        searched_node = []
        curr = (np.random.randint(size), np.random.randint(size))
        searched = False
        hops_used = 0

        for _ in range(search_hops):
            hops_used += 1
            neighbors = get_neighbors(curr[0], curr[1])
            if id in cache_storage[curr[0]][curr[1]]:
                #print(f"found {id} in node[{curr[0]}][{curr[1]}]")
                searched = True
                break

            for neighbor in neighbors:
                if id in cache_storage[neighbor[0]][neighbor[1]]:
                    #print(f"found {id} in node[{curr[0]}][{curr[1]}]")
                    searched = True
                    break
            if searched:
                break

            min_dist = float('inf')
            closest = None
            vect = cont_vector_array[id-1]

            for neighbor in neighbors:
                if neighbor not in searched_node:
                    neighbor_vect = net_vector_array[neighbor]
                    dist = np.linalg.norm(vect - neighbor_vect)
                    if dist < min_dist:
                        min_dist = dist
                        closest = neighbor
            
            searched_node.append(curr)
            if all(element in searched_node for element in neighbors):
                #print(f"Can not hop any more")
                break
            curr = closest
        
        if searched:
            cache_hit += 1
            total_hops_list.append(hops_used)
        else:
            total_hops_list.append(search_hops)
        
    if len(total_hops_list) > 0:
        average_hops = sum(total_hops_list) / len(total_hops_list)
    else:
        average_hops = 0

    return cache_hit, average_hops

import random, sys
import matplotlib.pyplot as plt

# チェビシェフ距離で理論ホップ数を求める関数
def theoretical_hops(cache_storage, N, num_tasks):
    """num_tasks 回ランダムに (cid, start_node) を取り平均距離を返す。"""
    dists = []
    for _ in range(num_tasks):
        cid   = random.randint(1, cont_num - 1)
        start = (random.randint(0, N-1), random.randint(0, N-1))

        # cid をキャッシュしているノード
        locs = [(x, y) for x in range(N) for y in range(N)
                if cid in cache_storage[x][y]]
        if not locs:                # どこにも無ければスキップ
            continue
        d = min(max(abs(x-start[0]), abs(y-start[1])) for x, y in locs)
        dists.append(d)

    return sum(dists)/len(dists) if dists else 0.0


# def run_experiment_over_sizes():
#     sizes      = [10, 20, 30, 40, 50]
#     sim_hops   = []
#     theo_hops  = []

#     for N in sizes:
#         # 既存コードはグローバル変数 size を参照しているので上書き
#         global size
#         size = N

#         # ① キャッシュ配置（1 回）※重い処理なので再利用
#         cache_storage, net_vec = cache_prop()

#         # ② 実測ホップ数
#         _, avg_hops = search_prop(cache_storage, net_vec)
#         sim_hops.append(avg_hops)

#         # ③ 理論ホップ数
#         avg_theo = theoretical_hops(cache_storage, N, TIMES_TO_SEARCH)
#         theo_hops.append(avg_theo)

#         print(f"N={N:2d} | simulation={avg_hops:.2f} | theory={avg_theo:.2f}")

#     return sizes, sim_hops, theo_hops
def run_experiment_over_sizes():
    sizes         = [10, 20, 30, 40, 50]
    mean_sim_hops = []
    mean_theo     = []

    for N in sizes:
        global size
        size = N

        sim_runs, theo_runs = [], []

        for _ in range(TIME_TO_SIMULATE):
            cache_storage, net_vec = cache_prop()

            # 実測ホップ
            _, avg_hops = search_prop(cache_storage, net_vec)
            sim_runs.append(avg_hops)

            # 理論ホップ
            avg_theo = theoretical_hops(cache_storage, N, TIMES_TO_SEARCH)
            theo_runs.append(avg_theo)

        mean_sim_hops.append(np.mean(sim_runs))
        mean_theo.append(np.mean(theo_runs))

        print(f"N={N:2d} | sim μ={np.mean(sim_runs):.2f} | theo μ={np.mean(theo_runs):.2f}")

    return sizes, mean_sim_hops, mean_theo


# def plot_results(sizes, sim_hops, theo_hops):
#     fs=16
#     plt.plot(sizes, sim_hops,  marker='o', label='既存手法')
#     plt.plot(sizes, theo_hops, marker='s', linestyle='--',
#              label='理論的最短経路')
#     plt.xlabel('Network size  (N×N)', fontsize=fs)
#     plt.ylabel('Average hops', fontsize=fs)           
#     plt.ylim(bottom=0) 
#     plt.legend(fontsize=fs)
#     plt.show()
def plot_results(sizes, mean_sim, mean_theo, fs=18):
    plt.plot(sizes, mean_sim,  marker='o', label='既存手法',  linewidth=2)
    plt.plot(sizes, mean_theo, marker='s', linestyle='--',
             label='理論的最短経路', linewidth=2)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.xlabel('Network size (N×N)', fontsize=fs)
    plt.ylabel('Average hops',       fontsize=fs)
    plt.legend(fontsize=fs)
    plt.show()

###   main   ###
def main():
    cache_hit_index = []
    cache_hit_rate_index = []
    average_hops_index = []
    ###   simulate   ###
    for i in range(TIME_TO_SIMULATE):
        cache_storage, net_vector_array = cache_prop()
        #print(cache_storage)
        cache_hit, avg_hops = search_prop(cache_storage, net_vector_array)
        cache_hit_index.append(cache_hit)
        cache_hit_rate_index.append(100*(cache_hit/TIMES_TO_SEARCH))
        average_hops_index.append(avg_hops)
        print(f"-----     {i+1} times simulated     -----")
        print(f"cache hit: {cache_hit}")
        print(f"cache hit rate: {100*(cache_hit/TIMES_TO_SEARCH)} %")
        print(f"average hops: {avg_hops}")
    
    print(f"average {TIME_TO_SIMULATE} times cache hit count: {sum(cache_hit_index)/len(cache_hit_index)}")
    print(f"average {TIME_TO_SIMULATE} times cache hit rate: {sum(cache_hit_rate_index)/len(cache_hit_rate_index)}")
    print(f"min hit rate: {min(cache_hit_rate_index)}")
    print(f"max hit rate: {max(cache_hit_rate_index)}")

    if average_hops_index:
        final_avg_hops = sum(average_hops_index) / len(average_hops_index)
        print(f"average {TIME_TO_SIMULATE} times hops: {final_avg_hops}")

if __name__ == "__main__":
    # main()
    sizes, sim_hops, theo_hops = run_experiment_over_sizes()
    plot_results(sizes, sim_hops, theo_hops)