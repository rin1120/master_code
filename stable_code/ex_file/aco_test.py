import numpy as np
import random
import matplotlib.pyplot as plt

# パラメータ設定
N = 20  # グリッドサイズ
m = 100  # 蟻の数
iter_max = 10  # イテレーション数
alpha = 1.0  # フェロモンの重要度
beta = 5.0   # ヒューリスティック情報の重要度
rho = 0.5    # フェロモン蒸発率
Q = 100      # フェロモン増加量

# ノード数
node_num = N * N

# エッジのコストとフェロモンの初期化
cost_matrix = np.ones((node_num, node_num))  # エッジのコストを1に設定
pheromone_matrix = np.ones((node_num, node_num))  # フェロモンを1に初期化

# 隣接ノードの取得関数
def get_neighbors(node):
    neighbors = []
    x, y = divmod(node, N)
    moves = [(-1, -1), (-1, 0), (-1, 1),
             ( 0, -1),          ( 0, 1),
             ( 1, -1), ( 1, 0), ( 1, 1)]  # 上下左右および斜めの移動
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < N and 0 <= ny < N:
            neighbors.append(nx * N + ny)
    return neighbors

# ヒューリスティック情報（コストの逆数）
heuristic_matrix = 1 / cost_matrix

# 最適経路とそのコスト
best_path = None
best_cost = float('inf')
best_costs = []  # 各イテレーションの最良経路コストを記録
average_costs = []  # 各イテレーションの平均経路コストを記録

# メインループ
for iteration in range(iter_max):
    all_paths = []
    all_costs = []
    for ant in range(m):
        path = []
        visited = set()
        current_node = 0  # スタートノード
        goal_node = node_num - 1  # ゴールノード
        path.append(current_node)
        visited.add(current_node)
        total_cost = 0

        while current_node != goal_node:
            neighbors = get_neighbors(current_node)
            allowed_nodes = [node for node in neighbors if node not in visited]
            if not allowed_nodes:
                total_cost = float('inf')  # 行き止まりの場合
                break
            probabilities = []
            for next_node in allowed_nodes:
                tau = pheromone_matrix[current_node][next_node] ** alpha
                eta = heuristic_matrix[current_node][next_node] ** beta
                probabilities.append(tau * eta)
            probabilities = probabilities / np.sum(probabilities)
            next_node = np.random.choice(allowed_nodes, p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            total_cost += cost_matrix[current_node][next_node]
            current_node = next_node

        all_paths.append(path)
        all_costs.append(total_cost)

        # 最良経路の更新
        if current_node == goal_node and total_cost < best_cost:
            best_cost = total_cost
            best_path = path

    # フェロモンの蒸発
    pheromone_matrix *= (1 - rho)

    # フェロモンの追加
    for path, cost in zip(all_paths, all_costs):
        if path[-1] == goal_node and cost < float('inf'):  # ゴールに到達した経路のみ
            delta_pheromone = Q / cost
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                pheromone_matrix[from_node][to_node] += delta_pheromone

    # 各イテレーションの最良経路コストを記録
    best_costs.append(best_cost)

    # 各イテレーションの平均経路コストを記録
    valid_costs = [cost for cost in all_costs if cost < float('inf')]
    if valid_costs:
        average_cost = np.mean(valid_costs)
    else:
        average_cost = float('inf')
    average_costs.append(average_cost)

# 結果の表示
print("最適経路のホップ数：", best_cost)
print("最適経路：", best_path)

# 経路コストのプロット
plt.figure()
plt.plot(best_costs, label='Best Path Cost')
plt.plot(average_costs, label='Average Path Cost')
plt.xlabel('Iteration')
plt.ylabel('Path Cost')
plt.title('Path Costs over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# 実際の最短経路を計算
def bfs_shortest_path(start_node, goal_node):
    from collections import deque
    visited = [False] * node_num
    prev = [None] * node_num
    queue = deque()
    queue.append(start_node)
    visited[start_node] = True

    while queue:
        current_node = queue.popleft()
        if current_node == goal_node:
            break
        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                prev[neighbor] = current_node
                queue.append(neighbor)
    
    # 最短経路の再構築
    path = []
    at = goal_node
    while at is not None:
        path.append(at)
        at = prev[at]
    path.reverse()
    return path

shortest_path = bfs_shortest_path(0, node_num - 1)
print("実際の最短経路のホップ数：", len(shortest_path) - 1)
print("実際の最短経路：", shortest_path)
