import matplotlib.pyplot as plt

# ACO Parameters
num_iterations = 50  # Total number of iterations

alpha_initial = 0.1  # Initial importance of pheromone
alpha_final = 1.0    # Final importance of pheromone
beta_initial = 5.0   # Initial importance of heuristic information
beta_final = 1.0     # Final importance of heuristic information

alpha_values = []
beta_values = []
iterations = list(range(num_iterations))

for search_iter in iterations:
    alpha = alpha_initial + (alpha_final - alpha_initial) * (search_iter / num_iterations)
    beta = beta_initial - (beta_initial - beta_final) * (search_iter / num_iterations)
    alpha_values.append(alpha)
    beta_values.append(beta)

plt.figure(figsize=(10,6))
plt.plot(iterations, alpha_values, label='Importance of Pheromone (α)', marker='o')
plt.plot(iterations, beta_values, label='Importance of Heuristic Information (β)', marker='s')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Change in Importance of Pheromone and Heuristic Information over Iterations')
plt.legend()
plt.grid(True)
plt.show()
