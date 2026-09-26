import numpy as np
from src.optimization.qpso import QPSOOptimizer

def simple_obj(x):
    return np.sum(x**2)

optimizer = QPSOOptimizer()
lb = np.array([-5.0, -5.0])
ub = np.array([5.0, 5.0])
result = optimizer.optimize(objective_function=simple_obj, lb=lb, ub=ub, population_size=10, max_iterations=20, seed=0, hyperparameters={})
print('Best score', result.best_score)
