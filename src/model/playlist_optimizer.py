# Travelling salesman problem

import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming as tsp_dynamic
from python_tsp.heuristics import solve_tsp_simulated_annealing as tsp_annealing
from python_tsp.heuristics import solve_tsp_local_search as tsp_annealing


def tsp_python(matrix, heuristic: bool = True):
    """ 
    Calculate lowest costing route

    Input: Cost matrix, heuristic: If not, use Held-Karp method (slower) otherwise heuristic
    Output: lowest cost route ==> order of indexes corresponding to songs in playlist
    """
    if not heuristic:
        permutation, cost = tsp_dynamic(matrix)
        return permutation, cost
    
    permutation, cost = tsp_annealing(matrix)
    return permutation, cost

# Other methods...?