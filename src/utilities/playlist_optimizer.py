import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming as tsp_dynamic
from python_tsp.heuristics import solve_tsp_simulated_annealing as tsp_annealing
from python_tsp.heuristics import solve_tsp_local_search as tsp_local
import pandas as pd
from cost_matrix import CostMatrix

def create_path_from_csv(filepath, type='a'):

    solvers = {'d': tsp_dynamic, 'l': tsp_local, 'a': tsp_annealing}
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns] # Remove space, make lowercase

    cost_matrix = CostMatrix(df)
    cost_matrix.compute_matrix() # Compute transition costs for all songs in the dataframe
    
    solver = solvers.get(type, tsp_annealing) # Get desired solver, default to annealing if input invalid

    permutation, total_cost = solver(cost_matrix.matrix)
    return permutation, total_cost 



# Other methods...?