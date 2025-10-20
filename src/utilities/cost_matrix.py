
from src.utilities.transition_cost import transition_cost
import numpy as np

class CostMatrix():
    def __init__(self, df, w_h = 0.5, w_bpm = 0.5):
        self.n = len(df)
        self.df = df
        self.matrix = np.zeros((self.n, self.n))
        self.weight_harm = w_h
        self.weight_bpm = w_bpm
        
    def compute_matrix(self):
        for i in range(self.n):
            for j in range(self.n):
                if i == j: 
                    self.matrix[i,j] = 0 # Avoid unnecessary calculations along diagonal
                else:
                    cost_ij = transition_cost(
                        self.df.iloc[i], 
                        self.df.iloc[j], 
                        self.weight_harm, 
                        self.weight_bpm
                        )
                    self.matrix[i,j] = cost_ij
                
        
    def get_cost(self, i, j):
        return self.matrix[i,j]
    
    def print(self):
        print(self.matrix)
