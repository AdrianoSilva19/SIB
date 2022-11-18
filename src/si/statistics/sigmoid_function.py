import sys
import numpy as np
sys.path.insert(0, 'src/si')

def sigmoid_function(x):
    return 1/(1+np.exp(-x))



