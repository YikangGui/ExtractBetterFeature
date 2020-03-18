import numpy as np
from scipy.sparse import csr_matrix
import os
import json

a = [1,2]
b = [[1,2,2],[10,11,21]]
a = np.array(a)
b = np.array(b)
print(np.dot(a,b))
print(np.dot(a,b).squeeze())