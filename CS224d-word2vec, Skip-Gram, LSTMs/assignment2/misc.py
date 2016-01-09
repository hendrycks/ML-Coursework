##
# Miscellaneous helper functions
##

import numpy as np

def random_weight_matrix(m, n):

    bound = np.sqrt(6.0/(m + n))
    A0 = np.random.uniform(-bound, bound, size=(m, n))

    assert(A0.shape == (m, n))
    return A0