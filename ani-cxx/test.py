

import numpy as np


d = np.load('./data/experiment_1/splits/cc_5.npz')
print(d['ood_test_idx'])