import numpy as np

RANDOM_SEED = 3
save_dir = "./"
SAT_SCALE = 1-np.exp(-0.25) # =N(r^2 = 1/Qs^2)  None -> no cut