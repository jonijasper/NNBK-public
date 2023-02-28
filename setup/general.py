import numpy as np

RANDOM_SEED = 3
save_dir = "./figs/"
SAT_SCALE = 1-np.exp(-0.25) # =N(r^2 = 1/Qs^2)  None -> no cut
training_fraction = 0.9 # fraction of datapoints used in training, rest for scoretest


# selected files from different areas of paramater space that pass through 
# default sat filter, 1-np.exp(-0.25) with range [0.1,1.0]
default_f2_tests = {
    'mvg': ['./datafiles/nnfit-files/constant-coupling/F2-files/f2_1.dat', 
            './datafiles/nnfit-files/constant-coupling/F2-files/f2_41.dat', 
            './datafiles/nnfit-files/constant-coupling/F2-files/f2_90.dat'], 
    'mve': ['./datafiles/nnfit-files/constant-coupling/F2-files/f2_98.dat', 
            './datafiles/nnfit-files/constant-coupling/F2-files/f2_127.dat', 
            './datafiles/nnfit-files/constant-coupling/F2-files/f2_163.dat'], 
    'mvge': ['./datafiles/nnfit-files/constant-coupling/F2-files/f2_460.dat', 
             './datafiles/nnfit-files/constant-coupling/F2-files/f2_513.dat', 
             './datafiles/nnfit-files/constant-coupling/F2-files/f2_571.dat'], 
    'mv2g': ['./datafiles/nnfit-files/constant-coupling/F2-files/f2_382.dat', 
             './datafiles/nnfit-files/constant-coupling/F2-files/f2_385.dat', 
             './datafiles/nnfit-files/constant-coupling/F2-files/f2_444.dat'], 
    'gbw': ['./datafiles/nnfit-files/constant-coupling/F2-files/f2_188.dat', 
            './datafiles/nnfit-files/constant-coupling/F2-files/f2_220.dat', 
            './datafiles/nnfit-files/constant-coupling/F2-files/f2_252.dat'], 
    'atan': ['./datafiles/nnfit-files/constant-coupling/F2-files/f2_262.dat', 
             './datafiles/nnfit-files/constant-coupling/F2-files/f2_310.dat', 
             './datafiles/nnfit-files/constant-coupling/F2-files/f2_350.dat']}