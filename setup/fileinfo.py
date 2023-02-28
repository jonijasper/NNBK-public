# F2, FL, and HERA data filepaths:
F2 = "datafiles/nnfit-files/constant-coupling/F2-files/"
FL = ""
HERA = ""


# F2/FL file detail info:
ec_shape = (10,8) 
gamma_shape = (10,9)
gamma2_shape = (10,10)
gamma_ec_shape = (5,5,5)

filenros = {"mvg": {"range": range(1,91), "shape": gamma_shape},
            "mve": {"range": range(91,171),  "shape": ec_shape},
            "gbw": {"range": range(171,261),  "shape": gamma_shape},
            "atan": {"range": range(261,351),  "shape": gamma_shape},
            "mv2g": {"range": range(351,451),  "shape": gamma2_shape},
            "mvge": {"range": range(451,576),  "shape": gamma_ec_shape}}


