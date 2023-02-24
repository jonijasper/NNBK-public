# F2, FL, and HERA data filepaths:
F2 = ""
FL = ""
HERA = ""


# F2/FL file detail info:
ec_shape = (10,8) 
gamma_shape = (10,9)
gamma2_shape = (10,10)
gamma_ec_shape = (5,5,5)

filenros = {"mvg": (range(1,91), gamma_shape),
            "mve": (range(91,171), ec_shape),
            "gbw": (range(171,261), gamma_shape),
            "atan": (range(261,351), gamma_shape),
            "mv2g": (range(351,451), gamma2_shape),
            "mvge": (range(451,576), gamma_ec_shape)}