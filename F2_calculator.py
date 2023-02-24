import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate, interpolate
from scipy import special
import numpy as np

mf = 0.14   # quark mass
Nc = 3  # number of colors
K0 = lambda x: special.k0(x)    # bessel
K1 = lambda x: special.k1(x)    # bessel
sigma0 = 2*2.5*16.4 # [mb] (1mb = 2.5 GeV^-2)
aem = 1/137 # alpha_em
Zf = [1/3,1/3,2/3] # quark charges in unit of e

zlimits = [0,1]
default_Qsqr = list(map(int,np.logspace(0,2,5)))[:-1]

# z_opts = {'limit':50,'epsrel':1e-2}
# r_opts = z_opts


def photon_wf_T(z,r,Q2):
    af = np.sqrt(Q2 * z*(1-z) + mf**2) 
    sum_ = 0
    for zf in Zf:
        sum_ += aem*zf**2/np.pi* \
            ( af**2 * K1(r*af)**2 * (z**2+(1-z)**2) + mf**2 * K0(r*af)**2)

    return 2*Nc*sum_

def photon_wf_L(z,r,Q2):
    af = np.sqrt(Q2 * z*(1-z) + mf**2)
    sum_ = 0
    for zf in Zf:
        sum_ += aem*zf**2/np.pi*4*Q2*z**2*(1-z)**2 * K0(r*af)**2

    return 2*Nc*sum_

def cs_dipole_MV(r):
    # dipole amplitude parameters
    Qs0sqr = 0.165
    gamma = 1.135
    L_qcd = 0.241
    ec = 1
    e = np.exp(1)

    N_MV = lambda r: 1 - np.exp( -1/4*(r**2*Qs0sqr)**gamma 
                    * np.log(1/(r*L_qcd)+ec*e) )
    # N_MV = lambda r: 1 - np.exp( -1/4*(r**2*Qs0sqr)**gamma)

    # return 1-np.exp(-r**2)
    int_db = sigma0/2
    return 2*int_db*N_MV(r)

def cs_dipole(r,N):
    if not N:
        return cs_dipole_MV(r)
    int_db = sigma0/2

    return 2*int_db*N(r)

def integrand_T(z,r,Q2,dip):
    return r/2*photon_wf_T(z,r,Q2)*cs_dipole(r,dip)

def integrand_L(z,r,Q2,dip):
    return r/2*photon_wf_L(z,r,Q2)*cs_dipole(r,dip)

def r_integral_T(r0,r1,z_integrated):
    cs = integrate.quad(z_integrated,r0,r1)[0]
    return cs

def r_integral_L(r0,r1,z_integrated):
    cs = integrate.quad(z_integrated,r0,r1)[0]
    return cs

def z_integral_T(z0,z1,r,Q2,dip):
    ft = integrate.quad(integrand_T,z0,z1,args=(r,Q2,dip))[0]
    return ft

def z_integral_L(z0,z1,r,Q2,dip):
    fl = integrate.quad(integrand_L,z0,z1,args=(r,Q2,dip))[0]
    return fl

def calculate_F2s(r,Q2list,dipole_amplitudes,return_values,*,zlims=None,ylist=None):
    '''
    args
        :dipole_amplitudes: list of N(r), use "MV" for fitted amplitude
        :return_values: list|str, possible returns are "Sigma", "F2", "FT", "FL", 
                                    "Sigma_MV", "zintegral_T", "zintegral_L"
        
        :y: list, needed if "Sigma" in return values, has to be same length as Q2list
    
    ---

    returns
        dict of values specified in return_values arg
        
        Sigma, F2, FT, FL, Sigma_MV are a list of shape Q2list, 
        
        zintegral_T, zintegral_L are dicts {"Q2": f(r)}
        
    '''
    zint_T_dict = {}
    zint_L_dict = {}
    Sigma_list = []
    F2_list = []
    FT_list = []
    FL_list = []
    Sigma_MV_list = []
    if zlims:
        z0 = zlims[0]
        z1 = zlims[1]    
    else:
        z0 = zlimits[0]
        z1 = zlimits[1]

    r0 = min(r)
    r1 = max(r)

    if dipole_amplitudes=="MV":
        dipole_interp=None
    else:
        dipole_interp = interpolate.interp1d(r,dipole_amplitudes)
    
    Q2sorted = sorted(Q2list)
    if len(Q2sorted)>3:
        Q2zint = [Q2sorted[0],Q2sorted[int(len(Q2sorted)/2)], Q2sorted[-1]]
    else:
        Q2zint = Q2sorted

    zint_T = np.vectorize(z_integral_T)
    zint_L = np.vectorize(z_integral_L)
    for i,Q2 in enumerate(Q2list):
        # needs to be calculated for all cases:
        fl_zint = zint_L(z0,z1,r,Q2,dipole_interp) # polarized cs z integral
        if Q2 in Q2zint:
            zint_L_dict[f"{Q2}"] = fl_zint

        fl_zint_interp = interpolate.interp1d(r,fl_zint)
        FL = Q2/(4*np.pi**2*aem)*r_integral_L(r0,r1,fl_zint_interp)

        if "FL" in return_values:
            FL_list.append(FL)

        # these need FT and F2
        if any(a in return_values for a in ("Sigma","FT","F2")):
            ft_zint = zint_T(z0,z1,r,Q2,dipole_interp) # polarized cs z integral
            if Q2 in Q2zint:
                zint_T_dict[f"{Q2}"] = ft_zint

            ft_zint_interp = interpolate.interp1d(r,ft_zint)
            FT = Q2/(4*np.pi**2*aem)*r_integral_T(r0,r1,ft_zint_interp)
            if "FT" in return_values:
                FT_list.append(FT)

        if "F2" in return_values:
            F2 = FT + FL
            F2_list.append(F2)

        if "Sigma" in return_values:
            y = ylist[i]
            Sigma = F2 - y**2/(1+(1-y)**2)*FL
            Sigma_list.append(Sigma)

        if "Sigma_MV" in return_values:
            fl_zint_mv = zint_L(z0,z1,r,Q2,None)
            ft_zint_mv = zint_T(z0,z1,r,Q2,None)
            fl_zint_interp_mv = interpolate.interp1d(r,fl_zint_mv)
            ft_zint_interp_mv = interpolate.interp1d(r,ft_zint_mv)
            FL_MV = Q2/(4*np.pi**2*aem)*r_integral_L(r0,r1,fl_zint_interp_mv)
            FT_MV = Q2/(4*np.pi**2*aem)*r_integral_T(r0,r1,ft_zint_interp_mv)

            y = ylist[i]
            F2_MV = FT_MV + FL_MV
            Sigma_MV = F2_MV - y**2/(1+(1-y)**2)*FL_MV
            Sigma_MV_list.appen(Sigma_MV)
    
    return_dict = dict()
    for _name, _list in zip(["zintegral_T","zintegral_L","Sigma","F2","FT","FL","Sigma_MV"], 
                            [zint_T_dict,zint_L_dict,Sigma_list,F2_list,FT_list,FL_list,Sigma_MV_list]):
        if _list:
            return_dict[_name] = _list

    return return_dict




if __name__=="__main__":
    def plott(z,r,Q2list):
        plotlist = ["zintegral_T", "zintegral_L", "F2", "FT", "FL"]
        plotvals = calc_F2s(r,Q2list,dipole_amplitudes="MV",zlims=z,
                            return_values=plotlist)

        fig, axs = plt.subplots(1,len(plotlist),figsize=(3*len(plotlist),4))

        for _l,_y in plotvals["zintegral_T"].items():
            axs[0].plot(r,_y, label=f"Q2={_l}")
            axs[0].vlines(r[_y.argmax()],0,max(_y),colors='k',
                    linestyles="dashed",alpha=0.5, linewidths=1.0)
        
        for _l,_y in plotvals["zintegral_L"].items():
            axs[1].plot(r,_y, label=f"Q2={_l}")
            axs[1].vlines(r[_y.argmax()],0,max(_y),colors='k',
                    linestyles="dashed",alpha=0.5, linewidths=1.0)


        axs[2].scatter(Q2list,plotvals["F2"])
        axs[3].scatter(Q2list,plotvals["FT"])
        axs[4].scatter(Q2list,plotvals["FL"])
        
            
        for ax in axs:
            # x_ticks = ax.xaxis.get_major_ticks()
            # x_ticks[1].label1.set_visible(False)
            ax.set_xscale("log")
            # ax.set_yscale("log")

        axs[0].legend(loc="upper left", frameon=False)
        axs[1].legend(loc="upper left", frameon=False)
        
        axs[0].set_xlabel("r [GeV$^{-1}$]")
        axs[1].set_xlabel("r [GeV$^{-1}$]")

        axs[2].set_xlabel("$Q^2$ [GeV$^{-2}$]")
        axs[3].set_xlabel("$Q^2$ [GeV$^{-2}$]")
        axs[4].set_xlabel("$Q^2$ [GeV$^{-2}$]")

        axs[0].set_ylabel(r"$y(r) = \frac{r}{2} \int \mathrm{d}z\  |\psi_T(z,r,Q^2)|^2 \sigma^{q\bar{q}p}(r)$")
        axs[1].set_ylabel(r"$y(r) = \frac{r}{2} \int \mathrm{d}z\  |\psi_L(z,r,Q^2)|^2 \sigma^{q\bar{q}p}(r)$")

        axs[2].set_ylabel(r"F2")
        axs[3].set_ylabel(r"FT")
        axs[4].set_ylabel(r"FL")
        

        plt.tight_layout()
        plt.show()

    r = np.logspace(-3,1,100)
    z = [0.01,0.99]
    # z = [0,1]
    Qsqr = list(map(int,np.logspace(0,2,10)))

    plott(z,r,Qsqr)