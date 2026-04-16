import numpy as np
import deepxde as dd
import scipy as sc
import matplotlib.pyplot as plt
np.random.seed(1234)

#%%
sensors = np.random.rand(100)

#%%

def ode_constructor(cheb_coeff, max_t):
    def microgrid_ode(t,y):
        delta_ss = 0.0004
        eta_chrg = 0.9
        eta_dis = 0.9
        tau_ss = 0.1 / 3600
        # tau_d = 0.3 / 3600
        e_ss = y[0]
        p_ss = y[1]
        # p_d = y[2]
        v = 0.0 if p_ss < 0 else 1.0
        u_ss = 5*np.polynomial.chebyshev.chebval(t/max_t, cheb_coeff)
        # print(t, u_ss)
        # u_d = 0


        return [-delta_ss * e_ss - v * p_ss / eta_dis - (1-v)* p_ss*eta_chrg,
                -p_ss / tau_ss + u_ss / tau_ss]

    return microgrid_ode


# plt.scatter(range(N), out[:,0], label="e_ss")
# plt.scatter(range(N), out[:,1], label="p_ss")
# plt.scatter(range(N), out[:,2], label="p_d")
# plt.legend()
# plt.show()

#%%
N_TEST = 3
MAX_T = 2

train_data = []

for i in range(N_TEST):
    cheb_coeff = 3*(np.random.rand(20) - 0.5)
    cheb_smpl = np.polynomial.chebyshev.chebval(sensors, cheb_coeff)
    # plt.scatter(sensors, cheb_smpl)
    # plt.show()

    ode = ode_constructor(cheb_smpl, MAX_T)
    integrator = sc.integrate.RK45(ode, 0, [100, 30],MAX_T)
    out = []
    t_out = []

    t_fixed = np.linspace(0, 2, 100)

    while integrator.status != "finished":
        integrator.step()
        out.append(integrator.y)
        t_out.append(integrator.t)

    out = np.array(out)
    t_out = np.array(t_out)
    print(out.shape)
    print(t_out.shape)

    f1 = sc.interpolate.interp1d(t_out, out[:,0])  # interpoliert entlang Zeitachse
    f2 = sc.interpolate.interp1d(t_out, out[:,1])  # interpoliert entlang Zeitachse
    out_aligned = np.array([f1(t_fixed), f2(t_fixed)]).T

    train_data.append({"cheb_coeff": cheb_coeff, "cheb_smpl": cheb_smpl, "out": out_aligned, "t": t_fixed})


#%%
np.polynomial.chebyshev.chebval(1.91, cheb_coeff)