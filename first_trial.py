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
def create_data(size: int, max_t: float):
    data = []
    for i in range(size):
        cheb_coeff = 3*(np.random.rand(20) - 0.5)
        # plt.scatter(sensors, cheb_smpl)
        # plt.show()

        ode = ode_constructor(cheb_coeff, max_t)
        t_fixed = np.linspace(0, 2, 100)
        sol = sc.integrate.solve_ivp(ode, [0,max_t], [100, 30], method="RK45", t_eval=t_fixed)

        data.append({"cheb_coeff": cheb_coeff, "out": sol.y, "t": sol.t})

    return data


#%%

train_data = create_data(100, 2)

test_data = create_data(100, 2)

#%%
np.save("train.npy", train_data)
np.save("test.npy", test_data)
print("Saved data to train.npy and test.npy")
#%%
X_train = (np.array([d["cheb_coeff"] for d in train_data], dtype=np.float32), np.float32(train_data[0]["t"]).reshape(-1,1))
y_train = np.array([d["out"][0] for d in train_data], dtype=np.float32)

X_test = (np.array([d["cheb_coeff"] for d in test_data], dtype=np.float32), np.float32(test_data[0]["t"]).reshape(-1,1))
y_test = np.array([d["out"][0] for d in test_data], dtype=np.float32)

data = dd.data.TripleCartesianProd(X_train, y_train, X_test, y_test)

#%%
net = dd.nn.DeepONetCartesianProd([20, 40, 40], [1,40,40], "relu", "Glorot normal")

#%%
model = dd.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=40000)