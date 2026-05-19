from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import deepxde as dd
import scipy as sc
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval
np.random.seed(1234)

def random_chebyshev(n,M,degree,return_coeff = False):
    '''
    This generates a random function output between the domain [-1,1] along a uniform grid of size n using chebyshev polynomials.
    '''
    coeff = (np.random.rand(degree+1)-0.5)*2*np.abs(M)
    x= np.linspace(-1,1,n)
    y = chebval(x,coeff)
    if return_coeff:
        return x,y,coeff
    else:
        return x,y

#%%
sensors = (np.random.random(100) -0.5)*2

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

# def ode_constructor(cheb_coeff, max_t):
#     def ode(t,y):
#         s1 = y[0]
#         s2 = y[1]
#         u = np.polynomial.chebyshev.chebval((t/max_t  -0.5) *2, cheb_coeff)
#
#         return [s2, -np.sin(s1) + u]
#
#     return ode

class OdeCallable:
    def __init__(self, cheb_coeff, max_t):
        self.cheb_coeff = cheb_coeff
        self.max_t = max_t

    def __call__(self, t, y):
        return ode_constructor(self.cheb_coeff, self.max_t)(t, y)

def _single_creation(args):
    max_t, seed = args
    x, y, coeff= random_chebyshev(20, 1, 20, True)
    ode = OdeCallable(coeff, max_t)
    t_fixed = np.linspace(0, max_t, 100)
    sol = sc.integrate.solve_ivp(ode, [0, max_t], [0, 0], method="RK45", t_eval=t_fixed)
    chebyshev_at_sensors = np.interp(sensors, x, y)
    # TODO: _y[0] nimmt nur eine dimension der ode (auslenkung hier)
    return {"in": chebyshev_at_sensors, "cheb_coeff": coeff, "out": sol.y.T[:,0], "t": sol.t}

#%%
def create_data(size: int, max_t: float, workers: int = 8, mode: str = "process"):
    seeds = range(size)
    args = [(max_t, s) for s in seeds]

    Executor = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor

    with Executor(max_workers=workers) as pool:
        data = list(pool.map(_single_creation , args, chunksize=max(1, size // workers)))

    return data


#%%

data = create_data(1000, 2, workers=12)
# train_data = np.load("./train.npy", allow_pickle=True)
train_size = int(len(data) * 0.8)

#%%
np.save("data.npy", data)
print("Saved data to train.npy and test.npy")
#%%
train_data = data[:train_size]
test_data = data[train_size:]

X_train = (np.array([d["in"] for d in train_data], dtype=np.float32), np.array(train_data[0]['t'], dtype=np.float32).reshape((-1,1)))
y_train = np.array([d["out"] for d in train_data], dtype=np.float32)

X_test = (np.array([d["in"] for d in test_data], dtype=np.float32), np.array(test_data[0]['t'], dtype=np.float32).reshape((-1,1)))
y_test = np.array([d["out"] for d in test_data], dtype=np.float32)

cart_data = dd.data.TripleCartesianProd(X_train, y_train, X_test, y_test)

#%%
net = dd.nn.DeepONetCartesianProd([100, 40,40], [1,40], "relu", "Glorot normal")

#%%
model = dd.Model(cart_data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
#%%
losshistory, train_state = model.train(epochs=50000)


#%%
# Realistic curve test case: daily power demand profile
def realistic_demand(t):
    """Simulated daily microgrid power demand (normalized)."""
    # Base load + morning/evening peaks
    base = 0.3
    # morning = 0.4 * np.exp(-((t/2 - 0.3) ** 2) / 0.005)
    evening = 0.5 * np.exp(-((t/2 - 0.75) ** 2) / 0.008)
    return 5*(base + evening)

# Convert realistic curve to Chebyshev coefficients (20 terms)
t_norm = np.linspace(0, 1, 500)
realistic_vals = realistic_demand(t_norm)
cheb_coeffs_raw = np.polynomial.chebyshev.chebfit(t_norm, realistic_vals, 19)
realistic_vals_at_sensor = realistic_demand(sensors)

# Solve the true ODE with realistic input
ode_realistic = OdeCallable(cheb_coeffs_raw, 2)
t_eval = np.linspace(0, 2, 200)
sol_true = sc.integrate.solve_ivp(ode_realistic, [0, 2], [0,0], method="RK45", t_eval=t_eval)
y_true_realistic = sol_true.y[0]

# Predict with trained DeepONet
X_realistic = (realistic_vals_at_sensor.reshape(1, -1), t_eval.reshape(-1, 1))
y_pred_realistic = model.predict(X_realistic)[0]

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(t_eval, y_true_realistic, label="True (IVP solve)", linewidth=2.5)
plt.plot(t_eval, y_pred_realistic, label="Predicted (DeepONet)", linewidth=2.5, linestyle="--")
plt.plot(t_eval, realistic_demand(t_eval), label="Load", linewidth=2.5)
plt.xlabel("Time")
plt.ylabel("e_ss (State of Charge)")
plt.title("Realistic Demand: True vs Predicted")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Compute error for realistic case
# realistic_error = np.linalg.norm(y_true_realistic - y_pred_realistic) / np.linalg.norm(y_true_realistic)
# print(f"Realistic case L2 relative error: {realistic_error:.6f}")