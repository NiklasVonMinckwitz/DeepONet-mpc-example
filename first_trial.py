from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

class OdeCallable:
    def __init__(self, cheb_coeff, max_t):
        self.cheb_coeff = cheb_coeff
        self.max_t = max_t

    def __call__(self, t, y):
        return ode_constructor(self.cheb_coeff, self.max_t)(t, y)

def _single_creation(args):
    max_t, seed = args
    rng = np.random.default_rng(seed)
    # Realistic Chebyshev coefficients: smooth decay + noise
    # This produces curves with base load + 1-2 peaks, like real power demand
    cheb_coeff = rng.normal(0, 1, 20) / np.arange(1, 21) * 20
    x = 5*np.polynomial.chebyshev.chebval(sensors, cheb_coeff)

    ode = OdeCallable(cheb_coeff, max_t)
    t_fixed = np.linspace(0, max_t, 100)
    sol = sc.integrate.solve_ivp(ode, [0, max_t], [100, 30], method="RK45", t_eval=t_fixed)
    return {"in": x, "cheb_coeff": cheb_coeff, "out": sol.y, "t": sol.t}

#%%
def create_data(size: int, max_t: float, workers: int = 8, mode: str = "process"):
    seeds = range(size)
    args = [(max_t, s) for s in seeds]

    Executor = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor

    with Executor(max_workers=workers) as pool:
        data = list(pool.map(_single_creation , args, chunksize=max(1, size // workers)))

    return data

#%%

data = create_data(1200, 2, workers=12)
# train_data = np.load("./train.npy", allow_pickle=True)

#%%
np.save("data.npy", data)
print("Saved data to train.npy and test.npy")
#%%
X_train = (np.array([d["in"] for d in data[:500]], dtype=np.float32), np.float32(data[0]["t"]).reshape(-1,1))
y_train = np.array([d["out"][0] for d in data[:500]], dtype=np.float32)

X_test = (np.array([d["in"] for d in data[500:]], dtype=np.float32), np.float32(data[0]["t"]).reshape(-1,1))
y_test = np.array([d["out"][0] for d in data[500:]], dtype=np.float32)

data = dd.data.TripleCartesianProd(X_train, y_train, X_test, y_test)

#%%
net = dd.nn.DeepONetCartesianProd([100, 40, 40], [1,40], "relu", "Glorot normal")

#%%
model = dd.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=70000)

#%%
print(X_test.shape)

#%%
# Example evaluation and plotting

# Predict on test data
y_pred = model.predict(X_test)

# Compute test L2 relative error
test_error = np.linalg.norm(y_test - y_pred) / np.linalg.norm(y_test)
print(f"Test L2 relative error: {test_error:.6f}")

# Plot true vs predicted for first 5 test samples
fig, axes = plt.subplots(5, 1, figsize=(10, 12))
for i in range(5):
    axes[i].plot(X_test[1].flatten(), y_test[:, i], label="True solution", linewidth=2)
    axes[i].plot(X_test[1].flatten(), y_pred[:, i], label="Predicted solution", linewidth=2, linestyle="--")
    axes[i].set_ylabel(f"Sample {i+1}")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel("Time")
plt.suptitle("DeepONet: True vs Predicted Solutions (Test Samples)")
plt.tight_layout()
plt.savefig("prediction_comparison.png", dpi=150)
plt.show()
print("Saved prediction comparison plot to prediction_comparison.png")

#%%
# Realistic curve test case: daily power demand profile
def realistic_demand(t):
    """Simulated daily microgrid power demand (normalized)."""
    # Base load + morning/evening peaks
    base = 0.3
    morning = 0.4 * np.exp(-((t/2 - 0.3) ** 2) / 0.005)
    evening = 0.5 * np.exp(-((t/2 - 0.75) ** 2) / 0.008)
    return 5*(base + morning + evening)

# Convert realistic curve to Chebyshev coefficients (20 terms)
t_norm = np.linspace(0, 1, 500)
realistic_vals = realistic_demand(t_norm)
cheb_coeffs_raw = np.polynomial.chebyshev.chebfit(t_norm, realistic_vals, 19)
realistic_vals_at_sensor = realistic_demand(sensors)

# Solve the true ODE with realistic input
ode_realistic = OdeCallable(cheb_coeffs_raw, 2)
t_eval = np.linspace(0, 2, 200)
sol_true = sc.integrate.solve_ivp(ode_realistic, [0, 2], [100, 30], method="RK45", t_eval=t_eval)
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