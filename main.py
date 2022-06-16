import numpy as np
import math
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

fun_par = np.random.rand(5)
fun_par[:4] = fun_par[:4] * 5
fun = lambda x : fun_par[0] + fun_par[1] * np.sin(fun_par[2] * x + fun_par[3])

num_samples = 100

x_train = np.sort(np.random.rand(num_samples, 1) * (2 * math.pi), axis=0)
x_train = np.linspace(0, 2* math.pi, num=num_samples)[:, np.newaxis]

y_train_no_noise = fun(x_train)
y_train = y_train_no_noise +\
       np.random.normal(loc=0.0, scale=0.1 * fun_par[4], size=(num_samples, 1))

model = MLPRegressor(
        hidden_layer_sizes=(10,10), 
        max_iter=25000, 
        n_iter_no_change=25000,
        activation='logistic', 
        verbose=True,
        alpha=0.000000001
    )
regr = model.fit(x_train, y_train)

x_pred = np.linspace(0, 2* math.pi, num=100)[:, np.newaxis]
y_pred = regr.predict(x_pred)

fig, axs = plt.subplots(2)
axs[0].plot(x_pred, y_pred, label="Predicted signal")
axs[0].plot(x_pred, fun(x_pred), label="Expected signal")
axs[0].legend()

axs[1].plot(x_train, y_train, label="Measured signal")
axs[1].plot(x_train, regr.predict(x_train)[:, np.newaxis], label="Expected signal")
axs[1].legend()
plt.show()
