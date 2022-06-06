import numpy as np
import matplotlib.pyplot as plt
from robust_iron_and_steel import run_iron_and_steel

n = 10
percentages = np.linspace(0.0001, 0.1, n)
rob = np.zeros(n)
for i in range(n):
    res = run_iron_and_steel(percentages[i])
    nom = res["nominal_objective"]
    rob[i] = res["robust_objective"]

fig, axs = plt.subplots(1, 1)
axs.set_title("Iron and Steel")
axs.plot(percentages, rob, c="k", lw=1)
axs.set_xlabel("% Uncertainty (all parameters)")
axs.set_ylabel("Objective Value")
axs.grid()
plt.show()
