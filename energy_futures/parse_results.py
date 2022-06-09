import numpy as np
import matplotlib.pyplot as plt
from robust_iron_and_steel_box import run_iron_and_steel_box
from robust_iron_and_steel_ellipse import run_iron_and_steel_ellipse
from robust_iron_and_steel_realistic import run_iron_and_steel_realistic


# fig, axs = plt.subplots(1, 1)
# axs.spines['right'].set_visible(False)
# axs.spines['top'].set_visible(False)
# axs.set_title("Iron and Steel - Realistic")
# gamma = np.logspace(-1,2,10)
# rob = np.zeros(len(gamma))
# for i in range(len(gamma)):
#     res = run_iron_and_steel_realistic(1e-6,gamma[i])
#     nom = res["nominal_objective"]
#     rob[i] = res["robust_objective"]
#     rob = np.array([0]+list(np.cumsum(np.diff(rob))/rob[0]))*100
# axs.plot(gamma, rob, c='k', lw=2)
# axs.set_xlabel("$\Gamma$")
# axs.set_ylabel("Increase in objective from nominal solution (%)")
# axs.grid()
# plt.savefig('outputs/realistic_iron_and_steel.pdf')

colors = ["k", "red", "blue", "green"]
fig, axs = plt.subplots(1, 1)
axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)
axs.set_title("Iron and Steel - Ellipse")
col_count = 0
# n = 30
# per = 4
# percentages = np.linspace(0.0001, 0.06, per)
# gamma = np.linspace(0, 40, n)
n = 20
per = 4
percentages = [0.01,0.02,0.04,0.06]
gamma = np.logspace(-2, 2, n)
rob = np.zeros(n)
for i in range(per):
    for j in range(n):
        res = run_iron_and_steel_ellipse(percentages[i], 1e-6, gamma[j])
        nom = res["nominal_objective"]
        rob[j] = res["robust_objective"]
    rob = np.array([0] + list(np.cumsum(np.diff(rob)) / nom)) * 100
    label = "Parameter uncertainty (%): " + str(np.round(percentages[i], 2))
    axs.plot(gamma, rob, c=colors[col_count], lw=2, label=label)
    col_count += 1
axs.set_xlabel("$\Gamma$")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
axs.set_xscale("log")
axs.legend()
plt.savefig("outputs/ellipse_iron_and_steel.pdf")


# ALL PARAMETERS AS A BOX
# tags = ['All Parameters','Learning Rate','Initial Capture Cost','Capture Rate']
# colors = ['k','red','blue','green']
# fig, axs = plt.subplots(1, 1)
# axs.spines['right'].set_visible(False)
# axs.spines['top'].set_visible(False)
# axs.set_title("Iron and Steel - Cartesian Product of Intervals")
# col_count = 0
# for tag_name in tags:
#     n = 20
#     percentages = np.linspace(0.0001, 0.2, n)
#     rob = np.zeros(n)
#     for i in range(n):
#         res = run_iron_and_steel_box(percentages[i],1e-6,tag=tag_name)
#         nom = res["nominal_objective"]
#         rob[i] = res["robust_objective"]
#     rob = np.array([0]+list(np.cumsum(np.diff(rob))/rob[0]))*100
#     axs.plot(percentages*100, rob, c=colors[col_count], lw=2,label=tag_name)
#     col_count += 1
# axs.legend()
# axs.set_xlabel("Parameter uncertainty (%)")
# axs.set_ylabel("Increase in objective from nominal solution (%)")
# axs.grid()
# plt.savefig('outputs/box_iron_and_steel.pdf')
