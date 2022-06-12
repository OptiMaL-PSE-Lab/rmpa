from cmath import nan
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from robust_iron_and_steel_box import run_iron_and_steel_box
from robust_iron_and_steel_ellipse import run_iron_and_steel_ellipse
from robust_iron_and_steel_realistic import run_iron_and_steel_realistic
import numpy as np 
import pickle 

g_plot = np.linspace(0.1,5,1000)
p_plot = np.exp(-(g_plot**2)/2)*100
fig,axs = plt.subplots(1,1)
axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)
axs.set_xlabel('$\Omega$')
axs.set_ylabel('Probability of constraint violation (%)')
axs.plot(g_plot,p_plot,c='k',lw=2)
axs.grid()
axs.set_yscale('log')
plt.savefig('outputs/conversion.pdf')

gamma = np.linspace(0.1,5,20)
rob = np.zeros(len(gamma))
for i in range(len(gamma)):
    res = run_iron_and_steel_realistic(1e-6,gamma[i])
    nom = res["nominal_objective"]
    rob[i] = res["robust_objective"]
rob = np.array([0]+list(np.cumsum(np.diff(rob))/nom))*100
fig, axs = plt.subplots(1, 1)
axs.plot(gamma, rob, c='k', lw=2)
axs.set_xlabel("$\Omega$")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
plt.savefig('outputs/realistic_iron_and_steel_omega.pdf')


gamma = np.linspace(0.1,5,20)
rob = np.zeros(len(gamma))
for i in range(len(gamma)):
    res = run_iron_and_steel_realistic(1e-6,gamma[i])
    nom = res["nominal_objective"]
    rob[i] = res["robust_objective"]
rob = np.array([0]+list(np.cumsum(np.diff(rob))/nom))*100
fig, axs = plt.subplots(1, 1)
prob = [(np.exp(-(gamma[i]**2)/2))*100 for i in range(len(gamma))]
axs.plot(prob, rob, c='k', lw=2)
axs.set_xlabel("Constraint violation probability (%)")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.set_xscale('log')
axs.grid()
plt.savefig('outputs/realistic_iron_and_steel_prob.pdf')


print('DONE REALISM CASE')

colors = ["k", "red", "blue", "green"]
fig, axs = plt.subplots(1, 1)
axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)
axs.set_title("Iron and Steel - Ellipse")
col_count = 0
n = 20
per = 4
percentages = [0.01,0.02,0.04,0.06]
gamma = np.linspace(0.1, 5, n)
prob = [(np.exp(-(gamma[i]**2)/2))*100 for i in range(len(gamma))]
rob = np.zeros(n)
for i in range(per):
    for j in range(n):
        res = run_iron_and_steel_ellipse(percentages[i], 1e-6, gamma[j])
        nom = res["nominal_objective"]
        rob[j] = res["robust_objective"]
    rob = np.array([0] + list(np.cumsum(np.diff(rob)) / nom)) * 100
    label = "Parameter uncertainty (%): " + str(np.round(percentages[i], 2))
    axs.plot(prob, rob, c=colors[col_count], lw=2, label=label)
    col_count += 1
axs.set_xlabel("Constraint violation probability (%)")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
axs.set_xscale("log")
axs.legend()
fig.savefig("outputs/ellipse_iron_and_steel_prob.pdf")


colors = ["k", "red", "blue", "green"]
fig, axs = plt.subplots(1, 1)
axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)
axs.set_title("Iron and Steel - Ellipse")
col_count = 0
n = 20
per = 4
percentages = [0.01,0.02,0.04,0.06]
gamma = np.linspace(0.1,5, n)
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
axs.set_xlabel("$\Omega$")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
axs.legend()
fig.savefig("outputs/ellipse_iron_and_steel_omega.pdf")

res = run_iron_and_steel_ellipse(0.05, 1e-6, 3.7)
with open('outputs/ellipse_results.pickle','wb') as handle:
    pickle.dump(res,handle)

ALL PARAMETERS AS A BOX
tags = ['All Parameters','Learning Rate','Initial Capture Cost','Capture Rate']
colors = ['k','red','blue','green']
fig, axs = plt.subplots(1, 1)
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title("Iron and Steel - Cartesian Product of Intervals")
col_count = 0
for tag_name in tags:
    n = 5
    percentages = np.linspace(0.0001, 0.2, n)
    rob = np.zeros(n)
    for i in range(n):
        res = run_iron_and_steel_box(percentages[i],1e-6,tag=tag_name)
        nom = res["nominal_objective"]
        print(nom)
        rob[i] = res["robust_objective"]
    rob = np.array([0]+list(np.cumsum(np.diff(rob))/rob[0]))*100
    axs.plot(percentages*100, rob, c=colors[col_count], lw=2,label=tag_name)
    plt.savefig('outputs/box_iron_and_steel.pdf') 
    invalid_index = [] 
    print(rob)
    for i in range(len(percentages)):
        if rob[i] < 0:
            invalid_index.append(i)


    rob = np.delete(rob,invalid_index)
    print(rob)
    col_count += 1
axs.legend()
axs.set_xlabel("Parameter uncertainty (%)")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
plt.savefig('outputs/box_iron_and_steel.pdf')
