import pandas as pd
import numpy as np
from tqdm import tqdm
from pyomo.environ import (
    log,
    TerminationCondition,
    ConcreteModel,
    Set,
    Var,
    Reals,
    ConstraintList,
    Objective,
    minimize,
    SolverFactory,
    value,
    maximize,
    Constraint,
    sqrt,
)
import time
import matplotlib.pyplot as plt
import os
import pickle
import multiprocessing as mp
import logging
import platform

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
os.system("clear")

'''
This code was created by Tom Savage 13/10/2022

For help email trs20@ic.ac.uk or ring 07446880063. 
'''

def run_fuel_switching_ellipse(percentage, epsilon, g):
    # path where data of multiple parameters lives
    data_path = "data/example_data.csv"
    n_data = 50 
    data = pd.read_csv(data_path,nrows=n_data)
    boiler_names = data['Technology'].values
    boiler_energy_required = data['Thermal Energy required from fuel (MJ/year)'].values
    boiler_energy_required = np.random.uniform(8000000,10000000,n_data)
    boilers = len(boiler_names)

    # NOTE 
    # To define uncertainty enter the range of each parameter below.
    # If no value for 'unc' is entered then a value will be assigned based 
    # on the following variable
    percentage_uncertainty = percentage

    # if all_percentage is 'True', given uncertainties are disregarded
    # and they are all given a base percentage 
    all_percentage = True

    p = {}
    # parameters that appear in the problem must be in this format
    p["Natural Gas LHV (MJ/kg)"] = {"val": 42,'unc':5}
    p["Hydrogen LHV (MJ/kg)"]    = {"val": 120,'unc':10}
    p["Natural Gas Price (£/tonne)"] = {"val": 292,'unc':20}
    p["Hydrogen Price (£/tonne)"]    = {"val": 1800,'unc':20}
    p["Natural Gas Boiler Efficiency"] = {"val": 0.9,'unc':0.05}
    p["Hydrogen Boiler Efficiency"] = {"val": 0.9,'unc':0.05}
    p["Natural Gas kgCO2e/MWh"] = {"val": 200,'unc':5}
    p["Hydrogen (kgCO2e/MWh)"] = {"val": 20,'unc':3}
    p["A0"] = {"val": 96071.958,'unc':100}
    p["Learning Rate"] = {"val": 0.07,'unc':0.01}
    p["UC0"] = {"val": 1800,'unc':20}

    # here we iteratively create parameters from the list of energies imported earlier
    for i in range(boilers):
        p[boiler_names[i] + ": Thermal Energy required from fuel (MJ/year)"] = {
            "val": boiler_energy_required[i],'unc': boiler_energy_required[i] * 0.05
        }

    # iteratively define the uncertainty to be a percentage, or just keep 
    # the value if it already exists
    for k, v in p.items():
        try:
            key_test = p[k]['unc']
        except KeyError:
            p[k]["unc"] = p[k]["val"] * percentage_uncertainty / 100
        if all_percentage is True:
            p[k]["unc"] = p[k]["val"] * percentage_uncertainty / 100
    # Assign decision variables here. 
    # The name of each variable is prepended with the name of the boiler and each
    # is associated with upper and lower bounds
    x = {}
    x["t"] = [-1e20, 1e20]
    for boiler in boiler_names:
        x[boiler + ": Renewable hydrogen incentive (£/MWh)"] = [0, 50]
    for boiler in boiler_names:
        x[boiler + ": Market Share"] = [0, 1]

    # this is needed to store all constraints
    con_list = []

    # this iteratively defines constraint functions based on an index i
    def make_c(i):
        def c(x, p):
            '''
            CONSTRAINT GOES HERE 
            '''
            # name of boiler at index i
            boiler = boiler_names[i]
            H = (3600 * p['Hydrogen Boiler Efficiency'] * p['UC0'] ) / (1000 * p['Hydrogen LHV (MJ/kg)'])
            D = 3.6 * (p['Natural Gas Price (£/tonne)'] / p['Natural Gas LHV (MJ/kg)'])
            return (H - x[boiler + ": Renewable hydrogen incentive (£/MWh)"]) - D # <= 0 
        return c

    # this adds constraints for indexes to the list
    # ... don't touch!
    for i in range(boilers):
        c = make_c(i)
        con_list += [c]

    def make_c(i):
        def c(x, p):
            '''
            CONSTRAINT GOES HERE 
            '''
            # name of boiler at index i
            boiler = boiler_names[i]
            boiler_energy_total = [p[boiler_names[j] + ": Thermal Energy required from fuel (MJ/year)"]/(1000*p["Hydrogen LHV (MJ/kg)"]) for j in range(boilers)]
            return x[boiler + ": Market Share"] - boiler_energy_total[i] / sum(boiler_energy_total)
        return c

    # this adds constraints for indexes to the list
    # ... don't touch!
    for i in range(boilers):
        c = make_c(i)
        con_list += [c]

    def c(x, p):
        '''
        CONSTRAINT GOES HERE 
        '''
        # name of boiler at index i
        boiler_energy_total = [p[boiler_names[j] + ": Thermal Energy required from fuel (MJ/year)"]/(1000*p["Hydrogen LHV (MJ/kg)"]) for j in range(boilers)]
        A = p['A0'] + sum([x[boiler_names[j]+ ": Market Share"]  for j in range(boilers)]) * sum(boiler_energy_total)
        UC = p['UC0'] * (A / p['A0']) ** (log(1 - p['Learning Rate']) / log(2)) 
        return (p['UC0'] - UC) / p['UC0'] - 0.04736

    # adding constraint to list
    con_list += [c]

    def c(x, p):
        '''
        OBJECTIVE CONSTRAINT GOES HERE 
        This should have the form: x['t'] - obj(x,p)
        If you want to know why look up 'epigraph form' 
        of an uncertain optimisation problem
        '''
        boiler_energy_total = [p[boiler_names[j] + ": Thermal Energy required from fuel (MJ/year)"]/(1000*p["Hydrogen LHV (MJ/kg)"]) for j in range(boilers)]
        A = p['A0'] + sum([x[boiler_names[j]+ ": Market Share"]  for j in range(boilers)]) * sum(boiler_energy_total)
        UC = p['UC0'] * (A / p['A0']) ** (log(1 - p['Learning Rate']) / log(2)) 
        return x['t'] - (p['UC0'] - UC)

    # adding objective constraint to list
    con_list += [c]


    # Don't touch anything from here! This should... all work when run
    def obj(x):
        return -x['t']

    def var_bounds(m, i):
        return (x[i][0], x[i][1])

    def uncertain_bounds(m, i):
        return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])

    snom = time.time()
    solver = "ipopt"
    m_upper = ConcreteModel()
    m_upper.x = Set(initialize=x.keys())
    m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
    p_nominal = {}
    for pk,pi in p.items():
        p_nominal[pk] = pi['val']
    m_upper.cons = ConstraintList()
    for con in con_list:
        m_upper.cons.add(expr=con(m_upper.x_v, p_nominal) <= 0)
    m_upper.obj = Objective(expr=obj(m_upper.x_v), sense=minimize)
    res = SolverFactory(solver).solve(m_upper)
    nominal_obj = value(m_upper.obj)
    term_con = res.solver.termination_condition
    x_opt = value(m_upper.x_v[:])

    enom = time.time()


    global solve_subproblem

    def solve_subproblem(i, x_opt):
        s = time.time()
        con = con_list[i]
        m = ConcreteModel()
        m.p = Set(initialize=p.keys())
        m.p_v = Var(m.p, domain=Reals, bounds=uncertain_bounds)

        upper = [p[str(i)]["val"] + p[i]["unc"] for i in p.keys()]
        lower = [(p[str(i)]["val"] - p[i]["unc"]) for i in p.keys()]
        sum_p = 0 
        param_vars = [m.p_v[str(i)] for i in p.keys()]
        for i in range(len(param_vars)):
            if upper[i]-lower[i] > 1e-20:
                p_n = (((param_vars[i]-lower[i])/(upper[i]-lower[i]))*2)-1
                sum_p += p_n**2 
        m.ellipse = Constraint(expr= sqrt(sum_p) <= g)


        m.obj = Objective(expr=con(x_opt, m.p_v), sense=maximize)
        try:
            solvern = SolverFactory("ipopt")
            solvern.options["max_iter"] = 10000
            solvern.solve(m)
            p_opt_list = value(m.p_v[:])
            p_opt = {}
            p_keys = list(p.keys())
            for k in range(len(p_opt_list)):
                if p_opt_list[k] is None:
                    p_opt_list[k] = p_nominal[p_keys[k]]
                p_opt[p_keys[k]] = p_opt_list[k]
            if value(m.obj) > epsilon:
                #print('Solved subproblem ',i,' in',time.time()-s,' seconds')
                return [value(m.obj), p_opt]
            else:
                #print('Solved subproblem ',i,' in',time.time()-s,' seconds')
                return [value(m.obj)]
        except ValueError:
            #print('Failed to solve subproblem ',i,' ...')
            return [None]

    pool = mp.Pool(mp.cpu_count()-2)
    spt = []
    while True:

        x_opt_nominal = value(m_upper.x_v[:])
        x_opt = {}
        for v in range(len(x)):
            x_opt[list(x.keys())[v]] = x_opt_nominal[v]
        robust = True
        s_s = time.time()
        res = pool.starmap(solve_subproblem, [(i, x_opt) for i in range(len(con_list))])
        e_s = time.time()
        spt.append(e_s - s_s)


        robust = True
        mcv = 0 
        for i in range(len(res)):
            if len(res[i]) > 1:
                robust = False
                if res[i][0] > mcv:
                    mcv = res[i][0]
                con = con_list[i]
                m_upper.cons.add(expr=con(m_upper.x_v, res[i][1]) <= 0)
        print('Maximum constraint violation: ',mcv)

        if robust is True:
            res = {}
            nominal_solution = {}
            for v in range(len(x)):
                robust_solution = x_opt
                nominal_solution[list(x.keys())[v]] = x_opt_nominal[v]
            print("Problem is robustly feasible")
            res["robust_solution"] = robust_solution
            res["nominal_solution"] = nominal_solution
            res["robust_objective"] = value(m_upper.obj)
            res["nominal_objective"] = nominal_obj

            break
        print("Solving upper level problem")
        res = SolverFactory(solver).solve(m_upper)
        term_con = res.solver.termination_condition
        if term_con is TerminationCondition.infeasible:
            print("Problem is robustly infeasible...")
            res = {}
            nominal_solution = {}
            for v in range(len(x)):
                nominal_solution[list(x.keys())[v]] = x_opt_nominal[v]
            res["robust_solution"] = None
            res["nominal_solution"] = x_opt_nominal
            res["robust_objective"] = None
            res["nominal_objective"] = nominal_obj
            break

    return res


colors = ["k", "red", "blue", "green"]
fig, axs = plt.subplots(1, 1)
axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)
axs.set_title("Fuel Switching - Ellipse")
axs.set_xlabel("Constraint violation probability (%)")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
axs.set_xscale("log")
axs.legend()
col_count = 0
n = 20
per = 4
percentages = [0.02,0.04,0.06,0.08]
gamma = np.linspace(0, 5, n)
prob = [(np.exp(-(gamma[i]**2)/2))*100 for i in range(len(gamma))]
for i in range(per):
    rob = np.zeros(n)
    for k in range(n):
        rob[k] = None
    rob[0] = 0 
    for j in range(1,n):
        res = run_fuel_switching_ellipse(percentages[i], 1e-4, gamma[j])
        nom = res["nominal_objective"]
        rob[j] = res["robust_objective"]
        print('Nominal: ',nom,' Robust: ',rob[j],'\n')
        if res["robust_objective"] != None:
            rob[j] = -((res["robust_objective"]/nom)-1)*100
        else:
            break 
        axs.plot(prob[:j+1], rob[:j+1], c=colors[col_count], lw=2)
        plt.savefig('outputs/ellipse_fuel_switching_all.png') 
    label = "% Uncertainty: " + str(100*np.round(percentages[i], 2))
    axs.plot(prob, rob, c=colors[col_count], lw=2,label=label)
    plt.savefig('outputs/ellipse_fuel_switching_all.pdf') 
    col_count += 1
axs.set_xlabel("Constraint violation probability (%)")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
axs.set_xscale("log")
axs.legend()
fig.savefig("outputs/ellipse_fuel_switching_prob_all.pdf")


# colors = ['k']
# fig, axs = plt.subplots(1, 1)
# axs.spines["right"].set_visible(False)
# axs.spines["top"].set_visible(False)
# axs.set_title("Fuel Switching - Ellipse")
# axs.set_xlabel("Constraint violation probability (%)")
# axs.set_ylabel("Increase in objective from nominal solution (%)")
# axs.grid()
# axs.set_xscale("log")
# col_count = 0
# n = 20
# gamma = np.linspace(0, 5, n)
# prob = [(np.exp(-(gamma[i]**2)/2))*100 for i in range(len(gamma))]
# rob = np.zeros(n)
# for k in range(n):
#     rob[k] = None
# rob[0] = 0 
# for j in range(1,n):
#     res,sp_av,m,t_nom = run_fuel_switching_ellipse(0, 1e-6, gamma[j])
#     nom = res["nominal_objective"]
#     rob[j] = res["robust_objective"]
#     if res["robust_objective"] != None:
#         rob[j] = -((res["robust_objective"]/nom)-1)*100
#     else:
#         break 
#     axs.plot(prob[:j+1], rob[:j+1], c=colors[col_count], lw=2)
#     plt.savefig('outputs/ellipse_fuel_switching_real.png') 
# axs.plot(prob, rob, c=colors[col_count], lw=2)
# plt.savefig('outputs/ellipse_fuel_switching_real.pdf') 
# axs.set_xlabel("Constraint violation probability (%)")
# axs.set_ylabel("Increase in objective from nominal solution (%)")
# axs.grid()
# axs.set_xscale("log")
# fig.savefig("outputs/ellipse_fuel_switching_prob_real.pdf")