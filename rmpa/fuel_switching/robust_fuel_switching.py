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
)
import time
import matplotlib.pyplot as plt 
import os
import pickle
import multiprocessing as mp
import logging
import platform
print(platform.processor())

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
os.system("clear")

data = pd.read_csv("data/fuel_switching_csv.csv", skiprows=6)[:355]

boiler_names = [i.replace("\xa0", "") for i in data["Technology"].values[1:]]
boiler_emissions = [
    float(i.replace(",", ""))
    for i in data["CO2 Emissions from natural gas combustion (kg/yr)"].values[1:]
]
boiler_cost_of_steam = data[
    "Price of steam produced from Natural gas combustion (£/MWh)"
].values[1:]
boiler_energy_required = [
    float(i.replace(",", ""))
    for i in data["Thermal Energy required from fuel (MJ/year)"].values[1:]
]
boilers = len(boiler_names)


def run_fuel_switching_box(percen,epsilon,tag):

    p = {}
    p["Natural Gas LHV (MJ/kg)"] = {"val": 42}
    p["Hydrogen LHV (MJ/kg)"] = {"val": 120}
    p["Natural Gas Price (£/tonne)"] = {"val": 292}
    p["Hydrogen Price (£/tonne)"] = {"val": 1800}
    p["Natural Gas Boiler Efficiency"] = {"val": 0.9}
    p["Hydrogen Boiler Efficiency"] = {"val": 0.9}
    p["Natural Gas kgCO2e/MWh"] = {"val": 200}
    p["Hydrogen (kgCO2e/MWh)"] = {"val": 20}
    p["A0"] = {"val": 96071.958}
    p["Learning Rate"] = {"val": 0.07}
    p["UC0"] = {"val": 1800}
    for i in range(boilers):
        p[boiler_names[i] + ": Thermal Energy required from fuel (MJ/year)"] = {
            "val": boiler_energy_required[i]
        }

    for k,v in p.items():
        p[k]['unc'] = 0

    if tag == 'All Parameters':
        for k,v in p.items():
            p[k]['unc'] = p[k]['val'] * percen
    elif tag == 'Learning Rate':
        p[tag]['unc'] = p[tag]['val'] * percen

    elif tag == 'Reported Energy Required':
        k = list(p.keys())[:11]
        for ki in k:
            p[ki]['unc'] = p[ki]['val'] * percen


    x = {}
    x["t"] = [-1e20, 1e20]
    for boiler in boiler_names:
        x[boiler + ": Renewable hydrogen incentive (£/MWh)"] = [0, 50]
    for boiler in boiler_names:
        x[boiler + ": Market Share"] = [0, 1]

    con_list = []


    def parse_vars(x):
        incentive = x[1 : boilers + 1]
        market_share = x[1 + boilers :]
        t = x[0]
        return incentive, market_share, t


    def parse_params(p):
        NG_LHV = p[0]
        H_LHV = p[1]
        NG_price = p[2]
        H_price = p[3]
        NG_eff = p[4]
        H_eff = p[5]
        NG_rho = p[6]
        H_rho = p[7]
        A0 = p[8]
        LR = -log(1 - p[9]) / log(2)
        UC0 = p[10]
        ind = 11
        boiler_energy = p[ind : ind + boilers]
        return (
            NG_LHV,
            H_LHV,
            NG_price,
            H_price,
            NG_eff,
            H_eff,
            NG_rho,
            H_rho,
            A0,
            LR,
            UC0,
            boiler_energy,
        )


    def make_c(i):
        def c(x, p):
            (
                NG_LHV,
                H_LHV,
                NG_price,
                H_price,
                NG_eff,
                H_eff,
                NG_rho,
                H_rho,
                A0,
                LR,
                UC0,
                E,
            ) = parse_params(p)
            O, S, t = parse_vars(x)
            J = E[i] / (3600 * H_eff)
            H = E[i] / H_LHV
            I = H / 1000
            K = I * UC0
            M = K / J
            D = 3.6*(NG_price/NG_LHV)
            return (M - O[i]) - D

        return c


    # for i in range(boilers):
    #     c = make_c(i)
    #     con_list += [c]

    # def make_c(i):
    #     def c(x, p):
    #         (
    #             NG_LHV,
    #             H_LHV,
    #             NG_price,
    #             H_price,
    #             NG_eff,
    #             H_eff,
    #             NG_rho,
    #             H_rho,
    #             A0,
    #             LR,
    #             UC0,
    #             E,
    #         ) = parse_params(p)
    #         O, S, t = parse_vars(x)
    #         J = E[i] / (3600 * H_eff)
    #         H = E[i] / H_LHV
    #         I = H / 1000
    #         K = I * UC0
    #         M = K / J
    #         Co = ((E[i]/NG_eff)/3600)*NG_rho
    #         Co_H = ((E[i]/3600)*H_eff*H_rho)
    #         return Co_H-Co

    #     return c


    for i in range(boilers):
        c = make_c(i)
        con_list += [c]


    def make_c(i):
        def c(x, p):
            (
                NG_LHV,
                H_LHV,
                NG_price,
                H_price,
                NG_eff,
                H_eff,
                NG_rho,
                H_rho,
                A0,
                LR,
                UC0,
                E,
            ) = parse_params(p)
            O, S, t = parse_vars(x)
            I = [(E[j] / H_LHV) / 1000 for j in range(boilers)]
            F = I[i] / sum(I)
            return S[i] - F

        return c


    for i in range(boilers):
        c = make_c(i)
        con_list += [c]


    def c(x, p):
        (
            NG_LHV,
            H_LHV,
            NG_price,
            H_price,
            NG_eff,
            H_eff,
            NG_rho,
            H_rho,
            A0,
            LR,
            UC0,
            E,
        ) = parse_params(p)
        O, S, t = parse_vars(x)
        I = [(E[j] / H_LHV) / 1000 for j in range(boilers)]
        U = [S[j] * sum(I) for j in range(boilers)]
        A = A0 + sum(U)
        UC = UC0 * (A / A0) ** (-LR)
        return (UC0 - UC) / UC0 - 0.04736


    con_list += [c]


    def c(x, p):
        (
            NG_LHV,
            H_LHV,
            NG_price,
            H_price,
            NG_eff,
            H_eff,
            NG_rho,
            H_rho,
            A0,
            LR,
            UC0,
            E,
        ) = parse_params(p)
        O, S, t = parse_vars(x)
        I = [(E[j] / H_LHV) / 1000 for j in range(boilers)]
        U = [S[j] * sum(I) for j in range(boilers)]
        A = A0 + sum(U)
        UC = UC0 * (A / A0) ** (-LR)
        return t-(UC0 - UC)


    con_list += [c]


    def obj(x):
        O, S, t = parse_vars(x)
        return -t


    def var_bounds(m, i):
        return (x[i][0], x[i][1])


    def uncertain_bounds(m, i):
        return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])


    snom = time.time()
    solver = "ipopt"
    m_upper = ConcreteModel()
    m_upper.x = Set(initialize=x.keys())
    m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
    p_nominal = [p[key]["val"] for key in p.keys()]
    x_vars = [m_upper.x_v[i] for i in x.keys()]
    m_upper.cons = ConstraintList()
    for con in con_list:
        m_upper.cons.add(expr=con(x_vars, p_nominal) <= 0)
    m_upper.obj = Objective(expr=obj(x_vars), sense=minimize)
    res = SolverFactory(solver).solve(m_upper)
    nominal_obj = value(m_upper.obj)
    term_con = res.solver.termination_condition

    x_opt = value(m_upper.x_v[:])
    enom = time.time()

    x_opt = value(m_upper.x_v[:])
    x_opt_nominal = x_opt
    global solve_subproblem
    def solve_subproblem(i,x_opt):
        con = con_list[i]
        m = ConcreteModel()
        m.p = Set(initialize=p.keys())
        m.p_v = Var(m.p, domain=Reals, bounds=uncertain_bounds)
        param_vars = [m.p_v[str(i)] for i in p.keys()]
        m.obj = Objective(expr=con(x_opt, param_vars), sense=maximize)
        try:
            solvern = SolverFactory('ipopt')
            solvern.options['max_iter']= 10000
            solvern.solve(m)
            p_opt = value(m.p_v[:])
            for k in range(len(p_opt)):
                if p_opt[k] is None:
                    p_opt[k] = p_nominal[k]
            if value(m.obj) > epsilon:
                return [value(m.obj),p_opt]
            else:
                return [None]
        except ValueError:
            return [None]

    pool = mp.Pool(mp.cpu_count())
    spt = []
    while True:
    # for it in range(1):
        x_opt = value(m_upper.x_v[:])
        robust = True
        max_con = -1e30

        # res = pool.map(
        #     solve_subproblem,
        #     np.arange(len(con_list)),
        # )

        s_s = time.time()
        res = pool.starmap(solve_subproblem, [(i,x_opt) for i in range(len(con_list))])
        e_s = time.time()
        spt.append(e_s-s_s)
        robust = True
        for i in range(len(res)):
            if len(res[i]) > 1:
                robust = False
                con = con_list[i]
                m_upper.cons.add(expr=con(x_vars, res[i][1]) <= 0)


        if robust is True:
            res = {}
            robust_solution = {}
            nominal_solution = {}
            for v in range(len(x)):
                robust_solution[list(x.keys())[v]] = x_opt[v]
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
    return res,np.mean(np.array(spt)),m_upper,enom-snom




# start = time.time()
# res,spt,m,tnom = run_fuel_switching_box(0.05,1e-4,tag='All Parameters')
# end = time.time()
# print('total time: ',end-start)
# print('av subprobs time: ',spt)
# print('total end cons: ',len(m.cons))
# print('nominal time: ',tnom)




tags = ['All Parameters','Learning Rate','Reported Energy Required']
colors = ['k','red','blue','green']
fig, axs = plt.subplots(1, 1)
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
#axs.set_title("Iron and Steel - Cartesian Product of Intervals")
col_count = 0
for tag_name in tags:
    n = 20
    percentages = np.linspace(0.0, 0.3, n)
    rob = np.zeros(n)
    for i in range(n):
        rob[i] = None
    rob[0] = 0 

    for i in range(1,n):
        res,subprob_av,m,nominal_time = run_fuel_switching_box(percentages[i],1e-4,tag=tag_name)
        print(res)
        nom = res["nominal_objective"]
        if res["robust_objective"] != None:
            rob[i] = -((res["robust_objective"]/nom)-1)*100
        else:
            break 
        axs.plot(percentages[:i+1]*100, rob[:i+1], c=colors[col_count], lw=2)
        plt.savefig('outputs/box_fuel_switching.png') 
    axs.plot(percentages*100, rob, c=colors[col_count], lw=2,label=tag_name)
    plt.savefig('outputs/box_fuel_switching.pdf') 
    invalid_index = [] 
    #print(rob)
    for i in range(len(percentages)):
        if rob[i] < 0:
            invalid_index.append(i)
    rob = np.delete(rob,invalid_index)
    #print(rob)
    col_count += 1

axs.legend()
axs.set_xlabel("Parameter uncertainty (%)")
axs.set_ylabel("Increase in objective from nominal solution (%)")
axs.grid()
plt.savefig('outputs/box_fuel_switching.pdf')
