import pandas as pd

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
import os
import pickle
import logging

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


p = {}
p["Natural Gas LHV (MJ/kg)"] = {"val": 42, "unc": 0.5}
p["Hydrogen LHV (MJ/kg)"] = {"val": 120, "unc": 0.5}
p["Natural Gas Price (£/tonne)"] = {"val": 292, "unc": 30}
p["Hydrogen Price (£/tonne)"] = {"val": 1800, "unc": 100}
p["Natural Gas Boiler Efficiency"] = {"val": 0.9, "unc": 0.05}
p["Hydrogen Boiler Efficiency"] = {"val": 0.9, "unc": 0.05}
p["Natural Gas kgCO2e/MWh"] = {"val": 200, "unc": 0.01}
p["Hydrogen kgCO2e/MWh"] = {"val": 20, "unc": 0.01}
p["A0"] = {"val": 96071.958, "unc": 0.001}
p["LR"] = {"val": 0.07, "unc": 0.005}
p["UC0"] = {"val": 1800, "unc": 50}

for i in range(boilers):
    p[boiler_names[i] + ": CO2 Emissions from natural gas combustion (kg/yr)"] = {
        "val": boiler_emissions[i],
        "unc": 50,
    }
for i in range(boilers):
    p[
        boiler_names[i]
        + ": Price of steam produced from Natural gas combustion (£/MWh)"
    ] = {"val": boiler_cost_of_steam[i], "unc": 50}
for i in range(boilers):
    p[boiler_names[i] + ": Thermal Energy required from fuel (MJ/year)"] = {
        "val": boiler_energy_required[i],
        "unc": 50,
    }

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
    boiler_CO2 = p[ind : ind + boilers]
    ind += boilers
    boiler_steam_price = p[ind : ind + boilers]
    ind += boilers
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
        boiler_CO2,
        boiler_steam_price,
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
            C,
            D,
            E,
        ) = parse_params(p)
        O, S, t = parse_vars(x)
        J = E[i] / (3600 * H_eff)
        H = E[i] / H_LHV
        I = H / 1000
        K = I * UC0
        M = K / J
        return (M - O[i]) - D[i]

    return c


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
            C,
            D,
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
        C,
        D,
        E,
    ) = parse_params(p)
    O, S, t = parse_vars(x)
    I = [(E[j] / H_LHV) / 1000 for j in range(boilers)]
    U = [S[j] * sum(I) for j in range(boilers)]
    A = A0 + sum(U)
    UC = UC0 * (A / A0) ** (-LR)
    return (UC0 - UC) / UC0 - 0.04


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
        C,
        D,
        E,
    ) = parse_params(p)
    O, S, t = parse_vars(x)
    I = [(E[j] / H_LHV) / 1000 for j in range(boilers)]
    U = [S[j] * sum(I) for j in range(boilers)]
    A = A0 + sum(U)
    UC = UC0 * (A / A0) ** (-LR)
    return (UC0 - UC) - t


con_list += [c]


def obj(x):
    t = x[0]
    return t


def var_bounds(m, i):
    return (x[i][0], x[i][1])


def uncertain_bounds(m, i):
    return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])


def build_lower_problem(con):
    m = ConcreteModel()
    m.p = Set(initialize=p.keys())
    m.p_v = Var(m.p, domain=Reals, bounds=uncertain_bounds)
    param_vars = [m.p_v[str(i)] for i in p.keys()]
    m.obj = Objective(expr=con(x_opt, param_vars), sense=maximize)
    return m


epsilon = 1e-4
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
upper_iteration = 0
if term_con is TerminationCondition.infeasible:
    print("Problem is nominally infeasible...")
    res = {}
    res["robust_solution"] = None
    res["nominal_solution"] = None
    res["robust_objective"] = None
    res["nominal_objective"] = None

else:
    x_opt = value(m_upper.x_v[:])
    x_opt_nominal = x_opt
    while True:
        x_opt = value(m_upper.x_v[:])
        robust = True
        max_con = -1e30
        for con in con_list:
            m_lower = build_lower_problem(con)
            try:
                SolverFactory(solver).solve(m_lower)
                p_opt = value(m_lower.p_v[:])
                for k in range(len(p_opt)):
                    if p_opt[k] is None:
                        p_opt[k] = p_nominal[k]
                if value(m_lower.obj) > epsilon:
                    robust = False
                    print(value(m_lower.obj))
                    m_upper.cons.add(expr=con(x_vars, p_opt) <= 0)
                if value(m_lower.obj) > max_con:
                    max_con = value(m_lower.obj)
            except ValueError:
                continue
        upper_iteration += 1
        print(
            "Upper iteration: ",
            upper_iteration,
            "\t Max constraint violation: ",
            max_con,
        )
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


print(res)
