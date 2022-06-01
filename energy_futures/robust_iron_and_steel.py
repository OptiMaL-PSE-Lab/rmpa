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
import logging

logging.getLogger("pyomo.core").setLevel(logging.ERROR)
os.system("clear")

data = pd.read_csv("data/iron_and_steel_csv.csv", skiprows=1)[:33]
plant_names = [i.replace("\xa0", "") for i in data["PLANT"].values[1:]]
plant_emissions = [
    float(i.replace(",", "")) for i in data["CO2 Emissions per plant (t/yr)"].values[1:]
]
plants = len(plant_names)

p = {}
p["capture_rate"] = {"val": 0.63, "unc": 0.05}
p["initial_captured"] = {"val": 10000000, "unc": 1000}
p["learning_rate"] = {"val": 0.18, "unc": 0.1}
p["initial_capture_cost"] = {"val": 55.4085, "unc": 1}
for i in range(plants):
    p[plant_names[i] + ": CO2 Emissions"] = {"val": plant_emissions[i], "unc": 500}

x = {}
x["t"] = [-1e20, 1e20]
for plant in plant_names:
    x[plant + ": Carbon Tax"] = [0, 100]
for plant in plant_names:
    x[plant + ": Market Share"] = [0, 1]

con_list = []


def parse_vars(x):
    carbon_tax = x[1 : plants + 1]
    market_share = x[1 + plants :]
    t = x[0]
    return carbon_tax, market_share, t


def parse_params(p):
    CR = p[0]
    UC0 = p[3]
    CO2 = p[4:]
    A0 = p[1]
    LR = -log(1 - p[2]) / log(2)
    return CR, UC0, CO2, A0, LR


def make_c(i):
    def c(x, p):
        CR, UC0, CO2, A0, LR = parse_params(p)
        carbon_tax, market_share, t = parse_vars(x)
        return CO2[i] * (UC0 - CR * carbon_tax[i])

    return c


for i in range(plants):
    c = make_c(i)
    con_list += [c]


def make_c(i):
    def c(x, p):
        CR, UC0, CO2, A0, LR = parse_params(p)
        carbon_tax, market_share, t = parse_vars(x)
        CO2_captured = [CO2[j] for j in range(plants)]
        plant_improvement = market_share[i] - CO2_captured[i] / sum(CO2_captured)
        return plant_improvement

    return c


for i in range(plants):
    c = make_c(i)
    con_list += [c]


def minimum_improvement(x, p):
    CR, UC0, CO2, A0, LR = parse_params(p)
    carbon_tax, market_share, t = parse_vars(x)
    return 0.2 - (1 - ((A0 + CR * sum(CO2) * sum(market_share)) / A0) ** (-LR))


con_list += [minimum_improvement]


def con(x, p):
    CR, UC0, CO2, A0, LR = parse_params(p)
    carbon_tax, market_share, t = parse_vars(x)
    return (UC0 * (1 - ((A0 + CR * sum(CO2) * sum(market_share)) / A0) ** (-LR))) - t


con_list += [con]


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


epsilon = 1e-6
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
term_con = res.solver.termination_condition
if term_con is TerminationCondition.infeasible:
    print("Problem is robustly infeasible...")
    feasible = False
    res = {}
    res["robust_solution"] = None
    res["nominal_solution"] = None

else:
    x_opt = value(m_upper.x_v[:])
    print(x_opt)
    x_opt_nominal = x_opt
    while True:
        robust = True
        for con in con_list:
            m_lower = build_lower_problem(con)
            try:
                SolverFactory(solver).solve(m_lower)
                p_opt = value(m_lower.p_v[:])
                print("Constraint violation: ", value(m_lower.obj), end="\r")
                for k in range(len(p_opt)):
                    if p_opt[k] is None:
                        p_opt[k] = p_nominal[k]
                if value(m_lower.obj) > epsilon:
                    robust = False
                    m_upper.cons.add(expr=con(x_vars, p_opt) <= 0)
            except ValueError:
                continue
        if robust is True:
            res = {}
            robust_solution = {}
            nominal_solution = {}
            for v in range(len(x)):
                robust_solution[list(x.keys())[v]] = x_opt[v]
                nominal_solution[list(x.keys())[v]] = x_opt_nominal[v]
            res["robust_solution"] = robust_solution
            res["nominal_solution"] = nominal_solution
            break
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
            break
        x_opt = value(m_upper.x_v[:])

print(res)
