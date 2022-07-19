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
import numpy as np 
import os
import logging
import matplotlib.pyplot as plt 


def iron_and_steel(p_list):
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)

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
        return (
            UC0 * (1 - ((A0 + CR * sum(CO2) * sum(market_share)) / A0) ** (-LR))
        ) - t

    con_list += [con]

    def obj(x):
        t = x[0]
        return t

    def var_bounds(m, i):
        return (x[i][0], x[i][1])

    epsilon = 1e-4
    solver = "ipopt"
    m_upper = ConcreteModel()
    m_upper.x = Set(initialize=x.keys())
    m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
    x_vars = [m_upper.x_v[i] for i in x.keys()]
    m_upper.cons = ConstraintList()
    for p in p_list:
        p_nominal = [p[key]["val"] for key in p.keys()]
        for con in con_list:
            m_upper.cons.add(expr=con(x_vars, p_nominal) <= 0)
    m_upper.obj = Objective(expr=obj(x_vars), sense=minimize)

    res = SolverFactory(solver).solve(m_upper)
    
    return m_upper

data = pd.read_csv("data/iron_and_steel_csv.csv", skiprows=1)[:33]
plant_names = [i.replace("\xa0", "") for i in data["PLANT"].values[1:]]
plant_emissions = [
    float(i.replace(",", ""))
    for i in data["CO2 Emissions per plant (t/yr)"].values[1:]
]

plants = len(plant_names)






p_list = []

for i in range(100):
    p = {}
    p["capture_rate"] = {"val": np.random.uniform(0.6,0.7)}
    p["initial_captured"] = {"val": 10000000}
    p["learning_rate"] = {"val": np.random.uniform(0.1,0.2)}
    p["initial_capture_cost"] = {"val": 55.4085}
    for i in range(plants):
        p[plant_names[i] + ": CO2 Emissions"] = {
            "val": plant_emissions[i]
        }
    p_list.append(p)


m = iron_and_steel(p_list)

print(value(m.x_v[:]))
print(value(m.obj))

