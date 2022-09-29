import pandas as pd
from pyomo.environ import (
    log,
    TerminationCondition,
    ConcreteModel,
    Constraint,
    Set,
    Var,
    Reals,
    ConstraintList,
    Objective,
    minimize,
    SolverFactory,
    value,
    maximize,
    Binary,
    summation
)
import numpy as np 
import os
import logging
import matplotlib.pyplot as plt 


def iron_and_steel(u,chance,solver,n_p_list):
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)
    p_list = []
    for i in range(n_p_list * 66):
        p = {}
        p["capture_rate"] = {"val": np.random.uniform(0.63-0.63*u,0.63+0.63*u)}
        p["initial_captured"] = {"val": np.random.uniform(10000000-10000000*u,10000000+10000000*u)}
        p["learning_rate"] = {"val": np.random.uniform(0.18-0.18*u,0.18+0.18*u)}
        p["initial_capture_cost"] = {"val": np.random.uniform(55.4085-55.4085*u,55.4085+55.4085*u)}
        for i in range(plants):
            p[plant_names[i] + ": CO2 Emissions"] = {
                "val": np.random.uniform(plant_emissions[i]-plant_emissions[i]*u,plant_emissions[i]+plant_emissions[i]*u)
            }
        p_list.append(p)
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


    m_upper = ConcreteModel()
    m_upper.x = Set(initialize=x.keys())
    m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
    x_vars = [m_upper.x_v[i] for i in x.keys()]
    m_upper.cons = ConstraintList()

    if len(p_list) > 66:
        m_upper.c = Set(initialize=np.arange(len(con_list)))
        m_upper.pl = Set(initialize=np.arange(len(p_list)))
        m_upper.b = Var(m_upper.c,m_upper.pl,domain=Binary)

        def sum_over_p(m, c): 
            return sum(m.b[c,pi] for pi in m_upper.pl) >= int((1-chance) * len(p_list))

        p_c = 0 
        for c in range(len(con_list)):
            con = con_list[c]
            for i in range(n_p_list):
                p_nominal = [p_list[p_c][key]["val"] for key in p_list[p_c].keys()]
                p_c += 1
                m_upper.cons.add(expr=con(x_vars, p_nominal) * m_upper.b[c,i] <= 0)

        m_upper.bincon = Constraint(m_upper.c, rule=sum_over_p)     
    else:
        for c in range(len(con_list)):
            con = con_list[c]
            for p in range(len(p_list)):
                p_nominal = [p_list[p][key]["val"] for key in p_list[p].keys()]
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


u = 0.000000001
chance = 1
n = 1
solver = 'ipopt'
m = iron_and_steel(u,chance,solver,n)
nom = value(m.obj)
print(nom)

u = 0.02
chance = 0.1
n = 20

solver = 'mindtpy'
m = iron_and_steel(u,chance,solver,n)
bin = value(m.b[:,:])
cl = int(len(bin)/n) 
bin = np.array(bin).reshape((cl,n))
for b in bin:
    print('Chance of violation: ',np.round((1-(sum(b)/len(b)))*100,2),'%')
obj = value(m.obj)
percen_inc = (100*obj/nom)-100
print(percen_inc)

