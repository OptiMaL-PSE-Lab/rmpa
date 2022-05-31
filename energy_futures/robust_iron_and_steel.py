import pandas as pd
from pyomo.environ import (
    log,
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

data = pd.read_csv("data/iron_and_steel_csv.csv")[:33]
plant_names = [i.replace("\xa0", "") for i in data["Plant"].values[1:]]
plant_emissions = [
    float(i.replace(",", "")) for i in data["CO2 Emissions per plant (t/yr)"].values[1:]
]
plants = len(plant_names)

p = {}
p["capture_rate"] = {"val": 0.63, "unc": 0.1}
p["initial_captured"] = {"val": 10000000, "unc": 1000}
p["learning_rate"] = {"val": 0.18, "unc": 0.02}
p["initial_capture_cost"] = {"val": 55.4085, "unc": 1}
for i in range(plants):
    p[plant_names[i] + ": CO2 Emissions"] = {"val": plant_emissions[i], "unc": 500}

x = {}
for plant in plant_names:
    x["t"] = [-1e20, 1e20]
    x[plant + ": Carbon Tax"] = [0, 100]
    x[plant + ": Market Share"] = [0, 1]

con_list = []


def make_c(i):
    def c(x, p):
        CR = p[0]
        UC0 = p[3]
        CO2 = p[4:]
        carbon_tax = x[1 : plants + 1]
        CO2_captured = [CO2[j] * CR for j in range(plants)]
        plant_cost_at_capture_rate = UC0 * CO2[i]
        plant_ctax_impact_on_cost = plant_cost_at_capture_rate - (
            carbon_tax[i] * CO2_captured[i]
        )
        return plant_ctax_impact_on_cost

    return c


for i in range(plants):
    c = make_c(i)
    con_list += [c]


def make_c(i):
    def c(x, p):
        CR = p[0]
        CO2 = p[4:]
        market_share = x[1 + plants :]
        CO2_captured = [CO2[j] * CR for j in range(plants)]
        all_captured = sum(CO2_captured)
        plant_market_occupancy = CO2_captured[i] / all_captured
        plant_improvement = market_share[i] - plant_market_occupancy
        return plant_improvement

    return c


for i in range(plants):
    c = make_c(i)
    con_list += [c]


def con(x, p):
    CR = p[0]
    A0 = p[1]
    LR = -log(1 - p[2]) / log(2)
    UC0 = p[3]
    CO2 = p[4:]
    market_share = x[1 + plants :]
    CO2_captured = [CO2[j] * CR for j in range(plants)]
    all_captured = sum(CO2_captured)
    CO2_captured_end = [market_share[j] * all_captured for j in range(plants)]
    all_CO2_captured = sum(CO2_captured_end)
    A = A0 + all_CO2_captured
    UC = UC0 * (A / A0) ** (-LR)
    return 0.2 - (UC0 - UC) / UC0


con_list += [con]


def epi_con(x, p):
    CR = p[0]
    A0 = p[1]
    LR = -log(1 - p[2]) / log(2)
    UC0 = p[3]
    CO2 = p[4:]
    t = x[0]
    market_share = x[1 + plants :]
    CO2_captured = [CO2[j] * CR for j in range(plants)]
    all_captured = sum(CO2_captured)
    CO2_captured_end = [market_share[j] * all_captured for j in range(plants)]
    all_CO2_captured = sum(CO2_captured_end)
    A = A0 + all_CO2_captured
    UC = UC0 * (A / A0) ** (-LR)
    return t - (UC0 - UC)


con_list += [epi_con]


def obj(x):
    t = x[0]
    return t


problem_count = 0


def var_bounds(m, i):
    return (x[i][0], x[i][1])


def uncertain_bounds(m, i):
    return (p[i]["val"] - p[i]["unc"], p[i]["val"] + p[i]["unc"])


epsilon = 1e-4
solver = "ipopt"

sip_lower_bound = [0]
cons_count = 0
start = time.time()
m_upper = ConcreteModel()
m_upper.x = Set(initialize=x.keys())
m_upper.x_v = Var(m_upper.x, domain=Reals, bounds=var_bounds)
p_nominal = [p[key]["val"] for key in p.keys()]
x_vars = [m_upper.x_v[i] for i in x.keys()]
m_upper.cons = ConstraintList()
for con in con_list:
    m_upper.cons.add(expr=con(x_vars, p_nominal) <= 0)
m_upper.obj = Objective(expr=obj(x_vars), sense=minimize)
SolverFactory(solver).solve(m_upper)
cons_count += len(m_upper.cons)

problem_count += 1
x_opt = value(m_upper.x_v[:])

p_list = [p_nominal]
x_list = [x_opt]
cut_count = 0
robust = True
while True:
    sip_lower_bound.append(value(m_upper.obj))
    robust = True

    v_list = []
    p_list = []
    for con in con_list:
        m_lower = ConcreteModel()
        m_lower.p = Set(initialize=p.keys())
        m_lower.p_v = Var(m_lower.p, domain=Reals, bounds=uncertain_bounds)
        param_vars = [m_lower.p_v[str(i)] for i in p.keys()]
        m_lower.obj = Objective(expr=con(x_opt, param_vars), sense=maximize)
        try:
            SolverFactory(solver).solve(m_lower)
            problem_count += 1
            p_opt = value(m_lower.p_v[:])
            print(value(m_lower.obj))
            p_list.append(p_opt)
            # print("Constraint violation: ", value(m_lower.obj))
            v_list.append(value(m_lower.obj))
            for k in range(len(p_opt)):
                if p_opt[k] is None:
                    p_opt[k] = p_nominal[k]

            # Adding all violations to upper problem
            if value(m_lower.obj) > epsilon:
                robust = False
                m_upper.cons.add(expr=con(x_vars, p_opt) <= 0)
                cut_count += 1

        except ValueError:
            continue

    if robust is True:
        # print("\nRobust solution: ", x_opt)
        break
    SolverFactory(solver).solve(m_upper)
    cons_count += len(m_upper.cons)
    problem_count += 1

    x_opt = value(m_upper.x_v[:])
    x_list.append(x_opt)

end = time.time()
wct = end - start

res = {}
res["wallclock_time"] = wct
res["problems_solved"] = problem_count
res["average_constraints_in_any_problem"] = cons_count / problem_count
res["constraints_added"] = cut_count
res["SplantsP_lower_bound"] = sip_lower_bound[1:]
res["solution"] = x_opt
