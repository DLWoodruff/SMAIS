# DLW July 2022; CVaR as in Lam, Qian paper
import pyomo.environ as pyo
from mpisppy.utils import config
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import numpy as np
from bootsp.statdist.sampler import Sampler
from scipy.stats import norm
import scipy.stats as stats
from smais.smais_utils import fix_xhat_up_to_stage, solve_fixed_up_to_stage


# Use this example: https://jump.dev/JuMP.jl/dev/tutorials/applications/two_stage_stochastic/ 

# Use this random stream:
sstream = np.random.RandomState(1)
# left = 150
# mode = 200
# right = 250


def make_model(xi, num_scens):

    # Create the concrete model object
    model = pyo.ConcreteModel("NewsVendor")

    model.x = pyo.Var(within=pyo.NonNegativeReals) # first stage, how many to buy
    model.y = pyo.Var(within=pyo.NonNegativeReals) # second stage, how many to sell

    model.obj = pyo.Objective(
        expr= -2 * model.x + 5 * model.y - 0.1 * (model.x-model.y),
        sense=pyo.maximize
    )

    def selling_rule(m):
        return m.y <= m.x
    model.selling_constraint = pyo.Constraint(rule=selling_rule)

    def demand_rule(m):
        return m.y <= xi
    model.demand_constraint = pyo.Constraint(rule=demand_rule)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.obj,
            nonant_list=[model.x],
            scen_model=model,
        )
    ]
    
    #Add the probability of the scenario
    if num_scens is not None :
        model._mpisppy_probability = 1/num_scens
    return model

def data_sampler(record_num, cfg):
    # return a single point from a sample
    # Note: we are syncronizing using the seed
    sstream.seed(record_num + cfg.seed_offset)  
    xi = sstream.triangular(left=150, mode=200, right=250)

    return xi

def scenario_creator(scenario_name, cfg):
    """ Create the NewsVendor examples 
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
       
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex then
    # converted mod 3 into one of the below avg./avg./above avg. scenarios

    scennum   = sputils.extract_num(scenario_name)
    sstream.seed(scennum + cfg.seed_offset)  # allows for 

    xi = sstream.triangular(left=150, mode=200, right=250)
    # print(f"{scenario_name}: {xi}")
    num_scens = cfg.get('num_scens', None)
    # print("inside cvar.py")
    # print(f"{num_scens= }")
    return make_model(xi , num_scens)

def g_calc(d, base_model, cfg, xhat):
    # set the paras to d 
    #change model paras to d

    # note: the demand rule need to take in scalar
    def demand_rule(m):
        return m.y <= d[0]
    base_model.demand_constraint = pyo.Constraint(rule=demand_rule)

    nlens = {"ROOT":len(d)}
    result = solve_fixed_up_to_stage(base_model, 1, xhat, nlens, cfg.solver_name, cfg.solver_options)
    return result
    
def p_calc(d, cfg):

    triangular_dist = stats.triang(c=(200 - 150) / (250 - 150), loc=150, scale=(250 - 150))

    if np.isscalar(d[0]):
        return triangular_dist.pdf(d)[0]
    else:
        return triangular_dist.pdf(d).flatten()

def find_limits(cfg):
    d_limits = [[150, 250]]
    return d_limits

#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to the model
    pass

#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"cfg" : cfg}
    return kwargs

def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
        (this function supports zhat and confidence interval code)
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow random sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    # Since this is a two-stage problem, we don't have to do much.
    sca = scenario_creator_kwargs.copy()
    sca["seed_offset"] = seed
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

if __name__ == "__main__":
    # main program just for developer testing

    solver_name = "cplex"
    cfg = config.Config()
    inparser_adder(cfg)
    num_scens = 1000
    cfg.quick_assign("num_scens", int, num_scens)
    scenario_names = scenario_names_creator(num_scens)
    scenario_creator_kwargs = kw_creator(cfg)
    
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    solver = pyo.SolverFactory(solver_name)
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=True)
    else:
        solver.solve(ef, tee=True, symbolic_solver_labels=True,)

    print(f"EF objective: {pyo.value(ef.EF_Obj)}")
    #sputils.ef_ROOT_nonants_npy_serializer(ef, "lam_cvar_nonants.npy")
    solfile = "foo.out"
    representative_scenario = getattr(ef,ef._ef_scenario_names[0])
    sputils.first_stage_nonant_writer(solfile, 
                                        representative_scenario,
                                        bundling=False)
    print(f"Solution written to {solfile}")
