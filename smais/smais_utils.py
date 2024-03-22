# utilities for the bootstrap code

import json
import enum
import inspect
import importlib
import mpisppy.utils.config as config
import mpisppy
import pyomo.environ as pyo

def cfg_for_fit_resample(use_MMW=False):
    """ Create and return a Config object for fit_resample

    Returns:
        cfg (Config): the Pyomo config object with fit_resample options added
    """
    cfg = config.Config('CFG')
    # module name gets special parsing
    cfg.add_to_config(name="module_name",
                      description="file name that had scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=False)
    cfg.add_to_config(name="initial_sample_size",
                      description="Sample size for initial training sample",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="assess_size",
                      description="Sample size for evaluating zhat",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="additional_sample_size",
                      description="Sample size for additional random sample in each iter",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="assess_batch_size",
                      description="batch size for parallelized rejection sampling for surrogate assessment",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="eval_batch_size",
                      description="batch size for parallelized rejection sampling for final estimation of zhat",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="sp_integral_size",
                      description="K: Monte Carlo Sample size used to integrate the denominator in the q expression",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="adaptive_error_threshold_factor",
                      description=" c_beta, the threshold factor for choosing assessment samples that have relative large error, between 0 and 1",
                      domain=float,
                      default=0.1)
    cfg.add_to_config(name="evaluation_N",
                      description="number of samples for final estimation of zhat",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="max_iter",
                      description="max iteration allowed in estimation",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="seed_offset",
                      description="For some instances this enables replication.",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="xhat_fname",
                      description="(optional) the name of an npy file with a pre-sored xhat; use 'None' when not present",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="solver_name",
                      description="name of solver (e.g. gurobi_direct)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="solver_options",
                      description="string that is passed directly to the solver (e.g. 'threads=2')",
                      domain=str,
                      default=None)                  
    cfg.add_to_config(name="surrogate_json",
                      description="json file name for parameters for surrogate model for fitting",
                      domain=str,
                      default=None)    
    cfg.add_to_config(name="log_frequency",
                      description="frequency to print out intermediate results during evaluation process, unit in seconds",
                      domain=float,
                      default=None)   
    cfg.add_to_config(name="num_processes",
                      description="number of processes to use for SM-AIS",
                      domain=int,
                      default=None)     
    return cfg
    
def module_name_to_module(module_name):
    if inspect.ismodule(module_name):
        module = module_name
    else:
        module = importlib.import_module(module_name)
    return module

def _process_module(module_name,use_MMW=False):
    # factored code
    module = module_name_to_module(module_name)
    cfg = cfg_for_fit_resample(use_MMW=use_MMW)
    assert hasattr(module, "inparser_adder"), f"The module {module_name} must have the inparser_adder function"
    module.inparser_adder(cfg)
    assert len(cfg) > 0, f"cfg is empty after inparser_adder in {module_name}"    
    return cfg


def cfg_from_json(json_fname):
    """ create a Pyomo config object for the bootstrap code from a json file
    Args:
        json_fname (str): json file name, perhaps with path
    Returns:
        cfg (Config object): populated Config object
    Note:
        Used by the simulation code
    """
    try:
        with open(json_fname, "r") as read_file:
            options = json.load(read_file)
    except:
        print(f"Could not read the json file: {json_fname}")
        raise
    assert "module_name" in options, "The json file must include module_name"
    cfg = _process_module(options["module_name"])

    badtrip = False

    def _dobool(idx):
        if idx not in options:
            badtrip = True
            # such an index will raise two complaints...
            print(f"ERROR: {idx} must be in json {json_fname}")
            return
        if options[idx].lower().capitalize() == "True":
            options[idx] = True
        elif options[idx].lower().capitalize() == "False":
            options[idx] = False
        else:
            badtrip = True
            print(f"ERROR: Needed 'True' or 'False', got {options[idx]} for {idx}")


    # get every cfg index from the json
    for idx in cfg:
        if idx not in options:
            badtrip = True
            print(f"ERROR: {idx} not in the options read from {json_fname}")
            continue
        if options[idx] != "None":
            # TBD: query the cfg to see if it is bool
            if str(options[idx]).lower().capitalize() == "True" or str(options[idx]).lower().capitalize() == "False":
                _dobool(idx)  # do not return options, just modify cfg
            cfg[idx] = options[idx]
        else:
            cfg[idx] = None

    # BootMethods.check_for_it(options["boot_method"])    
    if badtrip:
        raise RuntimeError(f"There were missing options in the json file: {json_fname}")
    else:
        cfg = surrogate_parse(cfg)
        return cfg

def surrogate_parse(cfg):
    json_fname = cfg.surrogate_json
    try:
        with open(json_fname, "r") as read_file:
            options = json.load(read_file)
    except:
        print(f"Could not read the json file: {json_fname}")
        raise
    surrogate_type = options.get('surrogate_type', None)
    assert surrogate_type     # do it better later
    cfg.add_to_config(name="surrogate_type",
                      description="surrogate type name",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="surrogate_kwargs",
                      description="dictionary of kwargs for surrogate",
                      domain=dict,
                      default=None)
    surrogate_kwargs   = options.get('kwargs', None)    

    cfg['surrogate_type'] = surrogate_type
    cfg['surrogate_kwargs'] = surrogate_kwargs
    
    return cfg
    # for idx, val in options.items():
    #     # only works if cfg is set up here for the given surrogate
    #     cfg[idx] = val
   


# def cfg_from_parse(module_name, name=None,use_MMW=False):
#     """ create a Pyomo config object for the bootstrap code from a json file
#     Args:
#         module_name (str): name of module with scenario creator and helpers
#         name (str): name for parser on the command line (e.g. user_boot)
#     Returns:
#         cfg (Config object): Config object populated by parsing the command line
#     """

#     cfg = _process_module(module_name,use_MMW=use_MMW)

#     parser = cfg.create_parser(name)
#     # the module name is very special because it has to be plucked from argv
#     parser.add_argument(
#             "module_name", help="amalgamator compatible module (often read from argv)", type=str,
#         )
#     cfg.module_name = module_name
    
#     args = parser.parse_args()  # from the command line
#     args = cfg.import_argparse(args)
    
#     return cfg


def compute_xhat(cfg, module):
    """  Deal with signatures specified by mpi-sppy to find an xhat (realy local to main_routine)
    Args:
        cfg (Config): paramaters
        module (Python module): contains the scenario creator function and helpers
    Returns:
        xhat (dict): the optimal nonants in a format specified by mpi-sppy
    Note: Basically, the code to solve for xhat must be provided in the module
    """
    xhat_fct_name = f"xhat_generator_{cfg.module_name}"
    if not hasattr(module, xhat_fct_name):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           f"{xhat_fct_name} when xhat-fname is not given")
    if not hasattr(module, "kw_creator"):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           f"kw_creator when xhat-fname is not given")
    if not hasattr(module, "scenario_names_creator"):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           f"scenario_names_creator when xhat-fname is not given")
    #Computing xhat_k

    xhat_scenario_names = module.scenario_names_creator(cfg.candidate_sample_size, start=cfg.sample_size)
    
    xgo = module.kw_creator(cfg)
    xgo.pop("solver_name", None)  # it will be given explicitly
    ###xgo.pop("solver_options", None)  # it will be given explicitly
    xgo.pop("num_scens", None)
    xgo.pop("scenario_names", None)  # given explicitly
    xhat_fct = getattr(module, xhat_fct_name)
    xhat_k = xhat_fct(xhat_scenario_names, solver_name=cfg.solver_name, **xgo)
    return xhat_k


def fix_xhat_up_to_stage(s, t, xhat, nlens,persistent_solver=None):
    # s:  model, t: stage
    # persistent_solver = None
    # if (sputils.is_persistent(s._solver_plugin)):
    #     persistent_solver = s._solver_plugin

    # nlens = s._mpisppy_data.nlens
    for node in s._mpisppy_node_list:
        if node.stage<=t:
            ndn = node.name
            if ndn not in xhat:
                raise RuntimeError("Could not find {} in {}"\
                                    .format(ndn, xhat))
            if xhat[ndn] is None:
                raise RuntimeError("Empty xhat, node={}".format( ndn))
            # if len(xhat[ndn]) != nlens[ndn]:
                raise RuntimeError("Needed {} nonant Vars for {}, got {}"\
                                    .format(nlens[ndn], ndn, len(cache[ndn])))
            for i in range(nlens[ndn]): 
                this_vardata = node.nonant_vardata_list[i]
                this_vardata._value = xhat[ndn][i]
                this_vardata.fix()
                if persistent_solver is not None:
                    persistent_solver.update_var(this_vardata)
                 
def solve_fixed_up_to_stage(base_model, t, xhat, nlens, solver_name, s_options):
    fix_xhat_up_to_stage(base_model, t, xhat, nlens)
    
    opt = pyo.SolverFactory(solver_name)
    results = opt.solve(base_model, options_string = s_options)
    pyo.assert_optimal_termination(results)
    active_obj = mpisppy.utils.sputils.find_active_objective(base_model)
    return pyo.value(active_obj)


if __name__ == "__main__":
    print("smais_utils does not have a main program.")
