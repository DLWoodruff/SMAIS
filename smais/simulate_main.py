# surrogate model class
import smt
import scipy
# from scipy.stats import qmc
import numpy as np
import scipy
import pyomo.environ as pyo
import scipy.optimize as opt
import sys
import mpisppy.confidence_intervals.ciutils as ciutils
import smais.smais_utils as smais_utils
import json
from smt.sampling_methods import LHS
from smt.surrogate_models import  KRG
from bootsp import boot_sp
import scipy.stats as stats
from mpisppy import global_toc
import smais.smais_parallel as smais_parallel
import smais.smais_serial as smais_serial
import smais.is_baselines as is_baselines

from scipy.stats import norm
from scipy.integrate import quad



if __name__ == "__main__":
    if len(sys.argv) !=3:
        print("usage: simualte_main.py json method")
        print("   e.g.  simulate_main.py cvar_is.json Surrogate_Parallel")
    json_fname = sys.argv[1]
    method = sys.argv[2]

    cfg = smais_utils.cfg_from_json(json_fname)

    if "deterministic_data_json" in cfg:
        json_fname = cfg.deterministic_data_json
        try:
            with open(json_fname, "r") as read_file:
                detdata = json.load(read_file)
        except:
            print(f"Could not read the json file: {json_fname}")
            raise
        cfg.add_to_config("detdata",
                        description="determinstic data from json file",
                        domain=dict,
                        default=detdata)

    module = smais_utils.module_name_to_module(cfg.module_name)
    xhat_fname = cfg["xhat_fname"]
    xhat = ciutils.read_xhat(cfg["xhat_fname"])

    base_model =  module.scenario_creator("Scenario0", cfg)

    d_limits = module.find_limits(cfg)
    

    if method == 'MC':
        ss = is_baselines.MC_sample_p_integral(module,  xhat, d_limits, cfg, base_model)
    elif method == 'Parpas':
        ss = is_baselines.MC_Parpas(module,  xhat, d_limits, cfg, base_model)
    elif method == 'Surrogate_Serial':
        ss = smais_serial.SMAIS_serial(module, xhat, d_limits, cfg, base_model)
    elif method == 'Surrogate_Parallel':
        ss =  smais_parallel.SMAIS_parallel(module, xhat, d_limits, cfg, base_model)
    else:
        raise RuntimeError(f"Unknown method: {method}")

    times, integrals, ci_lower, ci_upper, list_sizes = ss.main()

    # df = pd.DataFrame({
    #     'Times': times,
    #     'Integrals': integrals,
    #     'CI Lower': ci_lower,
    #     'CI Upper': ci_upper,
    #     'List Sizes': list_sizes
    # })

    # df.to_csv(f'temp_results/{method}' + cfg.module_name + '.csv', index=False,float_format='%.4f')



