# surrogate model class

import pyomo.environ as pyo

import sys
import mpisppy.confidence_intervals.ciutils as ciutils
import smais.smais_utils as smais_utils
import json

import smais.smais_parallel as smais_parallel
import smais.smais_serial as smais_serial
import smais.is_baselines as is_baselines

from io import StringIO
import tempfile
import unittest
import os
import shutil
from mpisppy.tests.utils import get_solver,round_pos_sig

solver_available,solver_name, persistent_available, persistentsolver_name= get_solver()

def set_up_module_xhat(cfg):
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

    return module, xhat, cfg, base_model, d_limits

class TS(unittest.TestCase):
    """ Test the simulate_main code.
        Assumes naming conventions for filenames"""

        
    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.temp_dir.cleanup()

 
    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_dir.name
        sys.path.append(cls.temp_path)

        abs_json_path = os.path.abspath('jsons/newsvendor_is.json')
        abs_krg_path = os.path.abspath('jsons/KRG.json')

        abs_news_py_path = os.path.abspath("../../examples/newsvendor_is.py")
        abs_news_xhat_path = os.path.abspath("../../examples/newsvendor_xhat.npy")

        os.chdir(cls.temp_path)
        print(f"cwd: {os.getcwd()}")

        f = open("__init__.py", "w")
        f.close()
        
        shutil.copy(abs_news_py_path, 'newsvendor_is.py')
        shutil.copy(abs_news_xhat_path, 'newsvendor_xhat.npy')
        shutil.copy(abs_json_path, 'newsvendor_is.json')
        shutil.copy(abs_krg_path, 'KRG.json')
     

        cls.cfg = smais_utils.cfg_from_json('newsvendor_is.json')
        cls.module, cls.xhat, cls.cfg, cls.base_model, cls.d_limits = set_up_module_xhat(cls.cfg)


    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_MC(self):

        MC_output = StringIO()
        sys.stdout = MC_output

        
        ss = is_baselines.MC_sample_p_integral(TS.module,  TS.xhat, TS.d_limits, TS.cfg, TS.base_model)
        times, integrals, ci_lower, ci_upper, list_sizes = ss.main()

        sys.stdout = sys.__stdout__
        self.assertIn('Final Estimation, integral:5', MC_output.getvalue()) # one digits precision

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_Parpas(self):

        MC_output = StringIO()
        sys.stdout = MC_output

        ss = is_baselines.MC_Parpas(TS.module,  TS.xhat, TS.d_limits, TS.cfg, TS.base_model)
        times, integrals, ci_lower, ci_upper, list_sizes = ss.main()

        sys.stdout = sys.__stdout__
        self.assertIn('Final Estimation, integral:5', MC_output.getvalue()) # one digits precision
    
    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_Surrogate_Serial(self):
        MC_output = StringIO()
        sys.stdout = MC_output
        ss = smais_serial.SMAIS_serial(TS.module,  TS.xhat, TS.d_limits, TS.cfg, TS.base_model)
        times, integrals, ci_lower, ci_upper, list_sizes = ss.main()
        sys.stdout = sys.__stdout__
        self.assertIn('Final Estimation, integral:5', MC_output.getvalue()) # one digits precision

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_Surrogate_Parallel(self):
        MC_output = StringIO()
        sys.stdout = MC_output
        ss = smais_parallel.SMAIS_parallel(TS.module,  TS.xhat, TS.d_limits, TS.cfg, TS.base_model)   
        times, integrals, ci_lower, ci_upper, list_sizes = ss.main()
        sys.stdout = sys.__stdout__
        self.assertIn('Final Estimation, integral:5', MC_output.getvalue()) # one digits precision


        

if __name__ == '__main__':
    unittest.main()