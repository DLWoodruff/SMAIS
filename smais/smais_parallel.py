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
import scipy.stats as stats
import time
from mpisppy import global_toc
from multiprocessing_on_dill import Pool, Manager, Value, Lock, Process
import os


# num_processes = 1

class Surrogate_Gonly_parallel:
    """
    Maintain a surrogate for g

    input:
        g_calc(fct(xhat, xi)): a function that can compute g
        p_calc(fct(xi)): computes density at xi
        xhat (mipsspy xhat): the x of interest
        d_limits: a list of upper and lower bounds for each dimension of xi

    attributes:
        current_xhat
        training_samples: list of (xi, g_value)

    functions:
        change_xhat: pass for now
        add_training_samples(list of xi): add tuple (xi, g_value)
        create_s_model: create the internal surrogate 
        evaluate_s(xi): evaluate surrogate at xi; relies on internal surrogate
        q(xi): evaluate q function at xi
    """
    def __init__(self,module, xhat, d_limits, base_model, cfg):
        self.g_calc = module.g_calc
        self.p_calc = module.p_calc
        self.xhat = xhat
        self.base_model = base_model
        self.cfg = cfg

        self.training_samples = []
        self.training_max = None

        if d_limits:
            self.d_limits = np.array(d_limits).astype(np.float64)
        else:
            raise RuntimeError("d_limits not given")
        
        self.l_bounds = self.d_limits[:, 0]
        self.u_bounds = self.d_limits[:, 1] 
        print(f"{self.l_bounds=}")
        print(f"{self.u_bounds=}")

        # compute the volume of the box that contains the random vector
        self.total_area = 1
        for l,u in self.d_limits:
            self.total_area *= u-l
        print(f"{self.total_area=}")
        
        self.d_dim = len(d_limits)  

        self.sm_area = -1
        self.sm_max = -1
        self.sobol_sampler = scipy.stats.qmc.Sobol(d=self.d_dim, scramble=True, seed=cfg.seed_offset) 

    def _change_xhat(self):
        pass

    def is_within_bounds(self,d):
        for i in range(len(d)):
            if d[i]< self.l_bounds[i] or d[i] > self.u_bounds[i]:
                return False
        return True

    def add_training_samples(self,d_list):
        if type(d_list[0]) is tuple:
            self.training_samples.extend(d_list)
        else:
            for d in d_list:   
                self.training_samples.append((d, self.g_calc(d, self.base_model, self.cfg, self.xhat))) 

    def predict_s_chunk(self, samples_chunk, sm):
        return self.sm.predict_values(samples_chunk).flatten()

    def parallel_predict_s(self, samples, sm, num_processes):
        """
        Performs parallel predictions on a set of samples using a specified surrogate model sm.

        This method divides the input samples into chunks, each chunk is then processed in parallel. 
        The predictions from each chunk are finally combined into a single array and returned.

        Parameters:
        - samples (np.ndarray): The input samples for which predictions are to be made.
        - sm (object): The surrogate model used for making predictions.
        - num_processes (int): The number of parallel processes to use for predictions.

        Returns:
        - np.ndarray: The combined predictions for all input samples from the surrogate model.
        """
        # Split the samples into chunks
        chunks = np.array_split(samples, num_processes)

        with Pool(processes=num_processes) as pool:
            # Process each chunk in parallel
            results = pool.starmap(self.predict_s_chunk, [(chunk, sm) for chunk in chunks])

        # Combine the results
        return np.concatenate(results)

    def find_integral_max(self):
        """
        Computes and records the expected value and maximum value of  `s*p` using the current surrogate model. This involves performing a Monte Carlo integration over the domain using Sobol sequences.
        The evaluation of the surrogate model s is implemented in parallel
        A sufficiently large sp_integral_size is required for accurate integral
        """ 

        K = self.cfg.sp_integral_size
        samples = self.sobol_sampler.random(K)
        samples = scipy.stats.qmc.scale(samples, self.l_bounds, self.u_bounds)
       
        ss_values = self.parallel_predict_s(samples, self.sm, self.cfg.num_processes)

        p_list = self.p_calc(samples, self.cfg)  
        q_list_scaled = np.multiply(np.absolute(ss_values), p_list)

        # multivariate monte carlo integral, average * area
        self.q_scale = np.mean(q_list_scaled) * self.total_area
        self.rejection_scaling = self.q_scale / max(q_list_scaled)
        print(f"area:{self.q_scale}")
        print(f"{self.rejection_scaling= }")

        global_toc(f"min, max value for the q value after normalization: {min(q_list_scaled)/self.q_scale}, {max(q_list_scaled)/self.q_scale}")
        global_toc('done finding integral max')
        
    def create_s_model(self,surrogate_type):
        """
        Trains a surrogate model based on the given surrogate type, using the training samples stored in the class instance. 
        This surrogate model takes `xi` as input and predicts `g` as output. 
        The density function associated with the surrogate model is s(x)/integral(s*p)

        Parameters:
        - surrogate_type (str): The type of surrogate model to initialize.
        """
        xt = np.array([xs[0] for xs in self.training_samples])
        yt = np.array([xs[1] for xs in self.training_samples])

        kwargs = self.cfg.surrogate_kwargs

        if surrogate_type == 'RBF':
            raise RuntimeError(f"RBF does can not be serialized (pickle or dill) for multi-processing")
            sm = RBF(d0=1, print_training=False)
        elif surrogate_type == 'KRG':
            # sm = KRG(theta0=[1e-2])
            sm = KRG(**kwargs)
        elif surrogate_type == 'RMTB':
            raise RuntimeError(f"RMTB does can not be serialized (pickle or dill) for multi-processing")

            kwargs['xlimits'] = np.array(self.d_limits)
            sm = RMTB(**kwargs)
        else:
            raise RuntimeError(f"surrogate_type {surrogate_type} not supported")
        
        sm.set_training_values(xt, yt)
        sm.options['print_prediction'] = False
        sm.train()
        self.sm = sm
        self.find_integral_max()
        
    def q_calc(self,d):
        """
        Compute q(d), for now only support single variable input

        Parameters:
        - random variable d

        Return:
        - q(d)
        """

        if self.is_within_bounds(d):      
            return abs(self.sm.predict_values(np.array(d).reshape(1, -1))[0][0]) * self.p_calc(d, self.cfg) / self.q_scale
        else:
            return 0

    def sample_from_q_worker(self, shared_list, batch_size, m, worker_id):
        """
        A worker function for parallel rejection sampling. Accepted samples are added to a shared list accessible by all worker processes.

        Parameters:
        - shared_list (multiprocessing.Manager().list): A shared list for storing accepted samples across processes.
        - batch_size (int): The number of samples to evaluate in each batch within this worker.
        - m (int): The target number of total samples to generate across all workers.
        - worker_id (int): An identifier for the worker process, used toseed numpy and sobol sampler.
        """
        # Ensure different seed for each process
        np.random.seed(os.getpid() + worker_id)  
        sobol_sampler = scipy.stats.qmc.Sobol(d=self.d_dim, scramble=True, seed=os.getpid() + worker_id)
        
        while True:
            batch_samples = []
            for _ in range(batch_size):
                sample_candidate = sobol_sampler.random()
                sample_candidate = scipy.stats.qmc.scale(sample_candidate, self.l_bounds, self.u_bounds)
                u = np.random.uniform(0, 1)
                q_value = self.q_calc(sample_candidate[0])

                if u <= q_value * self.rejection_scaling:
                    batch_samples.append(sample_candidate[0])
                    if len(batch_samples) == batch_size:
                        break

            shared_list.extend(batch_samples)
            if len(shared_list) >= m:
                break
    
    def sample_from_q_parallel(self, m, batch_size, sample_type=None):
        """
        Generates samples from the distribution q using rejection sampling in parallel, see Daniel Frisch and Uwe Hanebeck. 
        The scaling factor for acceptance is determined by max value of q associated with the surrogate model.
        TBD: sample_type, uniform proposal density vs gaussian proposal density  

        Parameters:
        - m (int): The total number of samples to generate.
        - batch_size (int): The number of samples each worker evaluate per batch.
        - TBD: sample_type (str, optional): The type of proposal density to use for generating samples. 

        Returns:
        - list: A list of samples generated through parallel rejection sampling, approximating the distribution q.

        """
        manager = Manager()
        shared_list = manager.list()

        processes = []

        for worker_id in range(self.cfg.num_processes):
            p = Process(
                target=self.sample_from_q_worker,
                args=(shared_list, batch_size, m, worker_id)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()  # Ensure all processes are properly closed

        return list(shared_list)[:m]
        

class SMAIS_parallel:
    """
    Outer class the implements the algorithm
    input:
        cfg (mpisppy config object): parameters
        g_calc(fct(xhat, d)): a function that can compute g
        p_calc(fct(d)): computes density at d
        xhat (mipsspy xhat): the x of interest

    attributes:  

    """

    def __init__(self, module,  xhat, d_limits, cfg, base_model):
        self.g_calc = module.g_calc
        self.p_calc = module.p_calc
        self.xhat = xhat 
        self.d_limits = np.array(d_limits)
        self.cfg=cfg
        self.base_model = base_model

        self.s_model = Surrogate_Gonly_parallel(module, xhat, d_limits, base_model, cfg)


    def _evaluate_z(self,evaluation_samples):
        """
        Evaluates zhat = mean of (g*p/q) using importance sampling, based on a set of evaluation samples.

        Parameters:
        - evaluation_samples (list or np.ndarray): A collection of samples to evaluate.

        Returns:
        - zhat (float): The estimated function value
        - g_list (np.ndarray): An array of the evaluated 'g' function values for each sample.
        - p_list (np.ndarray): An array of the evaluated 'p' values for each sample, representing the original distribution.
        - q_list (np.ndarray): An array of the evaluated 'q' values for each sample, representing the the importance sampling distribution.
        - zhat_list (np.ndarray): An array of the individual weighted function value estimates for each sample.
        """
        g_list = np.array( [self.g_calc(sample, self.base_model, self.cfg, self.xhat) for sample in evaluation_samples])
        p_list = np.array([self.p_calc(sample, self.cfg) for sample in evaluation_samples])
        q_list = np.array([self.s_model.q_calc(sample) for sample in evaluation_samples])

        zhat_list = np.divide(np.multiply(g_list,p_list), q_list)
        zhat = np.mean(zhat_list)

        return zhat, g_list,  p_list, q_list, zhat_list

    def evaluation_worker(self, shared_list, batch_size, m, worker_id):
        """
        A multiprocessing worker function designed for parallel evaluation of xhat using importance sampling. 
        It uses rejection sampling with a given batch size to sample from q, then add their computed 'z' values to a shared list until a specified number of evaluations is reached.

        Parameters:
        - shared_list (multiprocessing.Manager().list): A multiprocessing-managed shared list to store the 'z' values from all workers.
        - batch_size (int): The number of samples to evaluate in rejection sampling in each batch 
        - m (int): The total number of 'z' values to be generated and evaluated across all workers.
        - worker_id (int): An identifier for the worker, used to ensure different seeds for the random number generation and  the Sobol sequence generator.
        """
        np.random.seed(os.getpid() + worker_id)  
        sobol_sampler = scipy.stats.qmc.Sobol(d=self.s_model.d_dim, scramble=True, seed=os.getpid() + worker_id)
        
        while True:
            batch_values = []
            for _ in range(batch_size):
                sample_candidate = sobol_sampler.random()
                sample_candidate = scipy.stats.qmc.scale(sample_candidate, self.s_model.l_bounds, self.s_model.u_bounds)
                u = np.random.uniform(0, 1)
                q_value = self.s_model.q_calc(sample_candidate[0])

                if u <= q_value * self.s_model.rejection_scaling:
                    z_curr,_,_,_,_ = self._evaluate_z(sample_candidate)
                    batch_values.append(z_curr)
                    if len(batch_values) == batch_size:
                        break

            shared_list.extend(batch_values)
            if len(shared_list) >= m:
                break

    def importance_function_construction(self):
        global_toc("start importance function construction")
        LHS_sampler = LHS(xlimits=self.d_limits)
        d_list = LHS_sampler(self.cfg.initial_sample_size)

        self.s_model.add_training_samples(d_list)

        for iter in range(self.cfg.max_iter):
            global_toc(f"Starting Iteration {iter}")
            self.s_model.create_s_model(self.cfg.surrogate_type)
            global_toc('Created s model')

            evaluation_samples = self.s_model.sample_from_q_parallel(self.cfg.assess_size, self.cfg.assess_batch_size)
            global_toc('Obtained assessment samples from q')
            
            z_curr, g_list, p_list, q_list,zhat_list = self._evaluate_z(evaluation_samples)

            # a confidence interval for the current estimate
            s_dev = np.std(zhat_list, ddof=1)
            t_critical = stats.t.ppf(q = 1-0.025, df=len(zhat_list)-1)
            std_err = s_dev/np.sqrt(len(zhat_list))
            ci_clt = [z_curr - t_critical * std_err, z_curr +  t_critical * std_err]

            global_toc(f" Iteration: {iter}, zhat:{z_curr}, CI: {ci_clt}")

            s_list = self.s_model.sm.predict_values(np.array(evaluation_samples)).flatten()
            # Use the difference between g*p nd s*p
            gtimesp = g_list * p_list
            # print(f'{gtimesp=}')
            abs_diff = np.absolute(gtimesp-s_list * p_list)
            # print(f'{abs_diff=}')

            # Add samples so that: the abs difference is greater  than 5% * max abs value of g*p
            max_gp = np.max(np.abs(gtimesp))
            threshold = max_gp * self.cfg.adaptive_error_threshold_factor
            indices = np.where(abs_diff > threshold)[0]
            global_toc(f"{len(indices)} evaluation samples has error >10% of max value)")
            ranked_sample = [(evaluation_samples[i], g_list[i]) for i in indices]

            # Optional Additional Sample: if the error is large, add to ranked sample
            additional_samples = np.random.rand(self.cfg.additional_sample_size, self.d_limits.shape[0]) * (self.d_limits[:, 1] - self.d_limits[:, 0]) + self.d_limits[:, 0]
            z_curr, g_list, p_list, q_list,zhat_list = self._evaluate_z(additional_samples)
            s_list = self.s_model.sm.predict_values(np.array(additional_samples)).flatten()
            abs_diff = np.absolute(g_list * p_list-s_list * p_list)
            indices = np.where(abs_diff > threshold)[0]
            global_toc(f"{len(indices)} random additional samples has error >10% of max value)")
            ranked_sample = ranked_sample + [(additional_samples[i], g_list[i]) for i in indices]

            # so that, we allow some error, but the error should not change the shape of where is important
            # can also use 10%?
            if ranked_sample:
                global_toc(f"{len(ranked_sample)} were added to the surrogate)")
                self.s_model.add_training_samples(ranked_sample)
            else:
                global_toc("no more sample to add, time for evaluation")
                break

    def main(self):      
        
        
        start_time = time.time()
        last_logged_time = start_time
        self.importance_function_construction()
        global_toc("start evaluation")
 
        ci_lower = []
        ci_upper = []
        times = []
        integrals = []
        integrand = []
        list_sizes = []

        N = self.cfg.evaluation_N
        manager = Manager()
        evaluation_values_all = manager.list()

        processes = []

        for worker_id in range(self.cfg.num_processes):
            p = Process(
                target=self.evaluation_worker,
                args=(evaluation_values_all, self.cfg.eval_batch_size, N, worker_id)
            )
            processes.append(p)
            p.start()

        while any(p.is_alive() for p in processes):
            current_time = time.time()
            time_elapsed = current_time - last_logged_time
            if time_elapsed >= self.cfg.log_frequency:
                integrand = evaluation_values_all[:]
                if len(integrand)>5:

                    integral = np.mean(integrand)
                    s_dev = np.std(integrand, ddof=1)
                    t_critical = stats.t.ppf(q = 1-0.025, df=len(integrand)-1)
                    std_err = s_dev/np.sqrt(len(integrand))
                    ci_low = integral - t_critical * std_err  
                    ci_up = integral + t_critical * std_err  

                    ci_lower.append(ci_low)
                    ci_upper.append(ci_up)
                    integrals.append(integral)
                    times.append(current_time - start_time)
                    list_sizes.append(len(integrand))  # Keep track of the size of g_list at this time
                    last_logged_time = current_time  # Update last logged time

                    global_toc(f"{len(integrand)} points, integral:{integral}, confidence interval:[{ci_low}, {ci_up}]")
            time.sleep(0.3)

        for p in processes:
            p.join()

        # final estimation
        integral = np.mean(integrand)
        s_dev = np.std(integrand, ddof=1)
        t_critical = stats.t.ppf(q = 1-0.025, df=len(integrand)-1)
        std_err = s_dev/np.sqrt(len(integrand))
        ci_low = integral - t_critical * std_err  
        ci_up = integral + t_critical * std_err  

        
        print(f"Running with {self.cfg.num_processes} processes, Final Estimation, integral:{integral}, confidence interval:[{ci_low}, {ci_up}],  with {len(integrand)} points, total runtime: {time.time()-start_time}s")

        return times, integrals, ci_lower, ci_upper, list_sizes

