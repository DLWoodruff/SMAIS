# Include our implementation of basic MC method and the MCMC-IS method by Parpas. el
# This is for testing purpose for our paper; if you want to use MCMC-IS, see https://www.doc.ic.ac.uk/~pp500/mcmcSampling.html

# Download git, then use 
# pip install -e .
# To install mpi-sppy and boot-sp
# Whenever it says import statdist: import bootsp.statdist

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
from smt.surrogate_models import KRG
from bootsp import boot_sp
from sklearn.neighbors import KernelDensity
import os

import time
from scipy import stats
import scipy.integrate as integrate

import multiprocessing_on_dill
from multiprocessing_on_dill import Pool, Manager, Value, Lock


num_cpus =  int(os.getenv('SLURM_NTASKS', 1))
num_processes = 2

class check_probability:
    """ 
    Evaluate the integral of pdf `p` using regular Monte Carlo method. 
    This provides a guideline for choosing a proper `K` for sm-ais method: K should be large enough so that the integral of `p` is close to 1.

    Input:
        module (Module): example instance containing the functions `g_calc` and `p_calc` for calculating values used in probability computation.
        xhat (Any): candidate solution of interest.
        d_limits (List[Tuple[float, float]]): A list of tuples defining the lower and upper limits 
                                              of the domain in each dimension.
        cfg (Any): A configuration object containing parameters used in calculations
        base_model (Any): base pyomo model for the example.


    """
    def __init__(self, module,  xhat, d_limits, cfg, base_model):
        self.g_calc = module.g_calc
        self.p_calc = module.p_calc
        self.xhat = xhat 
        self.d_limits = np.array(d_limits)
        self.d_dim = len(d_limits)
        self.cfg=cfg
        self.base_model = base_model

        self.l_bounds = [ll[0] for ll in self.d_limits ]
        self.u_bounds = [ll[1] for ll in self.d_limits ]
        print(f"{self.l_bounds=}")
        print(f"{self.u_bounds=}")

        # compute the volume of the box that contains the random vector
        self.total_area = 1
        for l,u in self.d_limits:
            self.total_area *= u-l
        print(f"{self.total_area=}")

        self.sobol_sampler = scipy.stats.qmc.Sobol(d=self.d_dim, scramble=True, seed=cfg.seed_offset) 
        # self.LHS_sampler = scipy.stats.qmc.LatinHypercube(d=self.d_dim, scramble=True, seed=cfg.seed_offset) 
     

    def main(self):
        # use regular MC for integral and report integral periodically
        mm = 1000
        samples = self.sobol_sampler.random(mm)
        # samples = self.LHS_sampler.random(mm)
        samples = scipy.stats.qmc.scale(samples, self.l_bounds, self.u_bounds)
        gp = []
        integrals = []
        for i in range(mm):
            gp.append(self.p_calc(samples[i], cfg))
            integral = np.mean(gp) * self.total_area
            if i % 20 == 0:
                integrals.append(integral)
                print(f"iter:{i}, integral:{integral}")

class MC_sample_sobel_integral:
    """
    Estimate the function value by integrating g(x)*p(x) across the domain
    Integration is done by MC, sample using sobel sequence
    """
    def __init__(self, module,  xhat, d_limits, cfg, base_model):
        self.g_calc = module.g_calc
        self.p_calc = module.p_calc
        self.xhat = xhat 
        self.d_limits = np.array(d_limits)
        self.d_dim = len(d_limits)
        self.cfg=cfg
        self.base_model = base_model

        self.l_bounds = [ll[0] for ll in self.d_limits ]
        self.u_bounds = [ll[1] for ll in self.d_limits ]
        print(f"{self.l_bounds=}")
        print(f"{self.u_bounds=}")

        # compute the volume of the box that contains the random vector
        self.total_area = 1
        for l,u in self.d_limits:
            self.total_area *= u-l

        self.sobol_sampler = scipy.stats.qmc.Sobol(d=self.d_dim, scramble=True, seed=cfg.seed_offset) 

    def main(self):
        mm = 10000
        samples = self.sobol_sampler.random(mm)
        samples = scipy.stats.qmc.scale(samples, self.l_bounds, self.u_bounds)

        gp = []
        integrals = []
        for i in range(mm):
            gp.append(self.g_calc(samples[i], self.base_model, cfg, self.xhat) * self.p_calc(samples[i], cfg))
            integral = np.mean(gp) * self.total_area
            if i % 20 == 0:
                integrals.append(integral)
                print(f"{i} points, integral:{integral}")


class MC_sample_p_integral:
    """
    Estimates the function value by MC sampling according to the pdf `p`. Sampling from `p` is done by rejection sampling using sobol sequence

    Attributes:
        module (Module): example instance 
        xhat (Any): candidate solution of interest
        d_limits (List[Tuple[float, float]]): Domain limits for each dimension.
        cfg (Any): Configuration object.
        base_model (Any): base pyomo model for the example
    """


    def __init__(self, module,  xhat, d_limits, cfg, base_model):
        self.g_calc = module.g_calc
        self.p_calc = module.p_calc
        self.xhat = xhat 
        self.d_limits = np.array(d_limits)
        self.d_dim = len(d_limits)
        self.cfg=cfg
        self.base_model = base_model

        self.l_bounds = [ll[0] for ll in self.d_limits ]
        self.u_bounds = [ll[1] for ll in self.d_limits ]
        print(f"{self.l_bounds=}")
        print(f"{self.u_bounds=}")

        # compute the volume of the box that contains the random vector
        self.total_area = 1
        for l,u in self.d_limits:
            self.total_area *= u-l
        print(f"{self.total_area=}")

        self.sobol_sampler = scipy.stats.qmc.Sobol(d=self.d_dim, scramble=True, seed=cfg.seed_offset) 

    def main(self):
        # abuse of parameter: use sp_integral_size here as the total number of samples to try, some of them may be rejected 
        # sample according to `p` to estimate the integral
        mm = self.cfg.sp_integral_size
        samples = self.sobol_sampler.random(mm)
        samples = scipy.stats.qmc.scale(samples, self.l_bounds, self.u_bounds)
        p_list = self.p_calc(samples, self.cfg)

        p_area = np.mean(p_list) * self.total_area
        scaling = p_area / np.max(p_list)
        print(f"{p_area=}")
        print(f"{scaling=}")

        integrals = []
        ci_lower = []
        ci_upper = []
        times = []
        g_list = []
        list_sizes = []  # List to keep track of the sizes of g_list
        start_time = time.time()
        last_logged_time = start_time

    
        for i in range(mm):
            pp = self.p_calc(samples[i], self.cfg)
            u = np.random.uniform(0, 1)
            if u <= abs(pp) * scaling:
                g_list.append(self.g_calc(samples[i], self.base_model, self.cfg, self.xhat))
                current_time = time.time()
                time_elapsed = current_time - last_logged_time

                # periodically log the current estimate
                if time_elapsed >= self.cfg.log_frequency:
                    integral = np.mean(g_list)
                    s_dev = np.std(g_list, ddof=1)
                    t_critical = stats.t.ppf(q = 1-0.025, df=len(g_list)-1)
                    std_err = s_dev/np.sqrt(len(g_list))
                    ci_low = integral - t_critical * std_err  # Lower bound of confidence interval
                    ci_up = integral + t_critical * std_err  # Upper bound of confidence interval

                    ci_lower.append(ci_low)
                    ci_upper.append(ci_up)
                    integrals.append(integral)
                    times.append(current_time - start_time)
                    list_sizes.append(len(g_list))  # Keep track of the size of g_list at this time
                    last_logged_time = current_time  # Update last logged time

                    print(f"{len(g_list)} points, integral:{integral}, confidence interval:[{ci_low}, {ci_up}], time elapsed: {time_elapsed:.2f}s")
        print(f"Final Estimation, integral:{integral}, confidence interval:[{ci_low}, {ci_up}],  with {len(g_list)} points, time elapsed: {time_elapsed:.2f}s")

        time_elapsed = time.time() - start_time
        print(f'{time_elapsed=}')
        return times, integrals, ci_lower, ci_upper, list_sizes


class MC_p_integral_parallel:
    """
    a parallel implementation of the class MC_sample_p_integral
    """
    def __init__(self, module,  xhat, d_limits, cfg, base_model):
        self.g_calc = module.g_calc
        self.p_calc = module.p_calc
        self.xhat = xhat 
        self.d_limits = np.array(d_limits)
        self.d_dim = len(d_limits)
        self.cfg=cfg
        self.base_model = base_model

        self.l_bounds = [ll[0] for ll in self.d_limits ]
        self.u_bounds = [ll[1] for ll in self.d_limits ]
        print(f"{self.l_bounds=}")
        print(f"{self.u_bounds=}")

        # compute the volume of the box that contains the random vector
        self.total_area = 1
        for l,u in self.d_limits:
            self.total_area *= u-l
        print(f"{self.total_area=}")

        self.sobol_sampler = scipy.stats.qmc.Sobol(d=self.d_dim, scramble=True, seed=cfg.seed_offset) 


    def worker(self, sample_batch, scaling, worker_id):
        results = []
        np.random.seed(os.getpid() + worker_id) 
        for sample in sample_batch:
            try:        
                pp = self.p_calc(sample, self.cfg)
                u = np.random.uniform(0, 1)
                # print(f'{u=}')
                if u <= abs(pp) * scaling:
                    result = self.g_calc(sample, self.base_model, cfg, self.xhat)
                    results.append(result)
            except Exception as e:
                print(f"Worker error: {e}")
                raise  # Re-raise the exception to catch it in the main process
        return results

    def main(self):
        mm = 100000
        batch_size = 100
        # cfg.display()
        # qtui()

        samples = self.sobol_sampler.random(mm)
        samples = scipy.stats.qmc.scale(samples, self.l_bounds, self.u_bounds)
        p_list = self.p_calc(samples, cfg)
        p_area = np.mean(p_list) * self.total_area
        scaling = p_area / np.max(p_list)
        print(f"{p_area=}")
        print(f"{scaling=}")

        integrals = []
        ci_lower = []
        ci_upper = []
        times = []
        list_sizes = []  # List to keep track of the sizes of g_list
        start_time = time.time()
        last_logged_time = start_time

        sample_batches = np.array_split(samples, np.ceil(mm / batch_size))


        with Manager() as manager:
            print('Opened a manager')
            all_g_list = manager.list()

            # task_counter = Value('i', 0)
            # counter_lock = Lock()


            def append_if_not_none(results):
                if results:
                    all_g_list.extend(results)

            print("Submitting tasks to pool")
            with Pool(processes=num_processes) as pool:
                # result_objects = [pool.apply_async(self.worker, args=(sample, scaling, worker_id), callback=all_g_list.append) for worker_id, sample in enumerate(samples)]
                result_objects = [pool.apply_async(self.worker, args=(sample, scaling, worker_id), callback=append_if_not_none) for worker_id, sample in enumerate(sample_batches)]
                pool.close()

                # Start a loop to print the average every 1 second
                while True:
                    # Check if all tasks are completed
                    if all(result_object.ready() for result_object in result_objects):
                        break
                    # with counter_lock:
                    #     if task_counter.value >= mm:
                    #         break
                    current_time = time.time()
                    time_elapsed = current_time - last_logged_time

                    if time_elapsed >= 3.0:
                        g_list = all_g_list[:]
                        integral = np.mean(g_list)
                        s_dev = np.std(g_list, ddof=1)
                        t_critical = stats.t.ppf(q = 1-0.025, df=len(g_list)-1)
                        std_err = s_dev/np.sqrt(len(g_list))
                        ci_low = integral - t_critical * std_err  # Lower bound of confidence interval
                        ci_up = integral + t_critical * std_err  # Upper bound of confidence interval

                        ci_lower.append(ci_low)
                        ci_upper.append(ci_up)
                        integrals.append(integral)
                        times.append(current_time - start_time)
                        list_sizes.append(len(g_list))  # Keep track of the size of g_list at this time
                        last_logged_time = current_time  # Update last logged time

                        print(f"{len(g_list)} points, integral:{integral}, confidence interval:[{ci_low}, {ci_up}], time elapsed: {time_elapsed:.2f}s")

                    time.sleep(0.3)  # Wait for 1 second before printing the next average

                pool.join()

                # Check for exceptions
                for result_object in result_objects:
                    try:
                        result_object.get()  # This will re-raise any exception caught in the worker process
                    except Exception as e:
                        print(f"Error from worker process: {e}")

            print("Tasks submitted and completed")
            current_time = time.time()
            time_elapsed = current_time - start_time
            print(f'{time_elapsed=}')
        return times, integrals, ci_lower, ci_upper, list_sizes


class MC_Parpas:
    # Note: KDE library in python does not take vector bandwidth for multivariate variable
    # thus not competable with original matlab code provided by Parpas
    # Reflective Boundary option
    # fixed step size MCMC, or adaptive MCMC
    def __init__(self, module,  xhat, d_limits, cfg, base_model):
        self.g_calc = module.g_calc
        self.p_calc = module.p_calc
        self.xhat = xhat 
        self.d_limits = np.array(d_limits)
        self.d_dim = len(d_limits)
        self.cfg=cfg
        self.base_model = base_model

        self.l_bounds = self.d_limits[:, 0]
        self.u_bounds = self.d_limits[:, 1] 
        print(f"{self.l_bounds=}")
        print(f"{self.u_bounds=}")

        # compute the volume of the box that contains the random vector
        self.total_area = 1
        for l,u in self.d_limits:
            self.total_area *= u-l
        print(f"{self.total_area=}")

        self.sobol_sampler = scipy.stats.qmc.Sobol(d=self.d_dim, scramble=True, seed=cfg.seed_offset) 

    def q_calc(self,d):
        # not scaled properly
        return abs(self.g_calc(d, self.base_model, self.cfg, self.xhat)) *self.p_calc(d, self.cfg)
    
    def M_Hasting_propose(self, current):
   
        proposal = current + np.random.randn(self.d_dim) * self.sigmas
        # Reflective boundaries
        # for i in range(self.d_dim):
        #     while proposal[i] < self.l_bounds[i] or proposal[i] > self.u_bounds[i]:
        #         if proposal[i] < self.l_bounds[i]:
        #             proposal[i] = self.l_bounds[i] + (self.l_bounds[i] - proposal[i])
        #         if proposal[i] > self.u_bounds[i]:
        #             proposal[i] = self.u_bounds[i] - (proposal[i] - self.u_bounds[i])
        return proposal

    def M_Hasting(self, x0, mm):
        sigma_factor = 0.35
        self.sigmas = [(ul - ll) * sigma_factor for ll, ul in self.d_limits] 
        current = x0
        current_q = self.q_calc(current)
        samples = [current]
        accepted = 0
        for _ in range(mm - 1):
            proposal = self.M_Hasting_propose(current)
            proposal_q = self.q_calc(proposal)
            acceptance_ratio = min(1, proposal_q / current_q)
            if np.random.rand() < acceptance_ratio:
                # print("accepted")
                accepted += 1
                current = proposal
                current_q = proposal_q
            samples.append(current)
        print(f"HASTING--------------------- ACCEPTANCE RATE with sigma factor {sigma_factor}: {accepted/mm}")

        return np.array(samples)
    

    def adaptive_metropolis(self, x0, mm_hasting, adapt_interval=10, scale=0.1):
        scale = 2.4*2.4/self.d_dim
        samples = [x0]
        current = x0
        current_pdf = self.q_calc(x0)
        
        sigmas = np.array([(ul - ll) * 0.1 for ll, ul in self.d_limits])
        cov = np.diag(sigmas**2)

        # Sampling loop
        for i in range(1, mm_hasting):
            # Propose a new state
            proposal = np.random.multivariate_normal(samples[i-1], cov)
            for i in range(self.d_dim):
                while proposal[i] < self.l_bounds[i] or proposal[i] > self.u_bounds[i]:
                    if proposal[i] < self.l_bounds[i]:
                        proposal[i] = self.l_bounds[i] + (self.l_bounds[i] - proposal[i])
                    if proposal[i] > self.u_bounds[i]:
                        proposal[i] = self.u_bounds[i] - (proposal[i] - self.u_bounds[i])
            proposal_pdf = self.q_calc(proposal)
            
            # Acceptance probability
            accept_prob = min(1, proposal_pdf / current_pdf)
            
            # Accept or reject the proposal
            if np.random.rand() < accept_prob:
                samples.append(proposal)
                current = proposal
                current_pdf = proposal_pdf
            else:
                samples.append(current)
            
            # Adapt the proposal distribution
            if (i + 1) % adapt_interval == 0:
                # past_samples = samples[:i+1]
                cov = scale * np.cov(samples.T) + np.eye(self.d_dim) * scale* 0.1  # Add a small value to ensure positive definiteness

        return np.array(samples)

    
    def is_within_bounds(self,d):
        for i in range(len(d)):
            if d[i]< self.l_bounds[i] or d[i] > self.u_bounds[i]:
                return False
        return True


    def main(self):
        start_time = time.time()
        last_logged_time = start_time
        # x0 = np.array([np.random.uniform(ll, ul) for ll, ul in self.d_limits])
        x0 = (self.l_bounds + self.u_bounds) / 2
        mm_hasting = 1000
        MH_samples = self.M_Hasting(x0, mm_hasting)
        # MH_samples = self.adaptive_metropolis(x0, mm_hasting)
   
        q_kde =  KernelDensity(kernel='gaussian').fit(MH_samples)
 
        ci_lower = []
        ci_upper = []
        times = []
        integrals = []
        integrand = []
        list_sizes = []

        mm = 1000
        for i in range(mm):
            sample = q_kde.sample() 
            q = q_kde.score_samples(sample)    
            sample, q = sample[0], np.exp(q[0])
     
            if not self.is_within_bounds(sample):
                integrand.append(0)
            else:
                integrand.append(self.g_calc(sample, self.base_model, self.cfg, self.xhat) * self.p_calc(sample, self.cfg) / q )
            current_time = time.time()
            time_elapsed = current_time - last_logged_time
            if (i>10) and (time_elapsed >= self.cfg.log_frequency) :
                integral = np.mean(integrand)
                s_dev = np.std(integrand, ddof=1)
                t_critical = stats.t.ppf(q = 1-0.025, df=len(integrand)-1)
                std_err = s_dev/np.sqrt(len(integrand))
                ci_low = integral - t_critical * std_err  # Lower bound of confidence interval
                ci_up = integral + t_critical * std_err  # Upper bound of confidence interval

                ci_lower.append(ci_low)
                ci_upper.append(ci_up)
                integrals.append(integral)
                times.append(current_time - start_time)
                list_sizes.append(len(integrand))  # Keep track of the size of g_list at this time
                last_logged_time = current_time  # Update last logged time

                print(f"{len(integrand)} points, integral:{integral}, confidence interval:[{ci_low}, {ci_up}], time elapsed: {time_elapsed:.2f}s")
        
        print(f"Final Estimation, integral:{integral}, confidence interval:[{ci_low}, {ci_up}],  with {len(integrand)} points")

        return times, integrals, ci_lower, ci_upper, list_sizes


def mpisppy_zhat(cfg, module, xhat):
    max_count = 5000
    z_hat = boot_sp.evaluate_scenarios(cfg, module, range(max_count), xhat, duplication = False)
    print(f"{z_hat=}")
    return z_hat
        
