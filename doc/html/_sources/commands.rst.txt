.. _commands:

Commands
========

Running the programs
--------------------

The most general way is to use the ``python -m`` terminal command for the program for the desired mode:

.. code-block:: bash

   $ python -m smais.simulate_main filename method

where ``filename`` is the name of a Python such as `farmer.json` that contains a full :ref:`Arguments` set for the simulation, and ``method`` specify which method to run.
Currently the following four method were supported:
    - "MC": Regular MC, use Sobol Sequence to sample from the domain, and use rejection sampling to mimic the process of sampling from the distribtion `p`

    - "Parpas": A MCMC-IS method in [PUWT15]_

    - "Surrogate_Serial": A serial implementation of the smais method

    - "Surrogate_Parallel": A parallel implementation of the smais method, the number of processes is specified in json file

.. _Arguments:

Arguments
---------

The argument values are all given in a json file. In the json format, all string values are quote delimited.

*     ``module_name``, n/a: The name of of the python module that has the scenario creator and help functions given in the json file as a string such as "farmer_is". 
     
*     ``xhat_fname``: When xhat (the estimated, or candidate) solution is computed by another program (which is common and recommended in simulation mode), this argument gives the name of an numpy file that has the solution as string such as "xhat.npy". 

*     ``initial_sample_size``: Sample size for initial training sample. It corresponds to M in the paper and is given as an integer such as 40.  

*     ``assess_size``: Specifies the sample size used to evaluate the quality of intermediate surrogate models. This involves drawing samples based on the probability density function associated with the surrogate model and comparing the values predicted by the surrogate model with the actual function values. The sample size is defined as an integer, for instance, 20.

*     ``additional_sample_size``: Determines the sample size for further assessing the quality of intermediate surrogate models, by drawing uniformly distributed random samples and comparing the surrogate model's predictions with the actual function values. Similar to assess_size, this sample size is also specified as an integer, such as 20.

*     ``sp_integral_size``: Sample size for evaluating the integral of the surrogate model to determine the proper scaling of the importance sampling function. It corresponds to K in the paper and is given as a relatively large integer such as 1000.

*     ``adaptive_error_threshold_factor``: This parameter sets the threshold factor for identifying assessment samples with relatively large errors. If the absolute error between the surrogate model prediction and the actual function value exceeds this factor multiplied by the maximum value of the function, the error is deemed to be relatively large. It should be specified as a floating-point number between 0 and 1. It is denoted as c_beta in the paper.

*     ``evaluation_N``: Max number of samples to draw from the importance sampling function for the final estimation. It corresponds to N in the paper, and is given as an integer.

*     ``max_iter``: Max iteration allowed when adaptively train the surrogate model. It corresponds to N in the paper, and is given as an integer.

*     ``seed_offset``: This option is provided so that modelers who want to enable replication with difference seeds can do so.. It is given as an integer. Unless you have a reason to do otherwise, just use 0.

*     ``solver_name``: The name of the solver to be used given as a string such as "gurobi_direct".

*     ``solver_options```: A string specifying options for the solver, which are passed directly to it. For example, to set the number of threads used by the solver, use `thread=2`

*     ``surrogate_json``: Specifies the filename of a JSON file that contains information about the surrogate model type and its parameters. The value should be provided as a string, for example, "KRG.json". The JSON file must include two key elements:
   - "surrogate_type": A string identifying the type of surrogate model. Supported types currently include 'KRG', 'RBF', and 'RMTB'.

   - "kwargs": A dictionary containing additional arguments specific to the surrogate model type. 

*     ``log_frequency``: Defines the frequency at which intermediate results are printed out during the final evaluation process, with the unit being seconds. This parameter is provided as either a float number or an integer.

*     ``num_precesses``:  Specifies the maximum number of processes to be utilized for parallel computation. If the system has fewer processes available than the specified number, the "Surrogate_Parallel" method will automatically employ all available processes. This option is for "Surrogate_Parallel" only and is ignored for other methods.

*     ``assess_batch_size``: the batch size for each set of parallelized rejection sampling operations during the surrogate assessment phase. It is given as an integer, such as 200. This option is for "Surrogate_Parallel" only and is ignored for other methods.

*     ``eval_batch_size``: Specifies the batch size for each set of parallelized rejection sampling and function evaluation operations conducted during the final evaluation phase.  It is given as an integer, such as 100. This option is for "Surrogate_Parallel" only and is ignored for other methods.



In addition to these arguments, there may be problem-specific arguments (e.g. "crops_multiplier" for
the scalable farmer problem).

Farmer Examples
---------------

cd to ``/examples/farmer_is``.


.. code-block:: bash

   $ python -m smais.simulate_main farmer_is.json Surrogate_Serial
   
