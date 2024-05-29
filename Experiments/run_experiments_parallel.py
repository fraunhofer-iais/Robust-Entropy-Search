# Copyright (c) 2024 Fraunhofer IAIS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Dorina Weichert, dorina.weichert@iais.fraunhofer.de

import numpy as np
import GPy
from Algorithms.StableOpt.stableopt import AcquisitionStableOpt
from Algorithms.RES.res import AcquisitionRES
from Algorithms.UCB.ucb_min import UpperCB
from Algorithms.MES.mes import MaxValueEntropySearch
from Algorithms.KG.knowledge_gradient import KnowledgeGradient
from Algorithms.EI.expected_improvement import ExpectedImprovement
from Experiments.experiment_utils import load_experiment, write_data, find_optimum, optimize_acquisition

import ray
import copy


def run_experiment_parallel(experiment, algorithm, enter_point=0, stop_point=None, n_samples=1, exploration_weight=2, use_ray=True):  
    """
    Function to run the experiments. 

    Parameters
    ----------
    experiment : string
        Experiment to perform. One of 'branin', 'hartmann_3d', 'eggholder', 'robot_pushing', 'sin_plus_linear', 'synthetic_poly', 'within_model'.
    algorithm : string
        Algorithm to perform the experiment. One of 'stableopt', 'ES', 'RVES', 'MES', 'stableopt', 'EI', 'UCB'.
    enter_point : int, optional
        Initialization from which to start the experiment.
    stop_point : int, optional
        Last initialization to start the experiment. Defaults are set depending on the experiment.
    n_samples : int, optional
        Number of samples for RVES algorithm.
    exploration_weight : int, optional
        Exploration weight for StableOpt algorithm.
    use_ray : bool, optional
        A flag indicating wether to use ray for parallelization of the execution of the experiments.
    """
    
    # initialization
    noise_variance = 0.001
    np.random.seed(42)

    if use_ray:
        ray.init(local_mode=False, ignore_reinit_error=True, num_cpus=16)
    else:
        ray.init(local_mode=True, ignore_reinit_error=True)

    if (experiment == 'synthetic_polynomial') | (experiment == 'hartmann_3d'):
        initializations = 100
    elif (experiment == 'eggholder') | (experiment == 'branin') | (experiment == 'within_model') | (experiment == 'sin_plus_linear'):
        initializations = 50
    elif (experiment == 'robot_pushing'):
        initializations = 30

    if stop_point == None:
        end_of_calculation = initializations
    else:
        end_of_calculation = stop_point

    object_refs = [perform_experiment.remote(initialization, experiment, algorithm, noise_variance, n_samples, exploration_weight, use_ray) for initialization in range(enter_point, end_of_calculation)]

    ray.get(object_refs)

    ray.shutdown()
 
    

@ray.remote
def perform_experiment(initialization, experiment, algorithm, noise_variance, n_samples, exploration_weight, use_ray):
    # get the model and the scalers
    model, input_scaler, func_on_normalized, space, iterations, uncontrollables, objective_fun = load_experiment(experiment, algorithm, initialization)
    model.Gaussian_noise.constrain_fixed(noise_variance)

    # experiment loop
    for iteration in range(iterations):
        # use the acquisition
        if algorithm == 'stableopt':
            acquisition = AcquisitionStableOpt(model, space, uncontrollables, exploration_weight=exploration_weight)    
        elif algorithm == 'RVES':
            acquisition = AcquisitionRES(model=model, space=space, num_samples=n_samples, uncontrollables=uncontrollables, iteration=iteration)
            acquisition._initialize_acq()
        elif algorithm == 'UCB':
            acquisition = UpperCB(model, space)
        elif algorithm == 'MES':
            acquisition = MaxValueEntropySearch(model, space)
        elif algorithm == 'EI':
            acquisition = ExpectedImprovement(model, space)
        elif algorithm == 'KG':
            acquisition = KnowledgeGradient(model, space)

        x_new = optimize_acquisition(acquisition, algorithm, space)

        # find the location of the optimum of the model mean
        opt_location, _ = find_optimum(model=model, uncontrollables=uncontrollables, space=space, use_ray=use_ray, algorithm=algorithm)

        # write data
        write_data(experiment=experiment, algorithm=algorithm, input_scaler=input_scaler, initialization=initialization, iteration=iteration, x_new=x_new, opt_location=opt_location, objective_fun=objective_fun, n_samples=n_samples, exploration_weight=exploration_weight)

        if (experiment == "within_model") | (experiment == 'synthetic_polynomial'):
            model.set_XY(np.append(model.X, x_new, axis=0), 
                            np.append(model.Y, np.atleast_2d(func_on_normalized(x_new) + np.random.randn(1)*noise_variance),
                        axis=0))
        else:
            # model.set_XY fails in some cases
            model = GPy.models.GPRegression(X=np.append(model.X, x_new, axis=0), Y=np.append(model.Y, np.atleast_2d(func_on_normalized(x_new) + np.random.randn(1)*noise_variance),
                        axis=0), kernel=GPy.kern.RBF(model.X.shape[1], ARD=True))
            model.Gaussian_noise.constrain_fixed(noise_variance)

            # update the model for the robot pushing experiment
            if experiment == 'robot_pushing':
                if (iteration % 10 != 0) | (iteration == 0):
                    # no update of hyperparameters
                    pass 
                else:                   
                    try:
                        model1 = copy.deepcopy(model)
                        hmc = GPy.inference.mcmc.HMC(model1, stepsize=0.025)
                        hmc.sample(num_samples=1000) # Burnin
                        hmc.sample(num_samples=1000)
                        model = copy.deepcopy(model1)
                    except:
                        print("new function value:", func_on_normalized(x_new))
                        print('HMC update diverged. Keeping the old parameters.')
            elif (experiment == 'sin_plus_linear') | (experiment == 'hartmann_3d') | (experiment == 'branin') | (experiment == 'eggholder') :
                model.kern.lengthscale.constrain_bounded(1e-5, 10)
                model.kern.variance.constrain_bounded(1e-5, 10)
                model.optimize_restarts(10, verbose=False, robust=True)
    