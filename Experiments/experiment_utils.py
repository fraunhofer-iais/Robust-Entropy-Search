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
import pandas as pd
import GPy
import ray
from scipy.optimize import minimize
import os
from sklearn.preprocessing import MinMaxScaler
import datetime


from Algorithms.RES.util import find_min_max_continuous
from Experiments.robot_pushing.robot_pushing_functions import robot_push_3d, get_goals
from Experiments.within_model.within_model_comp import load_within_model_exp
from Experiments.twoD_polynomial.polynomial_2D import load_synth_poly
from Experiments.sin_plus_linear.sin_plus_linear import load_sin_p_linear
from Experiments.hartmann_3d.hartmann_3d import load_hartmann
from Experiments.oneD_experiments.test_problems import get_test


def load_experiment(experiment, algorithm, initialization):
    execution_path = os.getcwd()

    if (experiment == 'branin') | (experiment == 'eggholder') | (experiment == 'camel'):
        testfunction, uncontrollables, scalers, hyperparameters = get_test(experiment)
        gp_noise, signal_var, lengthscale = hyperparameters

        # for the 1D-problems:
        uncontrollables = uncontrollables.reshape(-1, 1)

        input_scaler, output_scaler = scalers[0], scalers[2]

        space = [{'name': 'controllables', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 1},
        {'name': 'uncontrollables', 'type': 'discrete', 'values': uncontrollables, 'dimensionality': 1}]

        if algorithm == 'stableopt':
            # change the sign (from min max to max min)
            def func_on_normalized(X):
                # assumption: X is scaled
                X_rescaled = input_scaler.inverse_transform(X)
                return -1 * output_scaler.transform(testfunction(X_rescaled).reshape(-1, 1)) + 1
        else:
            def func_on_normalized(X):
                # assumption: X is scaled
                X_rescaled = input_scaler.inverse_transform(X)
                return output_scaler.transform(testfunction(X_rescaled).reshape(-1, 1))

        objective_fun = testfunction

        # create model
        kern = GPy.kern.RBF(input_dim=2, ARD=True, lengthscale=lengthscale, variance=signal_var)

        # initialize the model
        n_init = 5

        df = pd.read_csv(execution_path + '/Experiments/oneD_experiments/' + 'initialization_' + experiment + '.csv')
        X = np.array(df.iloc[(initialization * n_init):(initialization * n_init + n_init), 0:2])
        X_scaled = input_scaler.transform(X)
        y = np.array(df.iloc[(initialization * n_init):(initialization * n_init + n_init), 2])
        y_scaled = output_scaler.transform(y.reshape(-1, 1))
            
        model = GPy.models.GPRegression(X=X_scaled, Y=y_scaled, kernel=kern, normalizer=False, noise_var=gp_noise)

        if experiment == 'eggholder':
            iterations = 80
        else:
            iterations = 20


    elif experiment == 'robot_pushing':
        # model
        # input_scaler
        input_scaler = MinMaxScaler()
        input_scaler.fit(np.array([[-5, -5, 1, -5, -5], [5, 5, 30, 5, 5]]))

        # output_scaler, maximum distance is 5 (d-0), minimum is 5-10*np.sqrt(2)
        # this is for maximization
        output_scaler = MinMaxScaler()
        output_scaler.fit(np.array([[5], [5-np.sqrt(2)]]))
        
        # func_on_normalized
        if algorithm == 'stableopt':
            def func_on_normalized(x):
                x = input_scaler.inverse_transform(x)
                if x.shape[0] == 1:
                    x = x.flatten()
                    return output_scaler.transform(np.atleast_2d(robot_push_3d(x[0], x[1], x[2], x[3], x[4])))
                else:
                    return np.array([output_scaler.transform((robot_push_3d(xx[0], xx[1], xx[2], xx[3], xx[4])).reshape(1, -1)).flatten() for xx in x])
        else:
            def func_on_normalized(x):
                x = input_scaler.inverse_transform(x)
                if x.shape[0] == 1:
                    x = x.flatten()
                    return -1 * output_scaler.transform(np.atleast_2d(robot_push_3d(x[0], x[1], x[2], x[3], x[4]))) + 1
                else:
                    return np.array([-1 * output_scaler.transform((robot_push_3d(xx[0], xx[1], xx[2], xx[3], xx[4])).reshape(1, -1)).flatten() + 1 for xx in x])
        
        def objective_fun(x):
            if x.shape[0] == 1:
                x = x.flatten()
                return np.atleast_2d(robot_push_3d(x[0], x[1], x[2], x[3], x[4]))
            else:
                return np.array([robot_push_3d(xx[0], xx[1], xx[2], xx[3], xx[4]) for xx in x]).reshape(x.shape[0], 1)


        # iterations
        iterations = 101

        # uncontrollables
        num_target_pairs = 30
        gx, gy = get_goals(num_target_pairs)
        uncontrollables = np.row_stack([gx[initialization], gy[initialization]])
        uncontrollables = input_scaler.transform(np.column_stack([np.zeros((2, 3)), uncontrollables]))[:, -2:]

        # space
        space = [{'name': 'controllables', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 3},
        {'name': 'uncontrollables', 'type': 'discrete', 'values': uncontrollables, 'dimensionality': uncontrollables.shape[1]}]

        # initialization
        X = np.concatenate([np.random.rand(2, 3), uncontrollables], axis=1)
        y = func_on_normalized(X)

        # integrate out hyperparameters and set these for the model
        kern1 = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        model = GPy.models.GPRegression(X=X, Y=y, kernel=kern1)
        model.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(2.,10.))
        model.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
        model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))

    elif experiment == 'within_model':
        objective_fun, model = load_within_model_exp(initialization)

        space = [{'name': 'controllables', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 1},
        {'name': 'uncontrollables', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 1}]

        input_scaler = None

        if algorithm == 'stableopt':
            model.set_Y(-1 * model.Y)

        def func_on_normalized(x):
            if algorithm == 'stableopt':
                return -1 * objective_fun(x)
            else:
                return objective_fun(x)

        iterations = 40

        uncontrollables = np.nan
    
    elif experiment == 'synthetic_polynomial':
        # in the synthetic polynomial, we search the max min (unusual change of sign)
        objective_fun, model = load_synth_poly(initialization)

        if algorithm != 'stableopt':
            model.set_Y(-1 * model.Y)

        def func_on_normalized(x):
            if algorithm == 'stableopt':
                return objective_fun(x)
            else:
                return -1*objective_fun(x)

        # iterations
        iterations = 100

        radii = np.linspace(0, 1, 2) #* 0.5 #4
        angles = np.linspace(0, 1, 6) #* np.pi

        xx, yy = np.meshgrid(radii, angles)
        uncontrollables = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

        # space
        space = [{'name': 'controllables', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 2},
        {'name': 'uncontrollables', 'type': 'discrete', 'values': uncontrollables, 'dimensionality': 2}]

        # scalers
        input_scaler = None

    elif experiment == 'sin_plus_linear':
        objective_fun, model, uncontrollables = load_sin_p_linear(initialization)

        if algorithm == 'stableopt':
            model.set_Y(-1 * model.Y)

        def func_on_normalized(x):
            if algorithm == 'stableopt':
                return -1 * objective_fun(x)
            else:
                return objective_fun(x)

        input_scaler = None
        iterations = 30

        space = [{'name': 'controllables', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 1},
        {'name': 'uncontrollables', 'type': 'discrete', 'values': uncontrollables, 'dimensionality': 1}]

    elif experiment == 'hartmann_3d':   
        objective_fun, model, controllables, uncontrollables = load_hartmann(initialization)

        if algorithm == 'stableopt':
            model.set_Y(-1 * model.Y)

        def func_on_normalized(x):
            if algorithm == 'stableopt':
                return -1 * objective_fun(x)
            else:
                return objective_fun(x)

        input_scaler = None
        iterations = 30

        space = [{'name': 'controllables', 'type': 'discrete', 'values': controllables, 'dimensionality': 2},
        {'name': 'uncontrollables', 'type': 'discrete', 'values': uncontrollables, 'dimensionality': 1}]

    return model, input_scaler, func_on_normalized, space, iterations, uncontrollables, objective_fun

def find_optimum(model=None, uncontrollables=np.nan, space=np.nan, use_ray=True, algorithm=None):
    # define the max function
    def max_function(x, uncontrollables, model):
        x = np.concatenate([np.repeat(x.reshape(1, -1), uncontrollables.shape[0], axis=0), uncontrollables], axis=1)
        if algorithm=='stableopt':
            return np.max(-1 * model.predict(x)[0])
        else:
            return np.max(model.predict(x)[0])

    if (space[1]['type'] == 'discrete') & (space[0]['type'] == 'continuous'):
        bounds = [(0, 1) for _ in range(space[0]['dimensionality'])]

        # use algorithm without gradients (useful also for discontinuos problems) with multiple restarts (and ray)
        x_init = np.random.rand(100, space[0]['dimensionality'])

        if use_ray:
            @ray.remote
            def optimization_function(xx, max_function, uncontrollables, model, bounds):
                return minimize(max_function, xx, args=(uncontrollables, model), bounds=bounds)

            opt_results = [optimization_function.remote(xx, max_function, uncontrollables, model, bounds) for xx in x_init]
            optimization_results = ray.get(opt_results)
        else:
            def optimization_function(xx, max_function, uncontrollables, model, bounds):
                return minimize(max_function, xx, args=(uncontrollables, model), bounds=bounds)
                
            optimization_results = [optimization_function(xx, max_function, uncontrollables, model, bounds) for xx in x_init]
        
        f_values = np.array([optimization_result.fun for optimization_result in optimization_results])
        x_values = np.array([optimization_result.x for optimization_result in optimization_results])

        # this seems doubled work, but we don't know the uncontrollable
        controllable_variables_opt = x_values[np.argmin(f_values)]
        x_pred = np.concatenate([np.repeat(controllable_variables_opt.reshape(1, -1), uncontrollables.shape[0], axis=0), uncontrollables], axis=1)
        means = model.predict(x_pred)[0]

        return x_pred[np.argmax(means)], np.max(means)
    
    elif (space[1]['type'] == 'discrete') & (space[0]['type'] == 'discrete'):
        xx, yy = np.meshgrid(np.arange(space[0]['values'].shape[0]), np.arange(space[1]['values'].shape[0]))

        x1 = xx.flatten()
        y1 = yy.flatten()
        x_calc = np.hstack((space[0]['values'][x1], space[1]['values'][y1]))

        f_values = model.predict(x_calc)[0]

        # reshape
        f_values_reshaped = f_values.reshape(xx.shape)

        # find the min max
        max_idx = np.argmax(f_values_reshaped, axis=0)
        max_values = f_values_reshaped[max_idx, np.arange(space[0]['values'].shape[0])]

        min_idx = np.argmin(max_values)
        min_max_value = max_values[min_idx]

        min_max_location = np.hstack((space[0]['values'][[min_idx]], space[1]['values'][[max_idx[min_idx]]]))

        return min_max_location, min_max_value

    elif (space[1]['type'] == 'continuous') & (space[0]['type'] == 'discrete'):
        raise(NotImplementedError('Optimization not implemented for discrete controllables and continuous uncontrollables.'))

    elif (space[1]['type'] == 'continuous') & (space[0]['type'] == 'continuous'):
        def objective(x, *args):
            while type(args) is tuple:
                args = args[0]
            model = args
            if algorithm == 'stableopt':                
                return -1 * model.predict(x)[0].flatten()
            else: 
                return model.predict(x)[0].flatten()
        
        def jac(x, nx, *args):
            while type(args) is tuple:
                args = args[0]
            model = args
            if algorithm=='stableopt':
                deriv = -1*model.predictive_gradients(x)[0][:, :, 0]
                return deriv[:, :nx].flatten(), deriv[:, nx:].flatten()
            else:
                deriv = model.predictive_gradients(x)[0][:, :, 0]
                return deriv[:, :nx].flatten(), deriv[:, nx:].flatten()
            
        min_max_value, min_max_location = find_min_max_continuous(model, objective=objective, jac=jac, num_restarts=10, space=space)
        return min_max_location, min_max_value
 

def write_data(experiment=None, algorithm=None, input_scaler=None, initialization=0, iteration=0, x_new=np.nan, opt_location=np.nan, objective_fun=None, n_samples=None, exploration_weight=2):
    execution_path = os.getcwd()
    # report:
    # the evaluated point and the current min max (location and function value) of the model
    # get the predictions
    opt_location = np.atleast_2d(opt_location)
    if input_scaler:
        new_candidate = input_scaler.inverse_transform(x_new)
        optimum_location = input_scaler.inverse_transform(opt_location)
    else:
        new_candidate = x_new
        optimum_location = opt_location

    opt_value = np.atleast_2d(objective_fun(opt_location))
    candidate_value = np.atleast_2d(objective_fun(x_new))

    # create a numpy array
    results = np.concatenate((np.atleast_2d(iteration), np.atleast_2d(new_candidate), np.atleast_2d(optimum_location), np.atleast_2d(candidate_value), np.atleast_2d(opt_value), np.atleast_2d(initialization)), axis=1)

    # header
    columns = ['i'] + ['x_cand_' + str(i) for i in range(x_new.size)] + ['x_opt_' + str(i) for i in range(x_new.size)] + ['cand_value', 'opt_value', 'init'] 

    df = pd.DataFrame(results,
                    columns=columns)
    df['experiment'] = experiment


    # create the necessary folders
    if algorithm=='stableopt':
        result_path = execution_path + '/Results/' + experiment + '/' + algorithm +'/' + 'exploration_' + str(exploration_weight) + '/'
    else:
        result_path = execution_path + '/Results/' + experiment + '/' + algorithm +'/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)  
    
    if n_samples:
        filename = 'init_' + str(initialization) +'_' + 'n_samples' + str(n_samples) + '.csv'
    else:
        filename = 'init_' + str(initialization) + '_.csv'

    # write the df
    if iteration==0:
        timestamp = str(datetime.datetime.now())
        filename = timestamp + filename
        df.to_csv(result_path + filename, index=False)
    else:
        filelist = os.listdir(result_path)
        if n_samples:
            filename = [file for file in filelist if ('init_' +str(initialization) + '_' + 'n_samples' + str(n_samples) + '.csv') in file][-1]
        else:
            filename = [file for file in filelist if ('init_' +str(initialization) + '_') in file][0]

        # append the df
        df.to_csv(result_path + filename, mode='a', header=False, index=False)          

    # create a printout
    print('iteration ', iteration)
    print('new candidate:', new_candidate)
    print('min max location:', optimum_location)
    print('current min max:', opt_value)


def optimize_acquisition(acquisition, algorithm, space):
    if algorithm == 'stableopt':
        x_new = acquisition._return_new_discrete_point()
    else:    
        if space[0]['type'] == 'continuous':   
            if space[1]['type'] == 'discrete':
                # 20 initial points for optimization
                x_inits = np.atleast_2d(np.random.rand(20, acquisition.space[0]['dimensionality']))
                bounds = [(0, 1) for _ in range(acquisition.space[0]['dimensionality'])]
                uncontrollables = space[1]['values']

                # optimization function
                def optimization_objective(x0, acquisition, uncontrollables):
                    # append the uncontrollable
                    X = np.hstack([np.tile(x0, (uncontrollables.shape[0], 1)), uncontrollables])
                    return np.min(acquisition._compute_acq(X))

                @ray.remote
                def minimize_with_ray(x_init, acquisition, optimization_objective):
                    res = minimize(optimization_objective, x0=x_init, args=(acquisition, uncontrollables), bounds=bounds, method='Nelder-Mead')  
                    return res
                
                futures = [minimize_with_ray.remote(x_init, acquisition, optimization_objective) for x_init in x_inits]
                result_objects = ray.get(futures)

                fx_opt = np.array([res.fun for res in result_objects])
                x_opt = np.array([res.x for res in result_objects])

                # I need to find the correct discrete uncontrollable, optimization was only over the controllables
                x_opt = x_opt[np.argmin(fx_opt)]

                x_opt = np.hstack([np.tile(x_opt, (uncontrollables.shape[0], 1)), uncontrollables])
                fx_opt = acquisition._compute_acq(x_opt)
                x_new = x_opt[np.argmin(fx_opt)] 

            elif space[1]['type'] == 'continuous':
                ndims = acquisition.model.X.shape[1]
                x_inits = np.random.rand(20, ndims)           

                bounds = [(0, 1) for _ in range(ndims)]

                def optimization_objective(x, acquisition):
                    return acquisition._compute_acq(x)
                
                @ray.remote
                def minimize_with_ray(x_init, acquisition, optimization_objective):
                    res = minimize(optimization_objective, x0=x_init, args=(acquisition), bounds=bounds, method='Nelder-Mead')  
                    return res
                
                futures = [minimize_with_ray.remote(x_init, acquisition, optimization_objective) for x_init in x_inits]
                result_objects = ray.get(futures)

                fx_opt = np.array([res.fun for res in result_objects])
                x_opt = np.array([res.x for res in result_objects])
                x_new = x_opt[np.argmin(fx_opt)]  

        elif space[0]['type'] == 'discrete': 
            if space[1]['type'] == 'discrete':

                # else:
                xx, yy = np.meshgrid(np.arange(space[0]['values'].shape[0]), np.arange(space[1]['values'].shape[0]))
                x1 = xx.flatten()
                y1 = yy.flatten()
                x_calc = np.hstack((space[0]['values'][x1], space[1]['values'][y1]))

                # calculate the function values on the grid
                if algorithm == 'RVES':
                    acq_values = acquisition._compute_acq_discrete(x_calc)
                else:
                    acq_values = acquisition._compute_acq(x_calc)

                # select the controllable that maximizes the acquisition function
                # maybe there are multiple optima:
                all_minima = np.flatnonzero(acq_values == acq_values.min())
                random_idx = np.random.choice(all_minima)
                x_new = x_calc[[random_idx]]
                
            elif space[1]['type'] == 'continuous':
                raise(NotImplementedError('optimization is not implemented for discrete controllables and continuous uncontrollables.'))

    return np.atleast_2d(x_new)

