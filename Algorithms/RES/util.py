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

from scipy.optimize import minimize
from GPy.util.linalg import jitchol
import numpy as np
from scipy.spatial import distance
import scipy as sc
import ray
from scipy.stats.mvn import mvnun
import matplotlib.pyplot as plt

def compute_stable_cholesky(K):
    K = 0.5 * (K + K.T)
    L = jitchol(K, maxtries=5)
    return L, True

def find_min_max_continuous(*args, objective=None, jac=None, num_restarts=20, space=None):
    # general version
    nx = space[0]['dimensionality']
    ntheta = space[1]['dimensionality']

    def max_objective(theta, x, *args):
        X = np.atleast_2d(np.append(x, theta))
        return -1*objective(X, args)
        
    def max_val(x, *args):
        #theta_inits = np.random.rand(num_restarts, ntheta)
        theta_inits = np.linspace(0, 1, num_restarts, endpoint=True)
        opt_locations = np.zeros((num_restarts, nx))
        opt_vals = np.zeros((num_restarts, 1))
        for idx, theta_init in enumerate(theta_inits):
            res = minimize(max_objective, x0=theta_init, args=(x, args), bounds=np.repeat(np.array([[0, 1]]), ntheta, axis=0))
            opt_locations[idx] = res.x
            opt_vals[idx] = res.fun

        return -1 * np.min(opt_vals)
    
    @ray.remote
    def minimize_max_val(*args, function=None, x_init=np.nan, bounds=np.nan):
        res = minimize(function, x0=x_init, bounds=bounds, args=args, method='Nelder-Mead')
        return res.x, res.fun
    

    # find the optimal controllable location
    # x_inits = np.random.rand(num_restarts, nx)
    x_inits = np.linspace(0, 1, num_restarts, endpoint=True)
    optimized_values = [minimize_max_val.remote(args, function=max_val, x_init=x_init, bounds=np.repeat(np.array([[0, 1]]), nx, axis=0)) for x_init in x_inits]
    optimized_values = ray.get(optimized_values)
    opt_x_locations = [optimized_values[i][0] for i in range(num_restarts)]
    opt_x_vals = [optimized_values[i][1] for i in range(num_restarts)]

    # find the location of the uncontrollable
    def max_theta(*args, function=None, x=np.nan, max_val=np.nan, ntheta=np.nan):
        opt_val = np.nan
        while ((np.isnan(opt_val)) | (np.abs(opt_val - max_val) > 1e-6)): 
            theta_init = np.random.rand(1, ntheta)
            res = minimize(function, x0=theta_init, args=(x, args), bounds=np.repeat(np.array([[0, 1]]), ntheta, axis=0))
            opt_val = -1 * res.fun

        return res.x

    opt_theta = max_theta(args, function=max_objective, x=opt_x_locations[np.argmin(opt_x_vals)], max_val=np.min(opt_x_vals), ntheta=ntheta)

    min_max_location = np.hstack((opt_x_locations[np.argmin(opt_x_vals)], opt_theta)).reshape(1, -1)
    min_max_value = objective(min_max_location, args)

    # plot_min_max(objective, min_max_location, args)

    return min_max_value, min_max_location

def plot_min_max(objective, min_max_location, *args):
    xx, yy = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
    X_grid = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))

    fun_vals_grid = objective(X_grid, args)
    fun_vals_grid_reshaped = fun_vals_grid.reshape(xx.shape)

    min_max_val_initial = np.min(np.max(fun_vals_grid_reshaped, axis=0))
    min_max_initial = X_grid[fun_vals_grid==min_max_val_initial]

    plt.contourf(xx, yy, fun_vals_grid_reshaped)
    plt.scatter(min_max_initial[:, 0], min_max_initial[:, 1], color='red')
    plt.scatter(min_max_location[:, 0], min_max_location[:, 1], color='white')
    plt.title('red: initial, white: optimized')
    plt.savefig('optimization_result.png')
    plt.close()

def find_min_max_discrete(fun_sample, fun_sample_idx, uncontrollables, num_restarts, space):
    if space[0]['type']=='continuous':
        # generate the bounds
        bounds_x = [(0, 1) for _ in range(space[0]['dimensionality'])]

        # generate the initial values
        x_inits = np.random.rand(num_restarts, len(bounds_x))

        # define the max function
        def max_function(x, uncontrollables, fun_sample, fun_sample_idx):
            x = np.concatenate([np.repeat(x.reshape(1, -1), uncontrollables.shape[0], axis=0), uncontrollables], axis=1)
            
            # find the current argmax
            i = np.argmax(fun_sample(x)[0][:, fun_sample_idx])

            return fun_sample(x[i])[0][:, fun_sample_idx]
            
        optimization_results = [minimize(max_function, xx, args=(uncontrollables, fun_sample, fun_sample_idx), bounds=bounds_x) for xx in x_inits]

        f_values = np.array([optimization_result.fun for optimization_result in optimization_results])

        return np.min(f_values)
    else:
        xx, yy = np.meshgrid(np.arange(space[0]['values'].shape[0]), np.arange(space[1]['values'].shape[0]))

        x1 = xx.flatten()
        y1 = yy.flatten()
        x_calc = np.hstack((space[0]['values'][x1], space[1]['values'][y1]))

                    # find the current argmax
        if space[1]['type'] == 'discrete':
            f_values = fun_sample(x_calc)[:, fun_sample_idx]
        else:
            f_values = fun_sample(x_calc)[0][:, fun_sample_idx]

        # reshape
        f_values_reshaped = f_values.reshape(xx.shape)

        # find the min max
        max_idx = np.argmax(f_values_reshaped, axis=0)
        max_values = f_values_reshaped[max_idx, np.arange(space[0]['values'].shape[0])]

        min_idx = np.argmin(max_values)
        min_max_value = max_values[min_idx]

        # min_max_location = np.hstack((space[0]['values'][[min_idx]], space[1]['values'][[max_idx[min_idx]]]))

        return min_max_value

    


def find_argmax(fun_sample, fun_sample_idx, x, space, uncontrollables=np.nan, mode='initialization', representers=np.nan, representer_worst_cases=np.nan):
    # returns the argmax function for discrete or continuous uncontrollables.

    if space[1]['type']== 'discrete':
        if mode == 'initialization':
            if (np.all(np.isnan(uncontrollables))):
                raise(ValueError('If mode is discrete, finding the argmax requires the uncontrollables. Actually, np.nan is given.'))
            
            worst_cases = np.zeros((x.shape[0], uncontrollables.shape[1]))
            worst_case_vals = np.zeros(x.shape[0])
            for row, xx in enumerate(x):
                xx = np.atleast_2d(xx)
                # add all uncontrollables to the current x
                x_grid = np.hstack((np.repeat(xx[:, :space[0]['dimensionality']], axis=0, repeats=uncontrollables.shape[0]), uncontrollables))
                # reshape predictions on X_grid, such that we find are able to find the maxima
                if space[0]['type'] == 'discrete':
                    fun_sample_val = fun_sample(x_grid)[:, fun_sample_idx]
                else:
                    fun_sample_val = fun_sample(x_grid)[0][:, fun_sample_idx]

                # find the worst case
                idx = np.argmax(fun_sample_val)
                worst_cases[row] = uncontrollables[idx]
                worst_case_vals[row] = fun_sample_val[idx]
            return worst_cases, worst_case_vals
        elif mode == 'optimization':
            # reshape x_calc to a grid
            Nc = int(x.shape[0] / space[1]['values'].shape[0])
            if space[0]['type'] == 'discrete':
                fun_sample_vals = fun_sample(x)[:, fun_sample_idx]
            else:
                fun_sample_vals = fun_sample(x)[0][:, fun_sample_idx]
            
            fun_sample_vals_grid = fun_sample_vals.reshape(space[1]['values'].shape[0], Nc).T
            
            # return worst_cases, worst_case_idx_grid
            worst_case_idx =  np.argmax(fun_sample_vals_grid, axis=1)
            worst_cases = np.tile(space[1]['values'][worst_case_idx], (space[1]['values'].shape[0], 1))
            worst_case_vals = np.tile(np.max(fun_sample_vals_grid, axis=1), (space[1]['values'].shape[0]))
            return worst_cases, worst_case_vals
    else:
        bounds = [(0, 1) for _ in range(space[1]['dimensionality'])]
        x = np.atleast_2d(x)

        def neg_function(theta, fun_sample, fun_sample_idx, x):
            x_eval = np.atleast_2d(np.concatenate([x, theta]))
            result = fun_sample(x_eval)
            return (-1 * result[0][:, fun_sample_idx]).item(), (-1 * result[1][:, len(x):, fun_sample_idx]).squeeze()

        if np.all(np.isnan(representers)):
            worst_cases = np.nan*np.ones((x.shape[0], space[1]['dimensionality']))
            worst_case_vals = np.nan*np.ones((x.shape[0]))
            for idx, xx in enumerate(x):
                theta_inits = np.random.rand(50, space[1]['dimensionality'])
                theta_inits = np.vstack((theta_inits, uncontrollables[idx])) # adds the train uncontrollable to the initials
                # use multistart optimization
                opt_results = [minimize(neg_function, theta_ini, bounds=bounds, jac=True, args=(fun_sample, fun_sample_idx, xx)) for theta_ini in theta_inits]

                f_values = np.array([opt_result.fun for opt_result in opt_results])
                theta_values = np.array([opt_result.x for opt_result in opt_results])
                worst_case_idx = np.argmin(f_values)
                worst_cases[idx] = theta_values[worst_case_idx]            
                worst_case_vals[idx] = -1*f_values[worst_case_idx]  

            representers = x
            representer_worst_cases = worst_cases
        else:
            # choose the uncontrollable of the closest representer in controllable direction
            # calculate the minimum distance in direction of the controllables for each x
            distance_matrix = distance.cdist(x, representers[:, :space[0]['dimensionality']])

            # find the minimum distance for each representer
            representer_index = np.argmin(distance_matrix, axis=1)
            controllable_distance = distance_matrix[np.arange(x.shape[0]), representer_index]
            epsilon_bound = 0.05

            # if x is within the epsilon bound, use the theta value of the representer for initializing the optimization, else use multi-start optimization
            worst_cases = np.nan*np.ones((x.shape[0], space[1]['dimensionality']))
            worst_case_vals = np.nan*np.ones((x.shape[0]))
            for idx, xx in enumerate(x):
                if controllable_distance[idx] < epsilon_bound:
                    # use the value
                    opt_result = minimize(neg_function, representer_worst_cases[representer_index[idx]], bounds=bounds, jac=True, args=(fun_sample, fun_sample_idx, xx))

                    worst_cases[idx] = opt_result.x
                    worst_case_vals[idx] = -1*opt_result.fun
              
                else:
                    theta_inits = np.random.rand(50, space[1]['dimensionality'])
                    theta_inits = np.vstack((theta_inits, uncontrollables[idx])) # adds the train uncontrollable to the initials

                    # use multistart optimization
                    opt_results = [minimize(neg_function, theta_ini, bounds=bounds, jac=True, args=(fun_sample, fun_sample_idx, xx)) for theta_ini in theta_inits]

                    f_values = np.array([opt_result.fun for opt_result in opt_results])
                    theta_values = np.array([opt_result.x for opt_result in opt_results])
                    worst_case_idx = np.argmin(f_values)
                    worst_cases[idx] = theta_values[worst_case_idx]            
                    worst_case_vals[idx] = -1*f_values[worst_case_idx]  

                    # add the point to the representers
                    representers = np.vstack((representers, xx))
                    representer_worst_cases = np.vstack((representer_worst_cases, worst_cases[idx]))

        return worst_cases, worst_case_vals, representers, representer_worst_cases

def Q(x):
    return 1 - sc.stats.norm.cdf(x)

def Z(x):
    return sc.stats.norm.pdf(x)

def calculate_2d_truncated_moments(mu, cov, lb, ub):
    # implementation follows Ang & Chen, 2001: Asymmetric Correlations of Equity Portfolios

    # find the correlation factor
    sigmas = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (np.prod(sigmas))

    mu_standard = np.zeros_like(mu)
    cov_standard = np.eye(2)
    cov_standard[0, 1] = rho
    cov_standard[1, 0] = rho

    lb_standard = (lb.flatten() - mu.flatten()) / sigmas
    ub_standard = (ub.flatten() - mu.flatten()) / sigmas

    h1 = lb_standard[0]
    h2 = ub_standard[0]
    k1 = lb_standard[1]
    k2 = ub_standard[1]

    if (np.abs(np.abs(rho)-1) < 1e-6): # high correlation
        lb_hat = np.max(lb_standard)
        ub_hat = np.min(ub_standard)

        mu_new, var_new = sc.stats.truncnorm(a=lb_hat, b=ub_hat).stats('mv')

        var_new = np.clip(var_new, 1e-10, np.infty)

        mu_new_unscaled = (mu_new * sigmas + mu.flatten()).reshape(mu.shape)
        
        cov_truncated_unscaled = np.eye(2)
        for i in range(2):
            for j in range(2):
                cov_truncated_unscaled[i, j] = var_new * sigmas[i] * sigmas[j]
        
        return mu_new_unscaled, cov_truncated_unscaled
    
    else:
        def L(h1, h2, k1, k2):
            return mvnun(lower=np.array([h1, k1]), upper=np.array([h2, k2]), means=mu_standard, covar=cov_standard)[0]
        
        def phi(x):
            return sc.stats.norm.pdf(x)

        def Phi(x):
            return sc.stats.norm.cdf(x)
        
        def psi(h1, h2, k1, k2, rho):
            denominator = np.sqrt(1 - rho**2)
            if rho != 0:
                return phi(h1) * (Phi((k2 - rho*h1)/denominator) - Phi((k1 - rho*h1)/denominator)) - phi(h2) * (Phi((k2 - rho*h2)/denominator) - Phi((k1 - rho*h2)/denominator))
            else:
                return phi(h1) * (Phi(k2) - Phi(k1)) - phi(h2) * (Phi(k2) - Phi(k1))

        def chi(k1, k2, h1, rho):
            denominator = np.sqrt(1 - rho**2)
            if rho != 0:
                return h1 * phi(h1) * (Phi((k2 - rho*h1)/denominator) - Phi((k1 - rho*h1)/denominator)) + rho * denominator / (np.sqrt(2 * np.pi) * (1 + rho**2)) * (phi(np.sqrt(k1**2 - 2*rho*k1*h1 + h1**2)/denominator) - phi(np.sqrt(k2**2 - 2*rho*k2*h1 + h1**2)/denominator))
            else:
                return h1 * phi(h1) * (Phi(k2) - Phi(k1))

        def upsilon(k1, k2, h1, rho):
            denominator = np.sqrt(1 - rho**2)
            if rho != 0:
                return h1 * phi(h1) * (Phi((k2 - rho*h1) / denominator) - Phi((k1 - rho*h1) / denominator))
            else:
                return h1 * phi(h1) * (Phi(k2) - Phi(k1))

        def alpha(k1, k2, h1, rho):
            denominator = np.sqrt(1 - rho**2)
            if rho != 0:
                return denominator / np.sqrt(2 * np.pi) * (phi(np.sqrt(k1**2 - 2*rho*k1*h1 + h1**2) / denominator) - phi(np.sqrt(k2**2 - 2*rho*k2*h1 + h1**2) / denominator))
            else: 
                return 1 / np.sqrt(2 * np.pi) * (phi(np.sqrt(k1**2 + h1**2)) - phi(np.sqrt(k2**2 + h1**2)))

        L_val = L(h1, h2, k1, k2)

        def first_moment(h1, h2, k1, k2, rho, L_val):
            return (psi(h1, h2, k1, k2, rho) + rho * psi(k1, k2, h1, h2, rho)) / L_val

        def second_moment(h1, h2, k1, k2, rho, L_val):
            return (L_val + chi(k1, k2, h1, rho) - chi(k1, k2, h2, rho) + rho**2 * chi(h1, h2, k1, rho) - rho**2 * chi(h1, h2, k2, rho)) / L_val

        m10 = first_moment(h1, h2, k1, k2, rho, L_val)
        m01 = first_moment(k1, k2, h1, h2, rho, L_val)

        if h1 == -np.infty:
            h1 = -1e9

        m20 = second_moment(h1, h2, k1, k2, rho, L_val)
        m02 = second_moment(k1, k2, h1, h2, rho, L_val)

        if rho != 0:
            m11 = (rho * L_val + rho * upsilon(h1, h2, k1, rho) - rho * upsilon(h1, h2, k2, rho) + rho * upsilon(k1, k2, h1, rho) - rho * upsilon(k1, k2, h2, rho) + alpha(h1, h2, k1, rho) - alpha(h1, h2, k2, rho)) / L_val
        else:
            m11 = (alpha(h1, h2, k1, rho) - alpha(h1, h2, k2, rho)) / L_val

        m_truncated = np.hstack((m10, m01))

        cov_truncated = np.eye(2)
        cov_truncated[0, 0] = m20 - m10**2
        cov_truncated[1, 1] = m02 - m01**2
        cov_truncated[0, 1] = m11 - m01 * m10
        cov_truncated[1, 0] = cov_truncated[0, 1]

        m_truncated_unscaled = (m_truncated * sigmas + mu.flatten()).reshape(mu.shape)
        cov_truncated_unscaled = np.copy(cov_truncated)
        for i in range(2):
            for j in range(2):
                cov_truncated_unscaled[i, j] = cov_truncated[i, j] * sigmas[i] * sigmas[j]

        return m_truncated_unscaled, cov_truncated_unscaled