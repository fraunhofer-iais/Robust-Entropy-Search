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

from .util import find_min_max_discrete, find_min_max_continuous, find_argmax,  compute_stable_cholesky, calculate_2d_truncated_moments
from .ssgp import SparseSpectrumGP
from .ep import expectation_propagation_trunc_gauss
import numpy as np
import ray
import scipy as sc
import copy

class AcquisitionRES():
    """
    Robust Entropy Search Acquisition Function.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, num_samples=1, uncontrollables=np.nan, iteration=int):
        np.random.seed(iteration)
        self.model = model
        self.model.normalize_Y = False
        self.space = space
        self.num_samples = num_samples

        if (self.space[0]['type'] == 'discrete') & (self.space[1]['type'] == 'discrete'):
            with_derivative = False
        else:
            with_derivative = True

        self.ssgp_model = SparseSpectrumGP.from_gp(model, 500, with_derivative)
        self.bounds = np.array([[0, 1] for _ in range(self.space[0]['dimensionality']+self.space[1]['dimensionality'])])
        if np.all(np.isnan(uncontrollables)):
            self.uncontrollables = np.array([[np.nan]*self.space[1]['dimensionality']])

        self.initialized_acq = False

        if (np.all(np.isnan(uncontrollables))) & (space[1]['type']=='discrete'):
            raise(ValueError("If the space of uncontrollables is discrete, you have to define them. Currently, they are np.nan."))
        
    def _initialize_acq(self):
        if self.space[1]['type'] == 'discrete':
            self._initialize_acq_discrete()
        else:
            self._initialize_acq_continuous_parallelized()
    
    def _compute_acq(self, x):
        if self.space[1]['type'] == 'discrete':
            return -1*self._compute_acq_discrete(x)
        else:
            return -1*self._compute_acq_continuous_parallelized(x)        
        
    def _initialize_acq_discrete(self):
        """
        Initializes the RES acquisition function
        """

        X = self.model.X
        self.Nu = self.space[1]['values'].shape[0]
        controllables = X[:, 0:self.space[0]['dimensionality']]       
        uncontrollables = self.space[1]['values']

        counter = 0 # there is a low chance (1/1000) to have sample that cannot be used for conditioning
        while (self.initialized_acq == False) and (counter < 5):
            # draw samples from the SSGP - this function returns function and gradient!
            with_derivative = not ((self.space[0]['type']=='discrete') & (self.space[1]['type']=='discrete'))
            self.fun_samples = self._draw_SSGP_samples(with_derivative=with_derivative)

            # for each sample: 
            self.g_stars = []
            self.mu1s = []
            self.cov1s = []
            self.X_grid = []

            for fun_sample_idx in range(self.num_samples):
                num_restarts = 10
                g_star = find_min_max_discrete(self.fun_samples, fun_sample_idx, uncontrollables, num_restarts, self.space)
                self.g_stars.append(g_star)

                # find the worst cases i for the actual train data
                worst_cases, worst_case_vals = find_argmax(self.fun_samples, fun_sample_idx, X, self.space, uncontrollables=uncontrollables)

                # the bounds
                lb = np.ones((2*X.shape[0], 1))
                lb[:X.shape[0]] *= -1e6 
                lb[X.shape[0]:] = g_star
                ub = np.tile(worst_case_vals, 2).reshape(-1, 1)

                mask = np.concatenate((np.all((X[:, self.space[0]['dimensionality']:] != worst_cases), axis=1), np.ones(X.shape[0], dtype=bool)))
                full_mask = (mask) & (np.linalg.norm(np.abs(lb - ub), axis=1) > 1e-6)

                lb = lb[full_mask]
                ub = ub[full_mask]

                X_grid = np.vstack((X, np.hstack((X[:, :self.space[0]['dimensionality']], worst_cases))))[full_mask]
                       
                if np.all(full_mask == False):
                    # resample
                    counter += 1
                    self.mu1s.append(np.nan)
                    break                    

                # create predictions with the original GP
                mu, cov = self.model.predict(Xnew=X_grid, full_cov=True)

                # apply EP
                if mu.shape[0] != 1:
                    mu_cond, cov_cond = expectation_propagation_trunc_gauss(mean=mu, cov=cov, lb=lb, ub=ub, n_max_sweeps=50, abs_tol=1e-6)
                else:
                    scale = np.sqrt(cov[0, 0])

                    # scale the bounds to be n standard deviations
                    lb_scaled = (lb - mu) / scale
                    ub_scaled = (ub - mu) / scale

                    mu_cond, cov_cond = sc.stats.truncnorm(a=lb_scaled, b=ub_scaled, loc=mu, scale=scale).stats('mv')

                mu_cond = mu_cond.reshape(mu.shape)
                self.mu1s.append(mu_cond)
                self.cov1s.append(cov_cond)
                self.X_grid.append(X_grid)

                if np.any(np.isnan(mu_cond)):
                    counter +=1
                    print('Need to resample, as a sample is not valid')
            
            if not np.any(np.isnan(self.mu1s)):
                self.initialized_acq = True   
                    
    def _compute_acq_discrete(self, x):
        """
        Computes the RES acquisition function for a discrete set of uncontrollables
        """
        x = np.atleast_2d(x)
        
        if self.initialized_acq == False:
            self._initialize_acq_discrete()
        
        # loop over the samples
        H5 = 0

        for fun_sample_idx in range(self.num_samples):
            # find the worst-case for the new x
            worst_cases, worst_case_vals = find_argmax(self.fun_samples, fun_sample_idx, x, self.space, uncontrollables=self.space[1]['values'], mode='optimization')

            sigma5 = np.zeros((x.shape[0]))
            for idx, xx in enumerate(x):
                xx = np.atleast_2d(xx)

                x_new = np.vstack((xx, np.hstack((xx[:, :self.space[0]['dimensionality']], worst_cases[[idx]]))))

                # prediction for the current x, taking into account all possible grid values
                K = self.model.kern.K(X=self.X_grid[fun_sample_idx])
                k_xX = self.model.kern.K(X=x_new, X2=self.X_grid[fun_sample_idx])

                # p( g(x) | g, y ) = N( g(x) | mu_g, cov_g )
                K_chol = compute_stable_cholesky(K)[0] 
                s = np.linalg.solve(K_chol.T, np.linalg.solve(K_chol, k_xX.T)) 

                covg = self.model.kern.K(x_new) - s.T @ k_xX.T
                a1 = s.T

                # p0(g(x) | y, g *) = N( g(x) | mu2, v2 )
                mu2 = a1 @ self.mu1s[fun_sample_idx]
                cov2 = covg + a1 @ self.cov1s[fun_sample_idx] @ a1.T

                if (np.linalg.norm(self.g_stars[fun_sample_idx] - worst_case_vals[idx]) < 1e-6):
                    # we're at the worst_case
                    if (np.linalg.norm(x_new[[0]]-x_new[[1]]) < 1e-6):
                        val = 0
                    else:
                        mu = mu2[0]
                        scale = np.sqrt(cov2[0, 0])

                        lb = -np.infty
                        ub = worst_case_vals[idx] 

                        # scale the bounds to be n standard deviations
                        lb_scaled = (lb - mu) / scale
                        ub_scaled = (ub - mu) / scale

                        val = sc.stats.truncnorm(a=lb_scaled, b=ub_scaled, loc=mu, scale=scale).stats('v')
              
                elif (np.linalg.norm(x_new[[0]]-x_new[[1]]) < 1e-6):
                    # we're at the worstcase but it's not g_star
                    mu = mu2[0]
                    scale = np.sqrt(cov2[0, 0])

                    lb = self.g_stars[fun_sample_idx]
                    ub = worst_case_vals[idx]

                    # scale the bounds to be n standard deviations
                    lb_scaled = (lb - mu) / scale
                    ub_scaled = (ub - mu) / scale

                    val = sc.stats.truncnorm(a=lb_scaled, b=ub_scaled, loc=mu, scale=scale).stats('v')
                    # calculate mu and sigma via scipy                  

                else:
                    lb = np.zeros(x_new.shape[0])
                    lb[0] = -np.infty
                    lb[1] = self.g_stars[fun_sample_idx]

                    ub = np.ones(2) * worst_case_vals[idx]

                    _, cov4 = calculate_2d_truncated_moments(mu2, cov2, lb, ub)

                    val = cov4[0][0]

                # fix potential numerical issues, that happen if an entry of the covariance matrix is slightly negative or lb_scaled > ub_scaled
                if np.isnan(val):
                    val = 0.
                elif val < 0.:
                    val = 0.

                sigma5[idx] = val

            # calculate predictive entropy
            H5 += 0.5 * np.log(sigma5 + self.model.Gaussian_noise.variance)
        
        # entropy without conditioning
        _, cov_unconstr = self.model.predict(x, full_cov=False)
        H0 = 0.5 * np.log(cov_unconstr + self.model.Gaussian_noise.variance)

        # entropy_difference
        Hdiff = H0.flatten() - H5 / self.num_samples
        
        # calculate mean change of entropy
        return Hdiff
    
    def _initialize_acq_continuous_parallelized(self):
            X = self.model.X
            controllables = X[:, :self.space[0]['dimensionality']]
            train_uncontrollables = X[:, self.space[0]['dimensionality']:]    

            @ray.remote
            def get_initial_values(fun_sample_idx, self, controllables, counter=0):
                mu_cond = np.nan
                fun_sample_idx_orig = copy.copy(fun_sample_idx)
                while (np.any(np.isnan(mu_cond))) & (counter < 3):
                    fun_sample_idx = (self.num_samples*counter) + fun_sample_idx_orig
                    counter += 1

                    def objective(X, *args):   
                        X = np.atleast_2d(X)
                        return self.fun_samples(X)[0][:, fun_sample_idx]

                    def jac(X, nx, *args):
                        X = np.atleast_2d(X)
                        deriv = self.fun_samples(X)[1][:, :, fun_sample_idx]
                        return deriv[:, :nx].flatten(), deriv[:, nx:].flatten()
                    
                    # find min max of sample
                    num_restarts = 10
                    g_star, min_max_location = find_min_max_continuous(objective=objective, jac=jac, num_restarts=num_restarts, space=self.space)  

                    # find the argmax value in the direction of uncontrollables for each controllable 
                    ## add representers and their values here (representers consist of a grid plus the train data)
                    i, worst_case_vals, representers, representer_worst_cases = find_argmax(fun_sample=self.fun_samples, fun_sample_idx=fun_sample_idx, x=controllables, space=self.space, uncontrollables=train_uncontrollables, mode='continuous')

                    # if any of the worst_case_vals is smaller then g_star, use that (may happen in unfortunate cases)
                    if np.min(worst_case_vals) < g_star:
                        g_star = np.min(worst_case_vals)

                    # the bounds
                    lb = np.ones((2*X.shape[0], 1))
                    lb[:X.shape[0]] *= -1e6 
                    lb[X.shape[0]:] = g_star
                    ub = np.tile(worst_case_vals, 2).reshape(-1, 1)

                    mask = np.concatenate((np.all((X[:, self.space[0]['dimensionality']:] != i), axis=1), np.ones(X.shape[0], dtype=bool)))
                    full_mask = (mask) & (np.linalg.norm(np.abs(lb - ub), axis=1) > 1e-6) 

                    lb = lb[full_mask]
                    ub = ub[full_mask]

                    X_grid = np.vstack((X, np.hstack((X[:, :self.space[0]['dimensionality']], i))))[full_mask]

                    # create predictions with the original GP
                    mu, cov = self.model.predict(Xnew=X_grid, full_cov=True)

                    # EP
                    if mu.shape[0] != 1:                    
                        mu_cond, cov_cond = expectation_propagation_trunc_gauss(mean=mu, cov=cov, lb=lb, ub=ub, n_max_sweeps=50, abs_tol=1e-6)  
                        if np.any(np.isnan(mu_cond)):
                            np.savez('error_init_MMES.npz', mu=mu, cov=cov, lb=lb, ub=ub)
                    else:
                        scale = np.sqrt(cov[0, 0])

                        # scale the bounds to be n standard deviations
                        lb_scaled = (lb - mu) / scale
                        ub_scaled = (ub - mu) / scale

                        mu_cond, cov_cond = sc.stats.truncnorm(a=lb_scaled, b=ub_scaled, loc=mu, scale=scale).stats('mv')

                        mu_cond = mu_cond.reshape(mu.shape)

                    if counter == 3:
                        print('had to resample in initialization')

                return g_star, i, X_grid, mu_cond, cov_cond, min_max_location, representers, representer_worst_cases, counter

            counter = 0 # there is a low chance (1/1000) to have sample that cannot be used for conditioning
            while (self.initialized_acq == False) and (counter < 5):
                # draw samples from the SSGP
                self.fun_samples = self._draw_SSGP_samples(n_samples=self.num_samples*3) # if you call fun_samples, the first index is the function value, the second its first-order derivative
                futures = [get_initial_values.remote(fun_sample_idx, self, controllables, 0) for fun_sample_idx in range(self.num_samples)]
                result_object = ray.get(futures)
                self.g_stars = [result_object[i][0] for i in range(self.num_samples)]
                self.uncontrollables = [result_object[i][1] for i in range(self.num_samples)]
                self.X_grid = [result_object[i][2] for i in range(self.num_samples)]
                self.mu1s = [result_object[i][3] for i in range(self.num_samples)]
                self.cov1s = [result_object[i][4] for i in range(self.num_samples)]
                self.min_max_locations = [result_object[i][5] for i in range(self.num_samples)]
                self.representers = [result_object[i][6] for i in range(self.num_samples)]
                self.representer_worst_cases = [result_object[i][7] for i in range(self.num_samples)]
                self.counters = [result_object[i][8] for i in range(self.num_samples)]

                if any([np.any(np.isnan(mu)) for mu in self.mu1s]):
                    counter +=1
                    print('need to resample, as a sample is not valid')
                else:
                    self.initialized_acq = True     

    def _compute_acq_continuous_parallelized(self, x):
        """
        Computes the RES acquisition function for continuous uncontrollables
        """
        x = np.atleast_2d(x)
        
        if self.initialized_acq == False:
            self._initialize_acq_continuous_parallelized()

        # loop over the samples
        H5 = 0

        @ray.remote
        def get_predictive_entropy(fun_sample_idx, x, self):
            i_new, worst_case_vals, representers, representer_worst_cases = find_argmax(fun_sample=self.fun_samples, fun_sample_idx=fun_sample_idx*self.counters[fun_sample_idx], x=x[:, :self.space[0]['dimensionality']], space=self.space, uncontrollables=x[:, self.space[0]['dimensionality']:], mode='continuous', representers=self.representers[fun_sample_idx], representer_worst_cases=self.representer_worst_cases[fun_sample_idx])

            sigma5 = np.zeros((x.shape[0]))

            for idx, xx in enumerate(x):
                xx = np.atleast_2d(xx)
                
                x_new = np.vstack((xx, np.hstack((xx[:, :self.space[0]['dimensionality']], i_new[[idx]]))))

                # prediction for the current x, taking into account all possible grid values
                K = self.model.kern.K(X=self.X_grid[fun_sample_idx])
                k_xX = self.model.kern.K(X=x_new, X2=self.X_grid[fun_sample_idx])

                # p( g(x) | g, y ) = N( g(x) | mu_g, cov_g )
                K_chol = compute_stable_cholesky(K)[0]
                s = np.linalg.solve(K_chol.T, np.linalg.solve(K_chol, k_xX.T)) 

                covg = self.model.kern.K(x_new) - s.T @ k_xX.T
                a1 = s.T

                # p0(g(x) | y, g *) = N( g(x) | mu2, v2 )
                mu2 = a1 @ self.mu1s[fun_sample_idx]
                cov2 = covg + a1 @ self.cov1s[fun_sample_idx] @ a1.T

                if worst_case_vals[idx] < self.g_stars[fun_sample_idx]:
                    # this might happen if we are in bad luck with the numerical solvers
                    self.g_stars[fun_sample_idx] = worst_case_vals[idx]

                if (np.linalg.norm(self.g_stars[fun_sample_idx] - worst_case_vals[idx]) < 1e-6):
                    # the value is close to the min max
                    if (np.linalg.norm(x_new[[0]]-x_new[[1]]) < 1e-6):
                        # we're at the worst case
                        val = 0
                    else:
                        mu = mu2[0]
                        scale = np.sqrt(cov2[0, 0])

                        lb = -np.infty
                        ub = worst_case_vals[idx] 

                        # scale the bounds to be n standard deviations
                        lb_scaled = (lb - mu) / scale
                        ub_scaled = (ub - mu) / scale

                        val = sc.stats.truncnorm(a=lb_scaled, b=ub_scaled, loc=mu, scale=scale).stats('v')
                elif (np.linalg.norm(x_new[[0]]-x_new[[1]]) < 1e-6):
                    # we're at the worstcase but it's value is not g_star
                    mu = mu2[0]
                    scale = np.sqrt(cov2[0, 0])

                    lb = self.g_stars[fun_sample_idx]
                    ub = worst_case_vals[idx]

                    # scale the bounds to be n standard deviations
                    lb_scaled = (lb - mu) / scale
                    ub_scaled = (ub - mu) / scale

                    val = sc.stats.truncnorm(a=lb_scaled, b=ub_scaled, loc=mu, scale=scale).stats('v')
                    # calculate mu and sigma via scipy
                else:
                    lb = np.zeros(x_new.shape[0])
                    lb[0] = -np.infty
                    lb[1] = self.g_stars[fun_sample_idx]

                    ub = np.ones(2) * worst_case_vals[idx]

                    _, cov4 = calculate_2d_truncated_moments(mu2, cov2, lb, ub)

                    val = cov4[0][0]  
                
                # clean numerical issues
                if np.isnan(val):
                    val = 0.
                elif val < 0.:
                    val = 0.

                sigma5[idx] = val

            # calculate predictive entropy
            H5 = 0.5 * np.log(sigma5 + self.model.Gaussian_noise.variance)

            return H5, representers, representer_worst_cases


        futures = [get_predictive_entropy.remote(fun_sample_idx, x, self) for fun_sample_idx in range(self.num_samples)]
        result_object = ray.get(futures)
        H5s = [result_object[i][0] for i in range(self.num_samples)]
        self.representers = [result_object[i][1] for i in range(self.num_samples)]
        self.representer_worst_cases = [result_object[i][2] for i in range(self.num_samples)]

        H5 = np.sum(H5s, axis=0)

        # calculate mean change of entropy
        _, cov_unconstr = self.model.predict(x, full_cov = False)
        H0 = 0.5 * np.log(cov_unconstr + self.model.Gaussian_noise.variance)
        
        return H0.flatten() - H5 / self.num_samples

    def _draw_SSGP_samples(self, with_derivative=True, n_samples=1):
        if with_derivative:
            fun_samples = self.ssgp_model.sample_posterior_handle(n_samples=n_samples)
        else:
            fun_samples = self.ssgp_model.sample_posterior_handle_no_derivative(n_samples=n_samples)

        return fun_samples
    
