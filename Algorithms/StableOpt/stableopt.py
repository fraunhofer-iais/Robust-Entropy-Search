from Algorithms.base import AcquisitionBase
from .maxUCB import AcquisitionmaxUCB
from .lcb import AcquisitionLCB
import numpy as np
from scipy.optimize import minimize
import ray

class AcquisitionStableOpt(AcquisitionBase):
    """
    only for use with fixed uncontrollables. I need to have a look on the continuous approach (use GDA?)
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, uncontrollables, exploration_weight=2):
        super(AcquisitionStableOpt, self).__init__(model, space)
        self.exploration_weight = exploration_weight
        self.uncontrollables = uncontrollables

    def _return_new_discrete_point(self):

        if self.space[0]['type'] == 'continuous':
            if self.space[1]['type'] == 'discrete':
                inner_acquisition = AcquisitionmaxUCB(self.model, self.space, uncontrollables=self.uncontrollables, exploration_weight=self.exploration_weight)

                # optimize it
                x_inits = np.random.rand(10, self.space[0]['dimensionality'])
                bounds = [(0, 1) for _ in range(self.space[0]['dimensionality'])]
                
                def gradient_of_objective(x):
                    return inner_acquisition._compute_acq_withGradients(x)[1]
                results = [minimize(inner_acquisition._compute_acq, x0=x_ini, bounds=bounds, jac=gradient_of_objective) for x_ini in x_inits]

                fxopts = [res.fun for res in results]
                xopts = [res.x for res in results]

                fxopt = np.nanmin(fxopts)
                opt_lin = np.arange(10)

                xopt = xopts[np.random.choice(opt_lin[fxopts == fxopt])]
                xopt = np.atleast_2d(xopt)

                # second step: min lcb over all uncontrollables at returned point
                # change x to be dim0 connected with the uncontrollables
                X_n = np.hstack([np.repeat(xopt, self.uncontrollables.shape[0], axis=0), self.uncontrollables])

                outer_acquisition = AcquisitionLCB(self.model, self.space, exploration_weight=self.exploration_weight)
                outer_acquisition_values = -1*outer_acquisition._compute_acq(X_n)
                outer_acquisition_opt = np.nanmin(outer_acquisition_values)

                opt_lin = np.arange(self.uncontrollables.shape[0])
                x_new = X_n[np.random.choice(opt_lin[outer_acquisition_values.flatten() == outer_acquisition_opt])]
            else:
                inner_acquisition = AcquisitionmaxUCB(self.model, self.space, uncontrollables=self.uncontrollables, exploration_weight=self.exploration_weight)

                # optimize it
                x_inits = np.random.rand(10, self.space[0]['dimensionality'])
                bounds = [(0, 1) for _ in range(self.space[0]['dimensionality'])]

                @ray.remote
                def find_x(objective, x_init, bounds):
                    res = minimize(objective, x0=x_init, bounds=bounds)
                    return res.x, res.fun

                #results = [minimize(inner_acquisition._compute_acq, x0=x_ini, bounds=bounds) for x_ini in x_inits]
                optimized_values = [find_x.remote(objective=inner_acquisition._compute_acq, x_init=x_ini, bounds=bounds) for x_ini in x_inits]
                optimized_values = ray.get(optimized_values)

                fxopts = [optimized_values[i][1] for i in range(10)]
                xopts = [optimized_values[i][0] for i in range(10)]

                fxopt = np.nanmin(fxopts)
                opt_lin = np.arange(10)
                
                xopt = xopts[np.random.choice(opt_lin[fxopts == fxopt])]
                xopt = np.atleast_2d(xopt)

                # second step: compute the theta
                def outer_objective(theta, x, self):
                    X = np.hstack((x, np.atleast_2d(theta)))
            
                    # prediction of mean and variance for all uncontrollables
                    m, s = self.model.predict(X)

                    # compute ucb
                    f_acqu = m - self.exploration_weight * np.sqrt(s)
                    return f_acqu
                
                theta_inits = np.random.rand(10, self.space[1]['dimensionality'])
                bounds = [(0, 1) for _ in range(self.space[1]['dimensionality'])]

                results = [minimize(outer_objective, x0=theta_ini, bounds=bounds, args=(xopt, self)) for theta_ini in theta_inits]

                fopts = [res.fun for res in results]
                theta_opts = [res.x for res in results]
                opt_lin = np.arange(10)

                f_opt = np.nanmin(fopts)

                x_new = np.hstack((xopt, np.atleast_2d(theta_opts[np.random.choice(opt_lin[fopts == f_opt])])))
        else:
            if self.space[1]['type'] == 'discrete':
                # first step: calculate x using ucb
                # generate all combinations of x and theta
                xx, yy = np.meshgrid(np.arange(self.space[0]['values'].shape[0], dtype=int), np.arange(self.space[1]['values'].shape[0], dtype=int))
                X = np.hstack((self.space[0]['values'][xx.flatten()], self.space[1]['values'][yy.flatten()]))

                # prediction of mean and variance for all uncontrollables
                m, s = self.model.predict(X)
                s[s < 0] = 0 # clean numerical instability

                # compute ucb
                ucb = m + self.exploration_weight * np.sqrt(s)

                # reshape this ucb-val correctly
                ucb_reshaped = np.reshape(ucb, xx.shape)

                # compute x
                max_min_val = np.nanmax(np.nanmin(ucb_reshaped, axis=0))
                xopt = X[(ucb==max_min_val).flatten(), :self.space[0]['dimensionality']]
                if xopt.shape[0] > 1:
                    xopt = xopt[np.random.choice(np.arange(xopt.shape[0])), :]

                xopt = np.atleast_2d(xopt)

                # compute lcb at that x
                x_theta = np.hstack((np.repeat(xopt, axis=0, repeats=self.space[1]['values'].shape[0]), self.space[1]['values']))
                m, s = self.model.predict(x_theta)

                lcb = m - self.exploration_weight * np.sqrt(s)

                lcb_opt = np.nanmin(lcb)
                opt_lin = np.arange(self.space[1]['values'].shape[0])

                x_new = x_theta[np.random.choice(opt_lin[lcb.flatten() == lcb_opt])]
            else:
                pass

        return np.atleast_2d(x_new)