# an acquisition function to find the argmin max ucb later (so it's max ucb over the discrete parameters)

from Algorithms.base import AcquisitionBase
import numpy as np
from scipy.optimize import minimize

class AcquisitionmaxUCB(AcquisitionBase):
    """

    """

    def __init__(self, model, space, uncontrollables, exploration_weight=2):
        super(AcquisitionmaxUCB, self).__init__(model, space)
        self.exploration_weight = exploration_weight
        self.uncontrollables = uncontrollables

    def _compute_acq(self, x):
        """
        Computes the GP-UCB, while minimizing over the dimension of the uncontrollable parameters
        """
        if self.space[1]['type'] == 'discrete':
            # change x to be dim0 connected with the uncontrollables
            x = np.atleast_2d(x)
            X = np.hstack([np.repeat(x, self.uncontrollables.shape[0], axis=0), self.uncontrollables])

            # prediction of mean and variance for all uncontrollables
            m, s = self.model.predict(X)

            # compute ucb
            f_acqu = m + self.exploration_weight * np.sqrt(s)

            # reshape to x0 dims x uncontrollables
            f_acqu = f_acqu.reshape(x.shape[0], self.uncontrollables.shape[0])

            # select the minimum over the uncontrollables
            f_acqu_min = np.min(f_acqu, axis=1)

            # change the sign as scipy optimizers minimize
            return -f_acqu_min.item()
        else:
            x = np.atleast_2d(x)
            theta_inits = np.random.rand(10, self.space[1]['dimensionality'])
            bounds = [(0, 1) for _ in range(self.space[1]['dimensionality'])]

            def optimization_objective(theta, x, self):
                X = np.hstack((x, theta.reshape(1, -1)))
                
                # prediction of mean and variance for all uncontrollables
                m, s = self.model.predict(X)

                # compute ucb
                f_acqu = m + self.exploration_weight * np.sqrt(s)

                return f_acqu
            
            def optimization_jac(theta, x, self):
                X = np.hstack((x, theta.reshape(1, -1)))
                dmdx, dsdx = self.model.predictive_gradients(X)

                f_acq_deriv = dmdx[:, self.space[0]['dimensionality']:] + np.atleast_3d(self.exploration_weight * dsdx[:, self.space[0]['dimensionality']:])
                return f_acq_deriv
                        
            # minimize the objective:
            theta_vals = []
            objective_values = []
            for theta_init in theta_inits:
                res = minimize(optimization_objective, x0=theta_init, args=(x, self), jac=optimization_jac, bounds=bounds)
                theta_vals.append(res.x)
                objective_values.append(res.fun)

            return -min(objective_values)

    def _compute_acq_withGradients(self, x):
        """
        Computes the derivative of the max confidence bound
        """
        # we know that x0 is the dimension to be changed, x1 (uncontrollables) is the dimension for the maximization
        # xt = max min ucb(x, theta)
        if self.space[1]['type'] == 'discrete':
            # change x to be dim0 connected with the uncontrollables
            x = np.atleast_2d(x)
            X = np.hstack([np.repeat(x, self.uncontrollables.shape[0], axis=0), self.uncontrollables])

            # prediction of mean and variance for all uncontrollables
            m, s = self.model.predict(X)
            dmdx, dsdx = self.model.predictive_gradients(X)

            # compute ucb
            f_acqu = m + self.exploration_weight * np.sqrt(s)

            # select the minimum over the uncontrollables
            f_acqu_min = np.min(f_acqu, axis=0)
            f_acqu_argmin = np.argmin(f_acqu, axis=0)

            df_acqu = dmdx[f_acqu_argmin, :self.space[0]['dimensionality'], :] + np.atleast_3d(self.exploration_weight * dsdx[f_acqu_argmin, :self.space[0]['dimensionality']])

            # change the sign as scipy optimizers minimize
            return -f_acqu_min, -df_acqu.squeeze()
        else:
            raise(NotImplementedError)
