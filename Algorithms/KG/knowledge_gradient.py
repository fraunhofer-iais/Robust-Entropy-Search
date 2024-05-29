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

from Algorithms.base import AcquisitionBase
import numpy as np
import copy
from scipy.optimize import minimize
import ray

class KnowledgeGradient(AcquisitionBase):
    """
    Implements the Knowledge Gradient Acquisition Function.
    """

    def __init__(self, model, space, inner_samples=32):
        super(KnowledgeGradient, self).__init__(model, space)
        self.inner_samples = inner_samples
        self.optimum_mean = self._find_optimum(self.model)
        if space[0]['type']=='continuous':
            space[0]['type'] = 'discrete'
            space[0]['values'] = np.random.rand(100, space[0]['dimensionality'])
        if space[1]['type']=='continuous':
            space[1]['type'] = 'discrete'
            space[1]['values'] = np.random.rand(100, space[1]['dimensionality'])


    def _compute_acq_discrete(self, x):
        """
        Computes the Knowledge Gradient on a discrete evaluation grid
        """
        # x is already a flattened grid
        m, s = self.model.predict(x)

        samples = m + s * np.random.randn(x.shape[0], self.inner_samples)
        model_sample = copy.deepcopy(self.model)

        @ray.remote
        def optimization_per_sample(self, samples, model_sample, x, sample_idx):
            kg = np.zeros((x.shape[0], 1))
            for idx in range(x.shape[0]):
                sample = samples[idx, sample_idx].reshape(1, 1)
                x_append = x[[idx]]

                # add the sample to the model            
                model_sample.set_XY(np.append(self.model.X, x_append, axis=0), np.append(self.model.Y, sample, axis=0))
    
                opt_mean = np.min(model_sample.predict(x)[0])
                
                kg[idx] = self.optimum_mean - opt_mean

            return kg

        kgs = [optimization_per_sample.remote(self, samples, model_sample, x, sample_idx) for sample_idx in range(self.inner_samples)]
        kgs = ray.get(kgs)

        kgs = np.array(kgs)[:, :, 0].T  
        f_acqu = np.sum(kgs, axis=1) / self.inner_samples

        return -1 * np.reshape(f_acqu, (-1, 1)) # as we minimize in the optimization routine

    def _compute_acq(self, x):
        """
        Computes the KG for continuous cases
        """
        x = np.atleast_2d(x)
        
        m, s = self.model.predict(x)

        samples = m + s * np.random.randn(x.shape[0], self.inner_samples)
        model_sample = copy.deepcopy(self.model)

        @ray.remote
        def optimization_per_sample(self, samples, model_sample, x, sample_idx):
            kg = np.zeros((x.shape[0], 1))
            for idx in range(x.shape[0]):
                sample = samples[idx, sample_idx].reshape(1, 1)
                x_append = x[[idx]]

                # add the sample to the model            
                model_sample.set_XY(np.append(self.model.X, x_append, axis=0), np.append(self.model.Y, sample, axis=0))
    
                opt_mean = self._find_optimum(model_sample)
                
                kg[idx] = self.optimum_mean - opt_mean
            return kg

        kgs = [optimization_per_sample.remote(self, samples, model_sample, x, sample_idx) for sample_idx in range(self.inner_samples)]
        kgs = ray.get(kgs)

        kgs = np.array(kgs)[:, :, 0].T  
        f_acqu = np.sum(kgs, axis=1) / self.inner_samples

        return -1 * np.reshape(f_acqu, (-1, 1))  # as we minimize in the optimization routine
    
    def _find_optimum(self, model):
        if self.space[0]['type'] == 'discrete':
            if self.space[1]['type'] == 'discrete':
                # find new optimum (new minimum) at all locations
                # evaluate the model at all locations
                xx, yy = np.meshgrid(np.arange(self.space[0]['values'].shape[0]), np.arange(self.space[1]['values'].shape[0]))
                x1 = xx.flatten()
                y1 = yy.flatten()
                x_calc = np.hstack((self.space[0]['values'][x1], self.space[1]['values'][y1]))

                opt_mean = np.min(model.predict(x_calc)[0])
            else:
                raise(NotImplementedError('KG is not implemented for discrete x and continuous theta.'))
        else:
            if self.space[1]['type'] == 'discrete':
                # find the new optimum by gradient descent in controllable direction for all potential uncontrollable parameters

                def optimization_function(x, model, theta):
                    X = np.atleast_2d(np.append(x, theta))
                    return model.predict(X)[0]
                
                n_restarts = 10
                opt_means = []
                bounds = [(0, 1) for _ in range(self.space[0]['dimensionality'])]
                for theta_init in self.space[1]['values']:
                    x_inits = np.random.rand(n_restarts, self.space[0]['dimensionality'])
                    for x_init in x_inits:
                        res = minimize(optimization_function, x_init, args=(model, theta_init), bounds=bounds)
                        opt_means.append(res.fun)
                
                opt_mean = np.nanmin(opt_means)
            else:
                # find the new optimum by gradient descent in both directions
                n_restarts = 10
                X_inits = np.random.rand(n_restarts, self.space[0]['dimensionality'] + self.space[1]['dimensionality'])
                bounds = [(0, 1) for _ in range(self.space[0]['dimensionality'] + self.space[1]['dimensionality'])]

                def objective(X, model):
                    return model.predict(X.reshape(1, -1))[0]

                opt_means = []
                for X_init in X_inits:
                    res = minimize(objective, X_init, args=(model), bounds=bounds)
                    opt_means.append(res.fun)

                opt_mean = np.nanmin(opt_means)

        return opt_mean
