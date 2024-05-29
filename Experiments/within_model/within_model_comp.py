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

import GPy
import numpy as np


def load_within_model_exp(initialization):
    np.random.seed(initialization)

    # hyperparameters of 2D GP
    # lengthscale in both direction
    l = 0.1
    sigma_f = 1
    sigma_n = 0.001

    kern = GPy.kern.RBF(2, lengthscale=l, variance=sigma_f)

    # draw 1000 input points
    X = np.random.rand(1000, 2)

    # draw function values at these points
    mu = np.zeros(1000)
    cov = kern.K(X, X)

    Z = np.random.multivariate_normal(mean=mu, cov=cov).reshape(1000, 1)

    # find the mean of these samples
    # Z_mean = np.mean(Z, axis=0).reshape(1000, 1)

    # use this mean to create a GP, use its mean as objective
    model_objective = GPy.models.GPRegression(X=X, Y=Z, kernel=kern, noise_var=sigma_n)

    def objective(x):
        x = np.atleast_2d(x)
        return model_objective.predict(x)[0]

    # initialize the model
    X_init = np.random.rand(1, 2)
    Y_init = objective(X_init)
    
    kern2 = GPy.kern.RBF(2, lengthscale=l, variance=sigma_f)
    model = GPy.models.GPRegression(X=X_init, Y=Y_init, kernel=kern2, noise_var=sigma_n)

    return objective, model

