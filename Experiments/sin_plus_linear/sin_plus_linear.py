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

def load_sin_p_linear(initialization):
    np.random.seed(initialization)

    def objective_function(x):
        x = np.atleast_2d(x)
        x_eval = x[:, [0]] + x[:, [1]]
        return np.sin(5 * x_eval**2 * np.pi) + 0.5 * x_eval
    
    uncontrollables = np.array([[0.1, 0.05]]).T
    
    x_init_model = np.hstack((np.random.rand(1, 1), np.random.choice(uncontrollables.flatten(), (1, 1))))
    f_init_model = objective_function(x_init_model)

    # train the model
    kernel = GPy.kern.RBF(2, ARD=True)
    model = GPy.models.GPRegression(X=x_init_model, Y=f_init_model, kernel=kernel)
    model.Gaussian_noise.constrain_fixed(0.001)
    model.optimize_restarts(10)

    return objective_function, model, uncontrollables
