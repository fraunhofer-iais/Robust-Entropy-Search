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

def load_hartmann(initialization):
    np.random.seed(initialization)

    def objective_function(x):
        alpha = np.array([[1., 1.2, 3.0, 3.2]]).T

        A = np.array([[3, 10, 30], [0.1, 10, 35], [3., 10, 30], [0.1, 10, 35]])

        P = 1e-4*np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])

        f = np.zeros((x.shape[0], 1))
        for i in range(4):
            tmp = np.zeros(f.shape)
            for j in range(3):
                tmp += (A[i, j]*(x[:, j] - P[i, j])**2)[:, None]
            f += alpha[i] * np.exp(-tmp)

        return f

    xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    controllables = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
    uncontrollables = np.array([[0.2], [0.5], [0.9]])
    
    controllables_init = np.random.choice(np.arange(controllables.shape[0]), size=1)
    uncontrollables_init = np.random.choice(np.arange(uncontrollables.shape[0]), size=1)
    
    x_init_model = np.hstack((controllables[controllables_init], uncontrollables[uncontrollables_init]))
    f_init_model = objective_function(x_init_model)

    # train the model
    kernel = GPy.kern.RBF(3, ARD=True)
    model = GPy.models.GPRegression(x_init_model, f_init_model, kernel=kernel)
    model.optimize_restarts(10)

    return objective_function, model, controllables, uncontrollables
