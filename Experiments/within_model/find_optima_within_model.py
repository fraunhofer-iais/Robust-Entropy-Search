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
import numpy as np
from Experiments.experiment_utils import load_experiment

xopts = []
fopts = []
for ini in range(50):
    model, input_scaler, output_scaler, func_on_normalized, space, iterations, uncontrollables, objective_fun = load_experiment('within_model', 'RVES', ini);

    def optimization_objective(x, func, space):
        # combine the actual value of x with all uncontrollables
        x = np.atleast_2d(x)
        theta_inits = np.linspace(0, 1, num_restarts)
        bounds = [(0, 1) for _ in range(space[1]['dimensionality'])]

        fx_opts = np.zeros(num_restarts)
        for idx, theta_init in enumerate(theta_inits):
            res = minimize(inner_objective, x0=theta_init, args=(x, func), bounds=bounds, method='Nelder-Mead')
            fx_opts[idx] = -1 * res.fun
            
        return np.max(fx_opts, axis=0)
    # 
    def inner_objective(theta, x, func):
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        X = np.hstack((x, theta))
        return -1*func(X)

    num_restarts = 20
    restarts = np.linspace(0, 1, num_restarts)
    bounds = [(0, 1) for _ in range(space[0]['dimensionality'])]
    # 
    x_opts = np.zeros_like(restarts)
    fx_opts = np.zeros(num_restarts)
    for idx, restart in enumerate(restarts):
        res = minimize(optimization_objective, x0=restart, args=(func_on_normalized, space), bounds=bounds, method='Nelder-Mead')
        x_opts[idx] = res.x
        fx_opts[idx] = res.fun
    # 
    argmin_val = np.argmin(fx_opts)
    x_opt = x_opts[[argmin_val]]
    f_opt = fx_opts[argmin_val]
    print('found optimal x')

    # find the correct uncontrollable 
    theta_inits = np.linspace(0, 1, num_restarts)
    fx_opts = np.zeros(num_restarts)
    theta_opts = np.zeros(num_restarts)
    for idx, theta_init in enumerate(theta_inits):
        res = minimize(inner_objective, x0=theta_init, args=(x_opt, func_on_normalized), bounds=bounds, method='Nelder-Mead')
        fx_opts[idx] = -1*res.fun
        theta_opts[idx] = res.x

    X_opt = np.hstack((x_opt, theta_opts[np.argmax(fx_opts)]))
    
    xopts.append(X_opt)
    fopts.append(f_opt)

np.savez('optima_within_model.npz', xopts=xopts, f_opts=fopts)