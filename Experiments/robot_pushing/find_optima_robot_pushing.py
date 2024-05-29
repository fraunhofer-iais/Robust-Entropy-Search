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
from scipy.optimize import minimize
from Experiments.experiment_utils import load_experiment

def find_optima_robot_pushing():
    # find the optima for all 30 initializations:
    xopts = []
    fopts = []
    for ini in range(30):
        model, input_scaler, output_scaler, func_on_normalized, space, iterations, uncontrollables, objective_fun = load_experiment('robot_pushing', 'RVES', ini);

        def optimization_objective(x, func, uncontrollables):
            # combine the actual value of x with all uncontrollables
            x = np.atleast_2d(x)
            X = np.hstack((np.repeat(x, repeats=uncontrollables.shape[0], axis=0), uncontrollables))
    
            vals = func(X)
            return np.max(vals, axis=0)
    
        num_restarts = 50
        restarts = np.random.rand(num_restarts, space[0]['dimensionality'])
        bounds = [(0, 1) for _ in range(space[0]['dimensionality'])]
    
        x_opts = np.zeros_like(restarts)
        fx_opts = np.zeros(num_restarts)
        for idx, restart in enumerate(restarts):
            res = minimize(optimization_objective, x0=restart, args=(func_on_normalized, space[1]['values']), bounds=bounds)
            x_opts[idx] = res.x
            fx_opts[idx] = res.fun
    
        argmin_val = np.argmin(fx_opts)
        x_opt = x_opts[[argmin_val]]
        f_opt = fx_opts[argmin_val]
    
        # find the correct uncontrollable
        opt_uncontrollable = uncontrollables[np.argmax(func_on_normalized(np.hstack((np.repeat(x_opt, repeats=uncontrollables.shape[0], axis=0), uncontrollables))))]

        x_opt = input_scaler.inverse_transform(np.append(x_opt, opt_uncontrollable).reshape(1, -1))
        f_opt = output_scaler.inverse_transform(f_opt.reshape(1, -1))
        xopts.append(x_opt)
        fopts.append(f_opt)
        
    np.savez('optima_robot_pushing.npz', x_opts=np.array(xopts), f_opts=np.array(fopts))