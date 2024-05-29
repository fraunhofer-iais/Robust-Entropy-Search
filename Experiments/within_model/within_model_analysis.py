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
import os
import pandas as pd
from scipy.optimize import minimize
from Experiments.experiment_utils import load_experiment

def analyse_within_model():
    """
    analyses the results of the within model comparison
    """
    
    savepath = 'Results/within_model/analysis/'
    results = np.load(savepath + 'optima_within_model.npz')

    fx_opts = results['f_opts']

    algorithms = ['RVES', 'RVES', 'RVES', 'RVES', 'stableopt', 'stableopt', 'stableopt', 'MES', 'UCB', 'EI', 'KG']
    experiment = 'within_model'
    samples_vec = [30, 5, 10, 30, 1, 2, 4, 1, 1, 1, 1] 


    for idx, algorithm in enumerate(algorithms):
        n_samples = samples_vec[idx]
        inference_regrets = []
        immediate_regrets = []

        if algorithm == 'stableopt':
            filepath = 'Results/' + experiment +'/' + algorithm + '/' 'exploration_' + str(n_samples) + '/'
        elif algorithm == 'RVES':
            filepath = 'Results/' + experiment +'/' + algorithm + '/samples_' + str(samples_vec[idx]) + '/'
        else:
            filepath = 'Results/' + experiment +'/' + algorithm + '/'

        files = os.listdir(filepath)
        max_inits = 50
        if algorithm in algorithms: 
            for ini in range(max_inits):

                _, _, _, _, _, _, _, objective_fun = load_experiment('within_model', algorithm, ini)
                fx_opt = fx_opts[ini]

                filename = [file for file in files if (('init_' + str(ini) + '_') in file)]

                df = pd.read_csv(filepath + filename[0])
        
                immediate_regret = objective_fun(np.array([df.x_cand_0, df.x_cand_1]).T).flatten()
                inference_regret = objective_fun(np.array([df.x_opt_0, df.x_opt_1]).T).flatten()

                immediate_regrets.append(np.abs(immediate_regret - fx_opt.item()))            
                inference_regrets.append(np.abs(inference_regret - fx_opt.item()))
                

            df_inference_regrets = pd.DataFrame(inference_regrets)
            df_immediate_regrets = pd.DataFrame(immediate_regrets)

            # save the dataframes due to the long runtime of the analysis
            df_inference_regrets.to_csv(savepath + 'within_model_inference_regret_' + algorithm + str(n_samples) + '.csv')
            df_immediate_regrets.to_csv(savepath + 'within_model_immediate_regret_' + algorithm + str(n_samples) + '.csv')

