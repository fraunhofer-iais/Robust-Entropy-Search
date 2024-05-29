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

from Experiments.run_experiments_parallel import run_experiment_parallel

for experiment in ['branin', 'hartmann_3d', 'eggholder', 'robot_pushing', 'sin_plus_linear', 'synthetic_poly', 'within_model']:
    for algorithm in ['stableopt', 'ES', 'RVES', 'MES', 'stableopt', 'EI', 'UCB', 'KG']:
        if algorithm == 'stableopt':
            for exploration_weight in [1, 2, 4]:
                run_experiment_parallel(experiment, algorithm, exploration_weight=exploration_weight)
        elif (algorithm == 'RVES') & (experiment == 'within_model'):
            for n_samples in [1, 5, 10, 30]:
                run_experiment_parallel(experiment, algorithm, n_samples=n_samples)
        else:
            run_experiment_parallel(experiment, algorithm)