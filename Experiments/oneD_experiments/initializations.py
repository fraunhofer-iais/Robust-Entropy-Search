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

# create initializations for the three different test problems without torch
# for every problem: create 100*5 random X values, distributed on the given slices
# for every problem: create corresponding responses

from test_problems import get_test
import numpy as np
import pandas as pd

np.random.seed(42)

problems = ['branin', 'eggholder', 'camel']
for problem in problems:
    testfunction, slices, scalers, hyperparameters = get_test(problem)

    # create 500 points on [0, 1]
    X1 = np.random.rand(500, 1)
    X2 = np.random.choice(slices, (500, 1)) # slices are already scaled

    X = scalers[0].inverse_transform(np.concatenate((X1, X2), axis=1)) # scale back for evaluation

    y = testfunction(X)

    # write to csv
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=['X1', 'X2', 'y'])
    df.to_csv('initialization_'+problem+'.csv', index=False)
