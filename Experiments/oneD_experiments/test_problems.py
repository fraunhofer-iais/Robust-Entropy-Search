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

# several subfunctions related to the test problems
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_test(tag):
    # 2 dimensional test functions
    if tag == 'branin':
        def testfunction(X):
            # 2-dimensional, 3 local minima, typically evaluated in [-5, 10], [10, 15]
            # this function is to be minimized
            a = 1
            b = 5.1/(4*(np.pi**2))
            c = 5/np.pi
            r = 6
            s = 10
            t = 1/(8*np.pi)

            x2 = X[:, 1]
            x1 = X[:, 0]

            return -(a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
        slices = np.array([[0., 4., 8., 12.]]).T

        # scalers: input domain of [-5, 10] [0, 15] to 0,1
        X_scale = np.array([[-5, 0], [10, 15]])

        # hyperparameters
        hypers = [0.001, 1, np.array([[0.2, 0.4]])]

        # Min-Max-Scaler on output
        scaler3 = MinMaxScaler()
        ymax = testfunction(np.array([[-np.pi, 12.275]]))
        ymin = testfunction(np.array([[-5, 0]]))
        scaler3.fit(np.array([ymin, ymax]))
    elif tag == 'eggholder':
        def testfunction(X):
            x2 = X[:, 1]
            x1 = X[:, 0]

            tfunction = -(x2 + 47)*np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))-x1*np.sin(np.sqrt(np.abs(x1-(x2+47))))
            return tfunction
        slices = np.array([[-512., 0., 185.]]).T

        # scalers: input domain of [-512, 512] [-512, 512] to 0,1
        X_scale = np.array([[-512, -512], [512, 512]])

        # hyperparameters
        hypers = [0.001, 1., np.array([[0.09, 0.09]])]

        # Min-Max-Scaler on output
        scaler3 = MinMaxScaler()
        ymax = testfunction(np.array([[-512, 512]]))
        ymin = testfunction(np.array([[512, 404.2319]]))
        scaler3.fit(np.array([ymin, ymax]))
    else:
        raise Exception('Testfunction not implemented.')

    scaler1 = MinMaxScaler()
    scaler1.fit(X_scale)

    slices = np.array(scaler1.transform(np.concatenate((np.zeros_like(slices), slices), axis=1)))[:, 1]

    # y-scale: monte-carlo integration, as scipy dblquad fails
    X = np.array(scaler1.inverse_transform(np.random.rand(10 ** 7, 2)))
    y = testfunction(X)
    scaler2 = StandardScaler()
    scaler2.fit(y.reshape(-1, 1))

    return testfunction, slices, [scaler1, scaler2, scaler3], hypers
