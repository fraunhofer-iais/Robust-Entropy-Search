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

def load_synth_poly(initialization):
    np.random.seed(initialization)

    def synth_poly_2d(x, noise_var=0.0):
        x = np.atleast_2d(x)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x1_terms = -2*x1**6 + 12.2*x1**5 - 21.2*x1**4 + 06.4*x1**3 + 04.7*x1**2 - 06.2*x1
        x2_terms = -1*x2**6 + 11.0*x2**5 - 43.3*x2**4 + 74.8*x2**3 - 56.9*x2**2 + 10.0*x2
        x12_terms = 4.1*x1*x2 + 0.1*x1**2*x2**2 - 0.4*x1*x2**2 - 0.4*x1**2*x2
        f = x1_terms + x2_terms + x12_terms
        f = f.T

        return f + np.sqrt(noise_var) * np.random.randn(*f.shape)
    
    def synth_poly_2d_standard(x, noise_var=0.):
        return (synth_poly_2d(x, noise_var) + 14.133) / 11.144
    
    def synth_poly_2d_standard_normed_input(x_normed, noise_var=0.):
        x_normed = np.atleast_2d(x_normed)
        x1 = x_normed[:, 0]
        x2 = x_normed[:, 1]

        x1 = x1 * (3.2 + 0.95) - 0.95
        x2 = x2 * (4.4 + 0.45) - 0.45

        x = np.array([[x1, x2]])
        return synth_poly_2d_standard(x, noise_var)
    
    def synth_poly_4d_standard_normed_input(xd_normed, noise_var=0.):
        xd_normed = np.atleast_2d(xd_normed)
        x1 = xd_normed[:, 0]
        x2 = xd_normed[:, 1]

        r = xd_normed[:, 2]
        theta = xd_normed[:, 3]

        x1 = x1 * (3.2 + 0.95) - 0.95
        x2 = x2 * (4.4 + 0.45) - 0.45

        r = r * 0.5
        theta = theta * 2 * np.pi

        d1 = r * np.cos(theta)
        d2 = r * np.sin(theta)

        x = np.array([[x1 + d1, x2 + d2]])
        return synth_poly_2d_standard(x, noise_var)

    # take 500 samples with function value above -15. and learn the model's hyperparameters
    x_init_model = np.random.rand(3000, 4)
    f_init_model = synth_poly_4d_standard_normed_input(x_init_model)
    x_init_model = x_init_model[(f_init_model > -15.).flatten()][:500]
    f_init_model = f_init_model[(f_init_model > -15.).flatten()][:500]
    assert len(f_init_model) == 500

    kernel = GPy.kern.RBF(4, ARD=True)
    model = GPy.models.GPRegression(x_init_model, f_init_model, kernel=kernel)
    model.Gaussian_noise.variance.constrain_fixed(0.1 / 11.144)
    model.optimize_restarts(10);   

    # take 10 combinations of pot_x and pot_d
    x_init = np.random.rand(10, 4)
    f_init = synth_poly_4d_standard_normed_input(x_init)

    # set the data of the model
    model.set_XY(x_init, f_init)

    return synth_poly_4d_standard_normed_input, model