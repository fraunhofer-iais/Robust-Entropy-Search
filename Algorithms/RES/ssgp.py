# Copyright (c) 2020 Robert Bosch GmbH
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
# Author: Lukas P. Fröhlich, lukas.froehlich@de.bosch.com

# This code has been partially modified by D. Weichert on January, 31st, 2024

import numpy as np
from scipy import linalg, stats
from .util import compute_stable_cholesky

class SparseSpectrumGP:
    def __init__(self, kernel, X, Y, noise_var, n_features, normalize_Y=False, with_derivative=True):
        self.input_dim = kernel.input_dim
        self.lengthscale = np.array(kernel.lengthscale)
        self.signal_var = np.array(kernel.variance)
        self.noise_var = np.array(noise_var)
        self.X = X
        self.Y = Y if Y.ndim == 2 else np.atleast_2d(Y).T

        self.normalize_Y = normalize_Y
        if self.normalize_Y:
            self.Y, self.y_mean = shift_y(Y)

        self.n_features = n_features
        self.w = None
        self.b = None
        if with_derivative:
            self.phi, self.dphidx = self._compute_phi()
        else: 
            self.phi = self._compute_phi_no_deriv()

        phi_train = self.phi(X)
        A = phi_train @ phi_train.T + self.noise_var * np.eye(self.n_features)
        chol_A = compute_stable_cholesky(A)

        B = np.eye(self.X.shape[0]) + phi_train.T @ phi_train / self.noise_var
        chol_B = compute_stable_cholesky(B)
        v = linalg.cho_solve(chol_B, phi_train.T)

        a_inv = (np.eye(self.n_features) - phi_train @ v / self.noise_var)

        self.theta_mu = linalg.cho_solve(chol_A, phi_train @ self.Y)
        self.theta_var = a_inv

    def predict(self, Xs, full_variance=False):
        phi_x = self.phi(Xs)

        mu = phi_x.T @ self.theta_mu
        mu = deshift_y(mu, self.y_mean) if self.normalize_Y else mu

        var_full = phi_x.T @ self.theta_var @ phi_x

        if full_variance:
            var_full = 0.5 * (var_full + var_full.T)
            return mu, var_full
        else:
            var = np.diag(var_full)
            return mu, var

    def sample_posterior(self, Xs, n_samples):
        fs_h = self.sample_posterior_handle(n_samples)
        return fs_h(Xs)

    def sample_posterior_handle(self, n_samples):
        """
        Generate handle to n_samples function samples that can be evaluated at x.
        """
        chol = compute_stable_cholesky(self.theta_var)[0]
        theta_samples = self.theta_mu + chol @ np.random.randn(self.n_features, n_samples)

        def handle_to_function_samples(x):
            if x.ndim == 1 and self.input_dim == 1:
                x = np.atleast_2d(x).T
            elif x.ndim == 1 and self.input_dim > 1:
                x = np.atleast_2d(x)

            h = self.phi(x).T @ theta_samples
            dh = self.dphidx(x).T @ theta_samples
            return (deshift_y(h, self.y_mean), dh) if self.normalize_Y else (h, dh) 

        return handle_to_function_samples
    
    def sample_posterior_handle_no_derivative(self, n_samples):
        """
        Generate handle to n_samples function samples that can be evaluated at x.
        """
        chol = compute_stable_cholesky(self.theta_var)[0]
        theta_samples = self.theta_mu + chol @ np.random.randn(self.n_features, n_samples)

        def handle_to_function_samples(x):
            if x.ndim == 1 and self.input_dim == 1:
                x = np.atleast_2d(x).T
            elif x.ndim == 1 and self.input_dim > 1:
                x = np.atleast_2d(x)

            h = self.phi(x).T @ theta_samples
            return deshift_y(h, self.y_mean) if self.normalize_Y else h 

        return handle_to_function_samples

    def sample_posterior_weights(self, n_samples):
        theta_samples = stats.multivariate_normal.rvs(mean=self.theta_mu.squeeze(),
                                                      cov=self.theta_var,
                                                      size=n_samples).T
        return theta_samples

    def _compute_phi(self):
            """
            Compute random features.
            """
            lin_3sigma = np.linspace(stats.norm.cdf(-3), stats.norm.cdf(3), self.n_features * self.input_dim)
            lin_inv_cdf = stats.norm.ppf(lin_3sigma)
            self.w = np.random.permutation(lin_inv_cdf).reshape(self.n_features, self.input_dim) / self.lengthscale
            self.b = np.random.permutation(np.linspace(0, 2 * np.pi, self.n_features))[:, None]

            return lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(self.w @ x.T + self.b), lambda x: -1*np.sqrt(2 * self.signal_var / self.n_features) * np.array([np.outer(self.w[i], np.sin(self.w @ x.T + self.b)[i]) for i in range(self.n_features)])
    
    def _compute_phi_no_deriv(self):
            """
            Compute random features.
            """
            lin_3sigma = np.linspace(stats.norm.cdf(-3), stats.norm.cdf(3), self.n_features * self.input_dim)
            lin_inv_cdf = stats.norm.ppf(lin_3sigma)
            self.w = np.random.permutation(lin_inv_cdf).reshape(self.n_features, self.input_dim) / self.lengthscale
            self.b = np.random.permutation(np.linspace(0, 2 * np.pi, self.n_features))[:, None]

            return lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(self.w @ x.T + self.b)
    


    @staticmethod
    def from_gp(gp, n_ssgp_features, with_derivative):
        """Construct a SparseSpectrumGP from a GP"""
        Y = gp.Y # this is unnormalized data. Normalized is gp.Y_normalized
        return SparseSpectrumGP(gp.kern, gp.X, Y, gp.Gaussian_noise.variance, n_ssgp_features, normalize_Y=gp.normalizer, with_derivative=with_derivative)

def deshift_y(Y, y_mean):
    return Y + y_mean

def shift_y(Y):
    y_mean = np.mean(Y)
    Y = Y - y_mean
    return Y, y_mean