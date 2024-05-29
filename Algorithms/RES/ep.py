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
from scipy import stats


def expectation_propagation_trunc_gauss(mean, cov, lb=None, ub=None, n_max_sweeps=50, abs_tol=1e-6):
    """
    Implements the Gaussian EP algorithm applied to a multivariate
    truncated Gaussian distribution of the following form:
    .. math::
        p(x) = \mathcal{N}(x | \mu, \Sigma) \prod_{i = 1}^{n} I_{lb_i < x_i < ub_i}
    Note: This implementation follows Algorithm 3 in this paper:
          https://www.microsoft.com/en-us/research/wp-content/uploads/2005/07/EP.pdf
    :param mean: Mean vector of the (non-truncated) Gaussian distribution.
    :param cov: Covariance of the (non-truncated) Gaussian distribution.
    :param lb: Lower bound for the truncation.
    :param ub: Upper bound for the truncation.
    :param n_max_sweeps: Maximum number of sweep over all factors.
    :param abs_tol: Tolerance below which a value is assumed to be converged.
    :return: Mean and covariance of an approximated Gaussian distribution.
    """
    def v(t, l, u):
        """
        Helper function for EP.
        """
        numerator = stats.norm.pdf(l - t) - stats.norm.pdf(u - t)
        denominator = stats.norm.cdf(u - t) - stats.norm.cdf(l - t)
        return numerator / denominator

    def w(t, l, u):
        """
        Helper function for EP.
        """
        numerator = (u - t) * stats.norm.pdf(u - t) - (l - t) * stats.norm.pdf(l - t)
        denominator = stats.norm.cdf(u - t) - stats.norm.cdf(l - t)
        return v(t, l, u) ** 2 + numerator / denominator

    # Dimension of the random variable
    mean = mean.squeeze()
    dim = mean.shape[0]

    # If no bound are defined, set them very large/small such that they become ineffective
    if lb is None and ub is None:
        return mean.copy(), cov.copy()
    elif lb is None:
        lb = -1e6 * np.ones(dim)
    elif ub is None:
        ub = 1e6 * np.ones(dim)

    # For numerical stability if bounds are very small / large
    jitter = 1e-10

    # Initialize approximating factors
    mu_n = np.zeros((dim,))
    pi_n = np.zeros((dim,))
    s_n = np.ones((dim,))

    # Initialize mean and variance of approximating Gaussian (follows from factor initialization)
    mean_hat = mean.copy()
    cov_hat = cov.copy()

    # Pick an index and perform updates
    for i_sweep in range(n_max_sweeps):
        # Check for convergence after each sweep over all factors
        mu_n_old = mu_n.copy()
        pi_n_old = pi_n.copy()
        s_n_old = s_n.copy()

        for j in range(dim):
            # Pre-computations
            t_j = cov_hat[:, j]
            d_j = pi_n[j] * cov_hat[j, j]
            e_j = 1 / (1 - d_j)

            phi_j = mean_hat[j] + d_j * e_j * (mean_hat[j] - mu_n[j])
            psi_j = cov_hat[j, j] * e_j

            phi_prime_j = phi_j / np.sqrt(psi_j)
            lb_prime_j = lb[j] / np.sqrt(psi_j)
            ub_prime_j = ub[j] / np.sqrt(psi_j)

            alpha_j = v(phi_prime_j, lb_prime_j, ub_prime_j) / np.sqrt(psi_j)
            beta_j = w(phi_prime_j, lb_prime_j, ub_prime_j) / psi_j

            # ADF update
            mean_hat += e_j * (pi_n[j] * (mean_hat[j] - mu_n[j]) + alpha_j) * t_j
            cov_hat += (pi_n[j] * e_j - e_j ** 2 * beta_j) * np.outer(t_j, t_j)

            # Factor update
            pi_n[j] = beta_j / (1 - beta_j * psi_j)
            mu_n[j] = alpha_j / (beta_j + jitter) + phi_j

            tmp1 = stats.norm.cdf(ub_prime_j - phi_prime_j) - stats.norm.cdf(lb_prime_j - phi_prime_j)
            tmp2 = np.exp(alpha_j ** 2 / (2 * beta_j + jitter)) / np.sqrt(1 - psi_j * beta_j)
            s_n[j] = tmp1 * tmp2

        # Calculate differences of factors before sweep
        mu_n_diff = np.max(np.abs(mu_n - mu_n_old))
        pi_n_diff = np.max(np.abs(pi_n - pi_n_old))
        s_n_diff = np.max(np.abs(s_n - s_n_old))

        if (np.array([mu_n_diff, pi_n_diff, s_n_diff]) <= abs_tol).all():
            # print("EP converged after {} iterations".format(i_sweep + 1))
            break

    return mean_hat, cov_hat