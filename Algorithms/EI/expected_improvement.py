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

from Algorithms.base import AcquisitionBase
import numpy as np
import scipy as sc

class ExpectedImprovement(AcquisitionBase):
    """
    """

    def __init__(self, model, space):
        super(ExpectedImprovement, self).__init__(model, space)

    def _compute_acq(self, x):
        """
        Computes the EI
        """
        x = np.atleast_2d(x)
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        u = (y_minimum - mean) / standard_deviation
        pdf = sc.stats.norm.pdf(u)
        cdf = sc.stats.norm.cdf(u)

        improvement = standard_deviation * (u * cdf + pdf)
        return -improvement

    def _compute_acq_withGradients(self, x):
        """
        Computes the EI with derivatives
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        d_mean_dx, d_variance_dx = self.model.predictive_gradients(x)
        d_mean_dx = d_mean_dx[:, :, 0]
        dstandard_deviation_dx = d_variance_dx / (2 * standard_deviation)

        u = (y_minimum - mean) / standard_deviation
        pdf = sc.stats.norm.pdf(u)
        cdf = sc.stats.norm.cdf(u)

        improvement = standard_deviation * (u * cdf + pdf)
        dimprovement_dx = dstandard_deviation_dx * pdf - cdf * d_mean_dx
        return -improvement, -dimprovement_dx