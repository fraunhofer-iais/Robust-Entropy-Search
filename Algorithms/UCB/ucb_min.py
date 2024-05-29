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

class UpperCB(AcquisitionBase):
    """
    Computes the GP-Upper Confidence Bound acquisition function for a minimization problem.
    """

    def __init__(self, model, space, exploration_weight=2):
        super(UpperCB, self).__init__(model, space)
        self.exploration_weight = exploration_weight

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound
        """
        x = np.atleast_2d(x)
        m, s = self.model.predict(x)
        f_acqu = (m - self.exploration_weight * np.sqrt(s)) # mean(x) - beta*std(x); as scipy optimizers want to minimize, negative
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        x = np.atleast_2d(x)
        m, s = self.model.predict(x)
        dmdx, dsdx = self.model.predictive_gradients(x)

        f_acqu = (m - self.exploration_weight * np.sqrt(s))
        df_acqu = dmdx - self.exploration_weight * dsdx
        return f_acqu, df_acqu