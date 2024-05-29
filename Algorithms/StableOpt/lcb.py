from Algorithms.base import AcquisitionBase
import numpy as np

class AcquisitionLCB(AcquisitionBase):
    """
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, exploration_weight=2):
        super(AcquisitionLCB, self).__init__(model, space)
        self.exploration_weight = exploration_weight

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound
        """
        m, s = self.model.predict(x)
        f_acqu = (m - self.exploration_weight * np.sqrt(s)) # mean(x) - beta*std(x); as scipy optimizers want to minimize, negative
        return -f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        x = np.atleast_2d(x)
        m, s = self.model.predict(x)
        dmdx, dsdx = self.model.predictive_gradients(x)

        f_acqu = (m - self.exploration_weight * np.sqrt(s))
        df_acqu = dmdx - self.exploration_weight * dsdx
        return -f_acqu, -df_acqu