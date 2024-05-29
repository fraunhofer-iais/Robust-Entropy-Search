# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# changes made by D. Weichert on January, 31st, 2024

import numpy as np
from scipy.optimize import bisect
from scipy.stats import norm

from Algorithms.base import AcquisitionBase

class MaxValueEntropySearch(AcquisitionBase):
    def __init__(
        self,
        model,
        space,
        num_samples=10,
        grid_size=5000,
    ) -> None:
        """
        MES acquisition function approximates the distribution of the value at the global
        minimum and tries to decrease its entropy. See this paper for more details:
        Z. Wang, S. Jegelka
        Max-value Entropy Search for Efficient Bayesian Optimization
        ICML 2017

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param num_samples: integer determining how many samples to draw of the minimum (does not need to be large)
        :param grid_size: number of random locations in grid used to fit the gumbel distribution and approximately generate
        the samples of the minimum (recommend scaling with problem dimension, i.e. 10000*d)
        """
        super(MaxValueEntropySearch, self).__init__(model, space)

        self.num_samples = num_samples
        self.grid_size = grid_size

        # Initialize parameters to lazily compute them once needed
        self.mins = None

        self.update_parameters()

    def update_parameters(self):
        """
        MES requires acces to a sample of possible minimum values y* of the objective function.
        To build this sample we approximate the empirical c.d.f of Pr(y*<y) with a Gumbel(a,b) distribution.
        This Gumbel distribution can then be easily sampled to yield approximate samples of y*

        This needs to be called once at the start of each BO step.
        """

        # First we generate a random grid of locations at which to fit the Gumbel distribution

        # get some random grid 
        if self.space[1]['type'] == 'continuous':
            grid = np.random.rand(self.grid_size, self.space[0]['dimensionality'] + self.space[1]['dimensionality'])
        else:
            grid_1 = np.random.rand(self.grid_size, self.space[0]['dimensionality'])
            grid_2 = np.random.choice(np.arange(self.space[1]['values'].shape[0]), (self.grid_size), replace=True)
            grid_2 = self.space[1]['values'][grid_2]
            grid = np.hstack((grid_1, grid_2))

        # also add the locations already queried in the previous BO steps
        grid = np.vstack([self.model.X, grid])
        # Get GP posterior at these points
        fmean, fvar = self.model.predict(grid)
        fsd = np.sqrt(fvar)

        # fit Gumbel distriubtion
        a, b = _fit_gumbel(fmean, fsd)

        # sample K times from this Gumbel distribution using the inverse probability integral transform,
        # i.e. given a sample r ~ Unif[0,1] then g = a + b * log( -1 * log(1 - r)) follows g ~ Gumbel(a,b).

        uniform_samples = np.random.rand(self.num_samples)
        gumbel_samples = np.log(-1 * np.log(1 - uniform_samples)) * b + a
        self.mins = gumbel_samples


    def _compute_acq(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the predicted change in entropy of p_min (the distribution
        of the minimal value of the objective function) if we evaluate x.
        :param x: points where the acquisition is evaluated.
        """
        x = np.atleast_2d(x)
        # Calculate GP posterior at candidate points
        fmean, fvar = self.model.predict(x)
        fsd = np.sqrt(fvar)
        # Clip below to improve numerical stability
        fsd = np.maximum(fsd, 1e-10)

        # standardise
        gamma = (self.mins - fmean) / fsd

        minus_cdf = 1 - norm.cdf(gamma)
        # Clip  to improve numerical stability
        minus_cdf = np.clip(minus_cdf, a_min=1e-10, a_max=1)

        # calculate monte-carlo estimate of information gain
        f_acqu_x = np.mean(-gamma * norm.pdf(gamma) / (2 * minus_cdf) - np.log(minus_cdf), axis=1)
        return -1*f_acqu_x.reshape(-1, 1)

def _fit_gumbel(fmean, fsd):
    """
    The Gumbel distribution for minimas has a cumulative density function of f(y)= 1 - exp(-1 * exp((y - a) / b)), i.e. the q^th quantile is given by
    Q(q) = a + b * log( -1 * log(1 - q)). We choose values for a and b that match the Gumbel's
    interquartile range with that of the observed empirical cumulative density function of Pr(y*<y)
    i.e.  Pr(y* < lower_quantile)=0.25 and Pr(y* < upper_quantile)=0.75.
    """

    def probf(x: np.ndarray) -> float:
        # Build empirical CDF function
        return 1 - np.exp(np.sum(norm.logcdf(-(x - fmean) / fsd), axis=0))

    # initialise end-points for binary search (the choice of 5 standard deviations ensures that these are outside the IQ range)
    left = np.min(fmean - 5 * fsd)
    right = np.max(fmean + 5 * fsd)

    def binary_search(val: float) -> float:
        return bisect(lambda x: probf(x) - val, left, right, maxiter=int(1e9))

    # Binary search for 3 percentiles
    lower_quantile, medium, upper_quantile = map(binary_search, [0.25, 0.5, 0.75])

    # solve for Gumbel scaling parameters
    b = (lower_quantile - upper_quantile) / (np.log(np.log(4.0 / 3.0)) - np.log(np.log(4.0)))
    a = medium - b * np.log(np.log(2.0))

    return a, b