"""
Code to define an exponential family distribution formulation of the
normal distribution with unknown mean and fixed variance.


Examples
--------
>>> D = 1
>>> phi_D = np.ones(1)
>>> v_D = np.ones(1)
>>> ef_dist = ExpFamLikelihood_NormalFixedVariance(phi_D, v_D)

# Create N data points linearly spaced between -5 and 5
>>> N = 5001
>>> x_ND = np.linspace(-8, 8, N).reshape((N, D))

# Evaluate log pdf at these N points
>>> logpdf_N = ef_dist.calc_log_pdf_N(x_ND)
>>> sum_of_pdf = np.trapz(np.exp(logpdf_N), x_ND[:,0])

# Verify the sum is close to 1.0
>>> print("%.8f" % sum_of_pdf)
1.00000000

>>> np.allclose(sum_of_pdf, 1.0, atol=1e-8, rtol=0.0)
True
"""

import numpy as np
import matplotlib.pyplot as plt

class ExpFamLikelihood_NormalFixedVariance(object):

    ''' Exponential Family Likelihood distribution for a fixed-variance normal.

    Attributes
    ----------
    phi_D : D-dim natural parameter
    v_D : D-dim fixed variance 
    
    '''

    def __init__(self, phi_D, v_D):
        self.phi_D = np.asarray(phi_D)
        self.v_D = np.asarray(v_D)
        self.D = self.phi_D.size

    def calc_cumulant(self):
        ''' Compute cumulant of this likelihood density

        Returns
        -------
        c : scalar
        '''
        return calc_likelihood_cumulant(self.phi_D, self.v_D)

    def calc_log_pdf_N(self, data_ND):
        ''' Compute log pdf at provided data

        Returns
        -------
        logpdf_N : 1D array, shape (N,)
            Each entry is the pdf of row n in data_ND
        '''
        sx_ND = calc_likelihood_sufficient_statistics_ND(data_ND, self.v_D)
        return (
            np.dot(sx_ND, self.phi_D)
            - self.calc_cumulant()
            + calc_likelihood_reference_measure_N(data_ND, self.v_D)
            )

def calc_likelihood_reference_measure_N(data_ND, v_D):
        ''' Compute reference measure of this likelihood density

        Returns
        -------
        h_N : 1D array, shape (N, )
        '''
        D = v_D.size
        return (
            -0.5 * D * np.log(2 * np.pi)
            -0.5 * np.sum(np.log(v_D))
            -0.5 * np.dot(np.square(data_ND), 1.0 / v_D)
            )

def calc_likelihood_sufficient_statistics_ND(data_ND, v_D):
    ''' Compute statistics for observed dataset

    Returns
    -------
    s_ND : 2D array, shape (N, D) 
    '''
    return data_ND / v_D[np.newaxis,:]

def calc_likelihood_cumulant(phi_D, v_D):
    ''' Compute cumulant of this likelihood density

    Returns
    -------
    c : scalar
    '''
    return 0.5 * np.sum(np.square(phi_D) / v_D)


if __name__ == '__main__':
    D = 1
    phi_D = np.zeros(D)
    v_D = np.ones(D)

    # Create an ef distribution
    ef_dist = ExpFamLikelihood_NormalFixedVariance(phi_D, v_D)

    # Create some data linearly spaced between -5 and 5
    N = 5001
    x_ND = np.linspace(-8, 8, N).reshape((N, D))

    logpdf_N = ef_dist.calc_log_pdf_N(x_ND)
    sum_of_pdf = np.trapz(np.exp(logpdf_N), x_ND[:,0])

    plt.plot(x_ND[:,0], np.exp(logpdf_N), 'k.-')
    plt.show()
    print(sum_of_pdf)

