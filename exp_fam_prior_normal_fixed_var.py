"""
Code to define an exponential family distribution formulation of the
conjugate prior for the normal distribution with fixed variance

Examples
--------
>>> D = 1
>>> v_D = 0.78 * np.ones(D)

>>> tau_D = 0.56 * np.ones(D)
>>> nu_1 = 0.34

# Create an ef distribution
>>> ef_dist = ExpFamPrior_NormalFixedVariance(nu_1, tau_D, v_D)

# Create some phi values linearly spaced between -5 and 5
>>> K = 8001
>>> phi_KD = np.linspace(-10, 10, K).reshape((K, D))

# Evaluate log pdf
>>> logpdf_N = ef_dist.calc_log_pdf_K(phi_KD)
>>> sum_of_pdf = np.trapz(np.exp(logpdf_N), phi_KD[:,0])

# Verify the sum is close to 1.0
>>> print("%.8f" % sum_of_pdf)
1.00000000

>>> np.allclose(sum_of_pdf, 1.0, atol=1e-8, rtol=0.0)
True
"""

import numpy as np
import matplotlib.pyplot as plt

from exp_fam_lik_normal_fixed_var import calc_likelihood_cumulant

class ExpFamPrior_NormalFixedVariance(object):

    ''' Exponential Family Conjugate Prior distribution for a fixed-var normal.

    Attributes
    ----------
    nu_1 : 1-dim pseudo-count parameter
    tau_D : D-dim shape parameter
    v_D : D-dim fixed variance 
    
    '''

    def __init__(self, nu_1, tau_D, v_D):
        self.nu_1 = np.asarray(nu_1, dtype=np.float64).reshape((1,))
        self.tau_D = np.asarray(tau_D)
        self.v_D = np.asarray(v_D)
        self.D = self.tau_D.size

    def calc_sufficent_statistics(
            self, phi_KD, return_tuple=False):
        K, D = phi_KD.shape
        negc_lik_K1 = np.asarray([
            -1 * calc_likelihood_cumulant(phi_k_D, self.v_D)
            for phi_k_D in phi_KD]).reshape((K, 1))

        if return_tuple:
            return phi_KD, negc_lik_K1
        else:
            return np.hstack([phi_KD, negc_lik_K1])

    def calc_cumulant(self):
        ''' Compute cumulant of this prior density

        Returns
        -------
        c : scalar
        '''
        return calc_prior_cumulant(self.nu_1, self.tau_D, self.v_D)

    def calc_log_pdf_K(self, phi_KD):
        ''' Compute log pdf of provided likelihood parameter values

        Returns
        -------
        logpdf_K : 1D array, shape (K,)
            Each entry is the pdf of row k of provided phi_KD
        '''
        sphi_KD, sc_K1 = self.calc_sufficent_statistics(
            phi_KD, return_tuple=True)
        return (
            np.dot(sphi_KD, self.tau_D)
            + np.dot(sc_K1, self.nu_1)
            - calc_prior_cumulant(self.nu_1, self.tau_D, self.v_D)
            )

    def calc_E_mu_D(self):
        ''' Compute expected mean parameter of the likelihood

        Returns
        -------
        E_mu_D : 1D array, size (D,)
        '''
        return self.tau_D / self.nu_1[0]

    def calc_E_phi_D(self):
        ''' Compute expected natural parameter of the likelihood

        Returns
        -------
        E_phi_D : 1D array, size (D,)
        '''
        return (self.v_D * self.tau_D) / self.nu_1[0]

    def calc_E_negative_c_of_phi(self):
        ''' Compute expected cumulant function of the likelihood

        $$
        \mathbb{E}_{\phi ~ ExpFamPrior} [ - c(\phi) ]
        $$

        Returns
        -------
        E_neg_c_of_phi : scalar float
        '''
        return (
            -0.5 / np.square(self.nu_1[0]) * np.sum(
                self.v_D * np.square(self.tau_D))
            -0.5 * self.D / self.nu_1[0]
            )

def calc_prior_cumulant(nu_1=None, tau_D=None, v_D=None):
    D = v_D.size
    return (
        + 0.5 * D * np.log(2 * np.pi)
        - 0.5 * D * np.log(nu_1[0])
        + 0.5 * np.sum(np.log(v_D))
        + 0.5 / nu_1[0] * np.sum(v_D * np.square(tau_D))
        )


if __name__ == '__main__':
    D = 1
    v_D = 0.78 * np.ones(D)

    tau_D = 0.56 * np.ones(D)
    nu_1 = 0.34

    # Create an ef distribution
    ef_dist = ExpFamPrior_NormalFixedVariance(nu_1, tau_D, v_D)

    # Create some phi values linearly spaced between -5 and 5
    K = 8001
    phi_KD = np.linspace(-10, 10, K).reshape((K, D))

    logpdf_N = ef_dist.calc_log_pdf_K(phi_KD)
    sum_of_pdf = np.trapz(np.exp(logpdf_N), phi_KD[:,0])

    plt.plot(phi_KD[:,0], np.exp(logpdf_N), 'k.-')
    print(sum_of_pdf)
    plt.show()

