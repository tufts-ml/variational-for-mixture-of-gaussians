import numpy as np

from exp_fam_prior_normal_fixed_var import ExpFamPrior_NormalFixedVariance
from exp_fam_lik_normal_fixed_var import calc_likelihood_sufficient_statistics_ND

if __name__ == '__main__':

    D = 3
    true_v_D = 0.25 * np.ones(D)
    true_phi_D = np.asarray([1.0, -2.0, 3.0])

    N = 100000
    prng = np.random.RandomState(0)
    x_ND = true_phi_D[np.newaxis,:] + true_v_D[np.newaxis,:] * prng.randn(N, D)

    # Prior
    prior = dict(
        tau_D=1.0 * np.ones(D),
        nu_1=0.25)

    ef_prior = ExpFamPrior_NormalFixedVariance(
        prior['nu_1'], prior['tau_D'], true_v_D)

    sx_ND = calc_likelihood_sufficient_statistics_ND(x_ND, true_v_D)

    print("True value of phi_D:")
    print(np.array2string(true_phi_D, precision=4))

    print("Empirical mean of observed x_ND:")
    print(np.array2string(np.mean(x_ND, axis=0), precision=4))

    for a in np.arange(0, 1+np.log10(N), 1):
        n = int(10**a)

        posterior = dict(
            tau_D=prior['tau_D'] + np.sum(sx_ND[:n], axis=0),
            nu_1=prior['nu_1'] + n)
        ef_post = ExpFamPrior_NormalFixedVariance(
            posterior['nu_1'], posterior['tau_D'], true_v_D)

        print("Expected value of phi_D after %d observations:" % n)
        E_phi_D = ef_post.calc_E_phi_D()
        print(np.array2string(E_phi_D, precision=4))


