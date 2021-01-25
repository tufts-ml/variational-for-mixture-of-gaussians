import numpy as np
import matplotlib.pyplot as plt

from exp_fam_lik_normal_fixed_var import (
    calc_likelihood_sufficient_statistics_ND,
    calc_likelihood_reference_measure_N)

from exp_fam_prior_normal_fixed_var import (
    ExpFamPrior_NormalFixedVariance, calc_prior_cumulant)

from scipy.special import logsumexp

def calc_ELBO_per_token(h_N, sx_ND, resp_NK, qdist_list, prior_dist, after_mstep=True):
    _, K = resp_NK.shape
    N, D = sx_ND.shape

    stats_dict = estep__update_summary_stats_dict(sx_ND, resp_NK)

    unif_pi_K = 1.0/K * np.ones(K)
    L_alloc = (
        np.dot(stats_dict['N_K'], unif_pi_K)
        )
    L_entropy = np.sum(stats_dict['H_K'])
    L_obs = (
        np.sum(h_N)
        - K * prior_dist.calc_cumulant()
        + np.sum([q_dist_k.calc_cumulant() for q_dist_k in qdist_list])
        )
    if not after_mstep:
        for k in range(K):
            nu_diff = stats_dict['N_K'][k] + prior_dist.nu_1[0] - qdist_list[k].nu_1[0]
            tau_diff_D = stats_dict['S_KD'][k] + prior_dist.tau_D - qdist_list[k].tau_D

            L_obs += nu_diff * qdist_list[k].calc_E_negative_c_of_phi()
            L_obs += np.dot(tau_diff_D, qdist_list[k].calc_E_phi_D())

    assert np.isscalar(L_alloc)
    assert np.isscalar(L_obs)
    return (L_alloc + L_entropy + L_obs) / (N * D)


def estep__update_resp_NK(sx_ND, qdist_list):

    logpdf_x_NK = np.zeros((N, K))
    for k in range(K):
        E_phi_k_D = qdist_list[k].calc_E_phi_D()
        E_negative_cumulant_k = qdist_list[k].calc_E_negative_c_of_phi()
        logpdf_x_NK[:,k] = np.dot(sx_ND, E_phi_k_D) + E_negative_cumulant_k

    logpdf_x_NK -= logsumexp(logpdf_x_NK, axis=1)[:,np.newaxis]
    resp_NK = logpdf_x_NK
    np.exp(resp_NK, out=resp_NK)
    assert np.allclose(np.sum(resp_NK, axis=1), 1.0)
    return resp_NK

def estep__update_summary_stats_dict(sx_ND, resp_NK, stats_dict=None):
    if stats_dict is None:
        stats_dict = dict()
    stats_dict['N_K'] = np.sum(resp_NK, axis=0)
    stats_dict['S_KD'] = np.dot(resp_NK.T, sx_ND)
    stats_dict['H_K'] = -1 * np.sum(resp_NK * np.log(1e-100 + resp_NK), axis=0)
    return stats_dict

def mstep__update_q(stats_dict, prior_dist):
    qdist_list = list()
    for k in range(K):
        qdist_k = ExpFamPrior_NormalFixedVariance(
            tau_D=stats_dict['S_KD'][k] + prior_dist.tau_D,
            nu_1=stats_dict['N_K'][k] + prior_dist.nu_1,
            v_D=prior_dist.v_D,
            )
        qdist_list.append(qdist_k)
    return qdist_list


def run_coordinate_ascent(data_ND, prior_dist,
        init_resp_NK=None, K=10,
        n_iters=100, random_state=0):
    N, D = data_ND.shape

    ## Randomly assign a few data points to given clusters K
    if init_resp_NK is None:
        prng = np.random.RandomState(random_state)
        chosen_ids_K = prng.choice(np.arange(N), K)

        init_resp_NK = np.zeros((N,K))
        for k in range(K):
            init_resp_NK[chosen_ids_K[k], :] = 0.01
            init_resp_NK[chosen_ids_K[k], k] = 1.00 - 0.01 * (K-1)

    h_N = calc_likelihood_reference_measure_N(data_ND, prior_dist.v_D)
    sx_ND = calc_likelihood_sufficient_statistics_ND(data_ND, prior_dist.v_D)

    # Initialize other parameters just by running updates from the given resp
    stats_dict = estep__update_summary_stats_dict(sx_ND, init_resp_NK)
    qdist_list = mstep__update_q(stats_dict, prior_dist)    

    elbo_history = list()

    for iterid in range(n_iters):

        resp_NK = estep__update_resp_NK(sx_ND, qdist_list)
        elbo = calc_ELBO_per_token(
            h_N, sx_ND, resp_NK, qdist_list, prior_dist, after_mstep=False)
        elbo_history.append(elbo)
        print("%4d/%d after estep % 11.6f" % (iterid, n_iters, elbo))
        if len(elbo_history) >= 2:
            assert (elbo_history[-1] - elbo_history[-2]) >= -1e-8

        stats_dict = estep__update_summary_stats_dict(sx_ND, resp_NK, stats_dict)
        qdist_list = mstep__update_q(stats_dict, prior_dist)
        elbo = calc_ELBO_per_token(
            h_N, sx_ND, resp_NK, qdist_list, prior_dist, after_mstep=True)
        elbo_history.append(elbo)
        print("%4d/%d after mstep % 11.6f" % (iterid, n_iters, elbo))

        if len(elbo_history) >= 2:
            assert (elbo_history[-1] - elbo_history[-2]) >= -1e-8

    return (resp_NK, stats_dict, qdist_list), elbo_history


if __name__ == '__main__':
    prng = np.random.RandomState(0)

    K = 3
    D = 2
    true_v_D = np.asarray([0.15, 0.3])

    os_KD = prng.choice(np.asarray([-1.0, +1.0]), size=K*D, replace=True).reshape((K,D))
    true_phi_KD = np.asarray([1.337, -2.0, 3.0]).reshape((K, 1)) * os_KD

    N = 3000
    data_list = list()
    for k in range(K):
        data_k_ND = (
            true_phi_KD[k][np.newaxis,:]
            + true_v_D[np.newaxis,:] * prng.randn(N//K, D))
        data_list.append(data_k_ND)
    data_ND = np.vstack(data_list)

    prior_dist = ExpFamPrior_NormalFixedVariance(
        nu_1=0.25, tau_D=1.0 * np.ones(D), v_D=true_v_D)

    elbo_history_list = list()
    config_list = list()
    for run in range(25):
        final_config, elbo_history = run_coordinate_ascent(
            data_ND, prior_dist,
            K=K,
            n_iters=50, random_state=np.random.randint(100))
        elbo_history_list.append(elbo_history)
        config_list.append(final_config)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    elbo_SR = np.vstack(elbo_history_list).T
    ax.plot(elbo_SR)

    ymax = elbo_SR.max()
    ymin = elbo_SR[-1,:].min()
    R = ymax - ymin
    ax.set_ylim(ymin - R, ymax + 0.1 * R)
    ax.set_xlabel('number of update iterations')
    ax.set_ylabel('log p(x) per token')

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    for row in range(5):
        for col in range(5):
            ax[row, col].plot(data_ND[:,0], data_ND[:,1], 'k.', alpha=0.1)

            k = 5 * row + col
            _, _, q_list = config_list[k]
            for k in range(K):
                E_phi_k_D = q_list[k].calc_E_phi_D()
                ax[row, col].plot(E_phi_k_D[0], E_phi_k_D[1], '+')

    plt.show()
