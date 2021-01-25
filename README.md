Contains a "natural exponential family" form of a Gaussian mixture model with conjugate priors.

Develops variational inference to fit the model to data.

## Code

* [exp_fam_lik_normal_fixed_var.py](https://github.com/tufts-ml/variational-for-mixture-of-gaussians/blob/master/exp_fam_lik_normal_fixed_var.py)

Class defining the Likelihood (in natural exp fam form) for a Normal with fixed variance

Tests: Verified that this distribution integrates to 1 numerically (and thus defines a valid PDF).


* [exp_fam_prior_normal_fixed_var.py](https://github.com/tufts-ml/variational-for-mixture-of-gaussians/blob/master/exp_fam_prior_normal_fixed_var.py)

Class defining the conjugate Prior (in natural exp fam form)

Tests: Verified that this distribution integrates to 1 numerically (and thus defines a valid PDF).


* [run_posterior_update.py](https://github.com/tufts-ml/variational-for-mixture-of-gaussians/blob/master/run_posterior_update.py)

Script demonstrating posterior inference (given N observations, prints expectations under the posterior)

**Expected output**

```
True value of phi_D:
[ 1. -2.  3.]
Empirical mean of observed x_ND:
[ 0.9995 -1.9987  3.0003]

Posterior Expected value of phi_D after 1 observations:
[ 1.3528 -1.32    2.7957]
Posterior Expected value of phi_D after 10 observations:
[ 1.2319 -1.8301  2.9466]
Posterior Expected value of phi_D after 100 observations:
[ 1.0251 -1.966   2.964 ]
Posterior Expected value of phi_D after 1000 observations:
[ 0.9988 -2.008   2.9887]
Posterior Expected value of phi_D after 10000 observations:
[ 0.9981 -2.      2.9986]
Posterior Expected value of phi_D after 100000 observations:
[ 0.9995 -1.9987  3.0003]
```


* [train_mixture_via_variational_coord_ascent.py](https://github.com/tufts-ml/variational-for-mixture-of-gaussians/blob/master/train_mixture_via_variational_coord_ascent.py)

Script demonstrating coordinate ascent with variational objective (given N observations from a mixture, recover the q approximate posteriors over cluster means)

Will print the numerical value of the per-token ELBO (evidence lower bound objective, normalized by total number of observed scalars).

Will show that after each update (estep update of local assignment posteriors q(z), mstep update of global parameter posteriors q(\phi)), that the objective *monotonically* improves.

**Expected output**

```
   0/50 after estep   -1.598049
   0/50 after mstep   -0.178127
   1/50 after estep   -0.169840
   1/50 after mstep   -0.167906
   2/50 after estep   -0.167323
   2/50 after mstep   -0.167139
   3/50 after estep   -0.167080
   3/50 after mstep   -0.167061
   4/50 after estep   -0.167054
   4/50 after mstep   -0.167052
   5/50 after estep   -0.167052
   5/50 after mstep   -0.167052
   6/50 after estep   -0.167052
   6/50 after mstep   -0.167051
   7/50 after estep   -0.167051
   7/50 after mstep   -0.167051
   8/50 after estep   -0.167051
   8/50 after mstep   -0.167051
   9/50 after estep   -0.167051
   9/50 after mstep   -0.167051
  10/50 after estep   -0.167051
  10/50 after mstep   -0.167051
```

# Visualizations

**Expected Plot of ELBO vs iteration**

Should show monotonic improvement in the objective over 25 random initializations.

![Monotonic improvement in ELBO](https://github.com/tufts-ml/variational-for-mixture-of-gaussians/blob/master/monotonic_ELBO.png?raw=true)


# Derivation

Writeup describing notation and updates:

https://www.overleaf.com/9129271598nkkxjpmbqjxc

See also Chapter 2 ("Background") from Mike's phd thesis document:

Michael C. Hughes
PhD Thesis
Brown University, 2016
https://cs.brown.edu/research/pubs/theses/phd/2016/hughes.michael.pdf#page=33


