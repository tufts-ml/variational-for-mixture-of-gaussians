Contains a "natural exponential family" form of a Gaussian mixture model with conjugate priors.

Develops variational inference to fit the model to data.

## Code

Class defining the Likelihood (in natural exp fam form) for a Normal with fixed variance:

Class defining the Prior (in natural exp fam form):

Script demonstrating posterior inference (given N observations, prints expectations under the posterior)


Script demonstrating coordinate ascent with variational objective (given N observations from a mixture, recover the q approximate posteriors over cluster means)



## Derivation

Writeup describing notation and updates:

https://www.overleaf.com/9129271598nkkxjpmbqjxc

See also Chapter 2 ("Background") from Mike's phd thesis document:

Michael C. Hughes
PhD Thesis
Brown University, 2016
https://cs.brown.edu/research/pubs/theses/phd/2016/hughes.michael.pdf#page=33


