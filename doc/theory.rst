Stochastic Variational Bayes - Theory
=====================================

Stochastic Variational Bayes is a method of performing Bayesian inference on the parameters
of a generative model given a data set assumed to be generated by the model with 
unknown additive noise.

The description below is a highly abbreviated account of the theory behind SVB.
For more detailed derivations, see the references cited.

For interactive tutorials implementing variational Bayesian inference on simple
examples, see the `Variational Bayes tutorial <https://vb-tutorial.readthedocs.io>`_

Bayesian inference
------------------

Bayes' theorem is a general statement about how ones belief about the value distribution
of a random variable should be updated in the light of new data. In the context of
model parameter inference it can be described as follows:

.. math::

    p(\theta \mid y) = q(\theta) = \frac{p(y \mid \theta) \, p(\theta)}{p(y)}

Here :math:`\theta` is the set of parameters of the model which we wish to infer.

:math:`p(\theta \mid y) = q(\theta)` is the *posterior* distribution, i.e. the inferred 
distribution of the model parameters :math:`\theta` given the data :math:`y`.

:math:`p(\theta)` is the *prior* probability of the model parameters :math:`\theta`. This describes the
distribution we believe the parameters would follow before any data has been seen
and might reflect, for example, existing estimates of physiological parameters or other
constraints (e.g. that a fractional parameter must lie between 0 and 1).

:math:`p(y \mid \theta)` is the *likelihood*, i.e. the probability of getting the data :math:`y`
from a given set of model parameters :math:`\theta`. This is determined by evaluating the model
prediction using the parameters :math:`\theta` and comparing it to the data. The difference between
the two must be the result of noise, and the likelihood of the noise can be calculated
from the noise model.

:math:`p(y)` is the *evidence* and is chiefly used when comparing one model with another.
For an inference problem using a single model it can be neglected as it is independent
of the parameters and simply provides a normalizing constant.

Variational Bayes
-----------------

The general Bayesian inference problem can, in general, only be solved by a sampling
method such as Markov Chain Monte Carlo (MCMC) where random samples are generated in
such a way that, through Bayes' theorem, they gradually provide a representative 
sample of the posterior distribution. Any properties of the posterior, such as mean
and variance, can be calculated from the sample once it is large enough to be
representative.

MCMC, however, is extremely computationally intensive especially for the kind of 
applications we are concerned with where we may be fitting between 2 and 20 parameters
independently at typically :math:`10^5` voxels. Variational Bayes is an approximate
method which re-formulates the inference problem in the form of a variational
principle, where we seek to maximise the *Free Energy*.

.. math::

    F(\theta) = \int q(\theta)\log \bigg( p(y \mid \theta)\frac{p(\theta)}{q(\theta)} \bigg) d\theta
    
Again :math:`\theta` is the set of model parameters, :math:`q(\theta)` is the posterior
distribution, :math:`p(\theta)` is the prior distribution and :math:`p(y \mid \theta)`
is the likelihood of the data given the parameters.

For completely general forms of the prior and posterior distributions, this integral
is expensive to compute numerically (and is unlikely to be solvable analytically).
However the advantage of the variational approach is that simplified forms can be chosen for the
prior and posterior such that the free energy can be calculated and optimized 
efficiently. The variational principle guarantees that the free energy calculated
using this method will be a lower bound on the 'true' free energy and therefore the
closest approximation we can find using our simplified distributions.

Typically we assume multivariate Gaussian 
distributions for the prior and posterior, and a noise model based on a Gaussian or
Gamma distribution.

One form of variational Bayes uses the calculus of variations to derive a set of
update equations for the model and noise parameters which can then be iterated 
until convergence [1]_. However this method requires particular choices of the prior
and posterior distributions, and the noise model, and thus lacks flexibility.
Any change to these distributions requires the update equations to be 
re-derived.

Stochastic variational Bayes
----------------------------

The free energy equation can be slightly re-written in the form of an expectation over the
posterior distribution :math:`q(\theta)`:

.. math::

    F(\theta) = E_{q(\theta)} \big[ \log(p(y \mid \theta) \big] - E_{q(\theta)} \bigg[ \log \Big( \frac{q(\theta)}{p(\theta)} \Big) \bigg]

This suggests an alternative calculation method based on taking a *sample* of
values from the posterior distribution. If this sample is large enough to be 
representative of the distribution, the expectation integrals from above can be approximated
by the mean over the samples:

.. math::

    E_{q(\theta)} \big[ \log(p(y \mid \theta) \big] \approx \frac{1}{S} \sum_s \log(p(y \mid \theta^s)

    E_{q(\theta)} \bigg[ \log \Big( \frac{q(\theta)}{p(\theta)} \Big) \bigg] \approx \frac{1}{S} \sum_s \bigg[ \log \Big( \frac{q(\theta^s)}{p(\theta^s)} \Big) \bigg]

Where we have :math:`S` samples of the full set of parameters, denoted :math:`\theta^s`.

The first of these terms is the negative of the *reconstruction loss* and is a measure of
how well the model prediction fits the data.

The second term is the *latent loss* and measures the closeness of the posterior
to the prior. In fact it is the Kullback-Leibler (KL) divergence between the
prior and posterior distributions.

This is more tractable than a numerical integration *provided* we can obtain
a representative sample from the posterior. Maximisation of the free energy
can then be done using a generic framework such as those developed for machine
learning applications which have the ability to automatically calculate gradients
of an objective function from a defined set of calculation steps.

Alternative forms of the latent loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The latent loss term can be alternatively written as follows, removing the stochastic approximation
for part of the log:

.. math::

    E_{q(\theta)} \bigg[ \log \Big( \frac{q(\theta)}{p(\theta)} \Big) \bigg] \approx E_{q(\theta)} \bigg[ \log(q(\theta)) \bigg] - \frac{1}{S} \sum_s \bigg[ \log ( p(\theta^s) ) \bigg]

The first term is the *entropy* of the posterior distribution. For many distributions this 
can be calculated analytically without reference to a sample, so we may be able reduce our
dependence on the choice of sample to some degree.

If both the prior and posterior are multivariate Gaussian distributions, we can 
go further and obtain a fully analytic expression for the latent loss using the known
result for the KL divergence of two MVNs [2]_:

.. math::

    E_{q(\theta)} \bigg[ \log \Big( \frac{q(\theta)}{p(\theta)} \Big) \bigg] = \frac{1}{2} \bigg\{ \mathrm{Tr}(\Sigma_p^{-1} \Sigma_q) + (\mu_p - \mu_q)^T\Sigma_p^{-1}(\mu_p - \mu_q) - N + \log\bigg( \frac{\det \Sigma_p}{\det \Sigma_q} \bigg)  \bigg\}

Here :math:`N` is the number of parameters in :math:`\theta`, and 
:math:`\mu_p, \Sigma_p, \mu_q, \Sigma_q` are the mean and covariance of the 
prior and posterior.

Obtaining the sample from the posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The problem of sampling from the posterior is of some significance. If the 
optimization is to work effectively it would be helpful if the gradients
of the sample values with respect to the variable parameters could be 
calculated. However this is difficult if we simply obtain a random 
sample from, for example, a Gaussian of given mean and variance. For 
Gaussian distributions, one way around this is known as the *reparameterization 
trick*. We obtain a sample from a *fixed* Gaussian (e.g. :math:`N(0, 1)`) and
then scale the values using the (variable) mean and variance of the posterior
distribution. This enables the gradients to be used in the optimization 
algorithm. The disadvantage of the method is that it does not immediately
generalise to other kinds of distributions.

Advantages of the stochastic approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main advantage of the stochastic approach is that the requirements on
the prior and posterior distributions are greatly reduced. The prior
distribution needs to be able to generate log probabilities for a set of
parameters, the posterior needs to be able to generate samples and its
own entropy, and we need some means of calculating the data likelihood
- this normally involves a noise model which can calculate the
probability of the observed deviations between a model prediction
and the actual data. Although we can take advantage of analytic results for
Gaussian distribution, the actual forms of the distributions are not 
constrained by the method (apart from the limitation of not always being able to use
the reparameterization trick).

References
----------

.. [1] *Chappell, M.A., Groves, A.R., Woolrich, M.W., "Variational Bayesian
   inference for a non-linear forward model", IEEE Trans. Sig. Proc., 2009,
   57(1), 223???236.*

.. [2] http://web.stanford.edu/~jduchi/projects/general_notes.pdf

