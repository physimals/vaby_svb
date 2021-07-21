"""
VABY_AVB - Definition of the posterior distribution
"""
import tensorflow as tf

import numpy as np

from vaby.utils import LogBase, TF_DTYPE
from vaby import dist

def tensor_array_shape(obj, dim):
    if isinstance(obj, np.ndarray):
        return obj.shape[dim]
    else:
        return obj.shape.as_list()[dim]

def get_posterior(idx, param, t, data_model, **kwargs):
    """
    Factory method to return a posterior

    :param param: svb.parameter.Parameter instance
    :
    """
    is_global = False # FIXME previously part of Parameter
    initial_mean, initial_var = None, None
    if param.post_init is not None:
        initial_mean, initial_var = param.post_init(param, t, data_model.data_flat)

        # If parameter defined on surface, project the volume init values
        # onto the surface
        if ((isinstance(initial_mean, (np.ndarray, tf.Tensor))) 
                and (tensor_array_shape(initial_mean, 0) 
                    == data_model.n_voxels)):
            initial_mean = tf.squeeze(data_model.voxels_to_nodes(
                tf.expand_dims(initial_mean, -1), False))
        if ((isinstance(initial_var, (np.ndarray, tf.Tensor))) 
                and (tensor_array_shape(initial_var, 0) 
                    == data_model.n_voxels)):
            initial_var = tf.squeeze(data_model.voxels_to_nodes(
                tf.expand_dims(initial_var, -1)))

    # The size of the posterior (number of positions at which it is 
    # estimated) is determined by the data_space it refers to, and 
    # in turn by the data model. If it is global, the reduction will
    # be applied when creating the GaussianGlobalPosterior later on 
    post_size = data_model.n_nodes

    if initial_mean is None:
        initial_mean = tf.fill([post_size], param.post_dist.mean)
    else:
        initial_mean = param.post_dist.transform.int_values(initial_mean)

    if initial_var is None:
        initial_var = tf.fill([post_size], param.post_dist.var)
    else:
        # FIXME variance not value?
        initial_var = param.post_dist.transform.int_values(initial_var)

    if (not is_global) and isinstance(param.post_dist, dist.Normal):
        return NormalPosterior(idx, initial_mean, initial_var, name=param.name, **kwargs)
    
    if is_global and isinstance(param.post_dist, dist.Normal):
        return GaussianGlobalPosterior(idx, initial_mean, initial_var, name=param.name, **kwargs)

    raise ValueError("Can't create (global: %s) posterior for distribution: %s" 
        % (is_global, param.post_dist))
        
class Posterior(LogBase):
    """
    Posterior distribution
    """
    def __init__(self, idx, **kwargs):
        LogBase.__init__(self, **kwargs)
        self._idx = idx

    def build(self):
        """
        Define tensors that depend on Variables in the posterior
        
        Only constant tensors and Variables should be defined in the constructor.
        """
        pass

    def _get_mean_var(self, mean, var, init_post):
        if init_post is not None:
            mean, cov = init_post
            #if mean.shape[0] != self.nnodes:
            #    raise ValueError("Initializing posterior with %i nodes but input contains %i nodes" % (self.nnodes, mean.shape[0]))
            if self._idx >= mean.shape[1]:
                raise ValueError("Initializing posterior for parameter %i but input contains %i parameters" % (self._idx+1, mean.shape[1]))
            
            # We have been provided with an initialization posterior. Extract the mean and diagonal of the
            # covariance and use that as the initial values of the mean and variance. Note that the covariance
            # initialization is only used if this parameter is embedded in an MVN
            mean = mean[:, self._idx]
            var = cov[:, self._idx, self._idx]
            self.log.info(" - Initializing posterior mean and variance from input posterior")
            self.log.info("     means=%s", np.mean(mean))
            self.log.info("     vars=%s", np.mean(var))
        return mean, var

    def sample(self, nsamples):
        """
        :param nsamples: Number of samples to return per parameter vertex / parameter

        :return: A tensor of shape [W, P, S] where W is the number
                 of parameter nodes, P is the number of parameters in the distribution
                 (possibly 1) and S is the number of samples
        """
        raise NotImplementedError()

    def entropy(self, samples=None):
        """
        :param samples: A tensor of shape [W, P, S] where W is the number
                        of parameter nodes, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples.
                        This parameter may or may not be used in the calculation.
                        If it is required, the implementation class must check
                        that it is provided

        :return Tensor of shape [W] containing vertexwise distribution entropy
        """
        raise NotImplementedError()

    def state(self):
        """
        :return Sequence of tf.Tensor objects containing the state of all variables in this
                posterior. The tensors returned will be evaluated to create a savable state
                which may then be passed back into set_state()
        """
        raise NotImplementedError()

    def set_state(self, state):
        """
        :param state: State of variables in this posterior, as returned by previous call to state()

        :return Sequence of tf.Operation objects containing which will set the variables in
                this posterior to the specified state
        """
        raise NotImplementedError()

    def log_det_cov(self):
        raise NotImplementedError()

    @property
    def is_gaussian(self):
        return isinstance(self, NormalPosterior)

class NormalPosterior(Posterior):
    """
    Posterior distribution for a single vertexwise parameter with a normal
    distribution
    """

    def __init__(self, idx, mean, var, **kwargs):
        """
        :param mean: Tensor of shape [W] containing the initial mean at each parameter vertex
        :param var: Tensor of shape [W] containing the initial variance at each parameter vertex
        """
        Posterior.__init__(self, idx, **kwargs)
        self.nnodes = tf.shape(mean)[0]
        self.name = kwargs.get("name", "NormPost")
        self.suppress_nan = kwargs.get("suppress_nan", True)

        mean, var = self._get_mean_var(mean, var, kwargs.get("init", None))
        mean = tf.cast(mean, TF_DTYPE)
        var = tf.cast(var, TF_DTYPE)
        self.mean_init = tf.where(tf.math.is_finite(mean), mean, tf.zeros_like(mean))
        self.var_init = tf.where(tf.math.is_nan(var), tf.ones_like(var), var)

        self.mean_variable = tf.Variable(mean, validate_shape=False)
        self.log_var = tf.Variable(tf.math.log(var), validate_shape=False)

    def build(self):
        self.var_variable = tf.exp(self.log_var)
        if self.suppress_nan:
            #self.mean = tf.where(tf.math.is_nan(self.mean_variable), tf.ones_like(self.mean_variable), self.mean_variable)
            #self.var = tf.where(tf.math.is_nan(self.var_variable), tf.ones_like(self.var_variable), self.var_variable)
            self.mean = tf.where(tf.math.is_nan(self.mean_variable), self.mean_init, self.mean_variable)
            self.var = tf.where(tf.math.is_nan(self.var_variable), self.var_init, self.var_variable)
        else:
            self.mean = self.mean_variable
            self.var = self.var_variable
        self.std = tf.sqrt(self.var)

    def sample(self, nsamples):
        eps = tf.random.normal((self.nnodes, 1, nsamples), 0, 1, dtype=TF_DTYPE)
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nnodes, 1, 1]), [1, 1, nsamples])
        sample = tf.add(tiled_mean, tf.multiply(tf.reshape(self.std, [self.nnodes, 1, 1]), eps))
        return sample

    def entropy(self, _samples=None):
        entropy = tf.identity(-0.5 * tf.math.log(self.var), name="%s_entropy" % self.name)
        return entropy

    def state(self):
        return [self.mean, self.log_var]

    def set_state(self, state):
        return [
            self.mean_variable.assign(state[0]),
            self.log_var.assign(state[1])
        ]

    def __str__(self):
        return f"posterior"

class GaussianGlobalPosterior(Posterior):
    """
    Posterior which has the same value at every parameter vertex
    """

    def __init__(self, idx, mean, var, **kwargs):
        """
        :param mean: Tensor of shape [W] containing the mean at each parameter vertex
        :param var: Tensor of shape [W] containing the variance at each parameter vertex
        """
        Posterior.__init__(self, idx, **kwargs)
        self.nnodes = tf.shape(mean)[0]
        self.name = kwargs.get("name", "GlobalPost")
        self.suppress_nan = kwargs.get("suppress_nan", True)

        mean, var = self._get_mean_var(mean, var, kwargs.get("init", None))
        
        # Take the mean of the mean and variance across nodes as the initial value
        # in case there is a vertexwise initialization function
        self.initial_mean_global = tf.reshape(tf.reduce_mean(mean), [1])
        self.initial_var_global = tf.reshape(tf.reduce_mean(var), [1])
        self.mean_variable = tf.Variable(self.initial_mean_global, 
                                         dtype=TF_DTYPE, validate_shape=False,
                                         name="%s_mean" % self.name)
        self.log_var = tf.Variable(tf.math.log(tf.cast(self.initial_var_global, TF_DTYPE)), validate_shape=False,
                                   name="%s_log_var" % self.name)

    def build(self):
        self.var_variable = tf.exp(self.log_var)
        if self.suppress_nan:
            self.mean_global = tf.where(tf.math.is_nan(self.mean_variable), self.initial_mean_global, self.mean_variable)
            self.var_global = tf.where(tf.math.is_nan(self.var_variable), self.initial_var_global, self.var_variable)
        else:
            self.mean_global = self.mean_variable
            self.var_global = self.var_variable

        self.mean = tf.tile(self.mean_global, [self.nnodes])
        self.var = tf.tile(self.var_global, [self.nnodes])
        self.std = tf.sqrt(self.var)

    def sample(self, nsamples):
        """
        FIXME should each parameter vertex get the same sample? Currently YES
        """
        eps = tf.random.normal((1, 1, nsamples), 0, 1, dtype=TF_DTYPE)
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nnodes, 1, 1]), [1, 1, nsamples])
        sample = tf.add(tiled_mean, tf.multiply(tf.reshape(self.std, [self.nnodes, 1, 1]), eps))
        return sample

    def entropy(self, _samples=None):
        entropy = tf.identity(-0.5 * tf.math.log(self.var), name="%s_entropy" % self.name)
        return entropy

    def state(self):
        return [self.mean_global, self.log_var]

    def set_state(self, state):
        return [
            self.mean_variable.assign(state[0]),
            self.log_var.assign(state[1])
        ]

    def __str__(self):
        return "Global posterior"

class FactorisedPosterior(Posterior):
    """
    Posterior distribution for a set of parameters with no covariance
    """

    def __init__(self, posts, **kwargs):
        Posterior.__init__(self, -1, **kwargs)
        self.posts = posts
        self.nparams = len(self.posts)
        self.name = kwargs.get("name", "FactPost")
        self.nnodes = posts[0].nnodes

        # Regularisation to make sure cov is invertible. Note that we do not
        # need this for a diagonal covariance matrix but it is useful for
        # the full MVN covariance which shares some of the calculations
        self.cov_reg = 1e-5*tf.eye(self.nparams)

    def build(self):
        for post in self.posts:
            post.build()

        means = [post.mean for post in self.posts]
        variances = [post.var for post in self.posts]
        self.mean = tf.stack(means, axis=-1, name="%s_mean" % self.name)
        self.var = tf.stack(variances, axis=-1, name="%s_var" % self.name)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

        # Covariance matrix is diagonal
        self.cov = tf.linalg.diag(self.var, name='%s_cov' % self.name)

    def sample(self, nsamples):
        samples = [post.sample(nsamples) for post in self.posts]
        sample = tf.concat(samples, 1)
        return sample

    def entropy(self, _samples=None):
        entropy = tf.zeros([self.nnodes], dtype=TF_DTYPE)
        for post in self.posts:
            entropy = tf.add(entropy, post.entropy(), name="%s_entropy" % self.name)
        return entropy

    def state(self):
        state = []
        for post in self.posts:
            state.extend(post.state())
        return state

    def set_state(self, state):
        ops = []
        for idx, post in enumerate(self.posts):
            ops += post.set_state(state[idx*2:idx*2+2])
        return ops

    def log_det_cov(self):
        """
        Determinant of diagonal matrix is product of diagonal entries
        """
        return tf.reduce_sum(tf.math.log(self.var), axis=1, name='%s_log_det_cov' % self.name)

    def latent_loss(self, prior):
        """
        Analytic expression for latent loss which can be used when posterior and prior are
        Gaussian

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence

        :param prior: Vertexwise Prior instance which defines the ``mean`` and ``cov`` nodes
                      attributes
        """
        prior_cov_inv = tf.linalg.inv(prior.cov)
        mean_diff = tf.subtract(self.mean, prior.mean)

        term1 = tf.linalg.trace(tf.matmul(prior_cov_inv, self.cov))
        term2 = tf.matmul(tf.reshape(mean_diff, (self.nnodes, 1, -1)), prior_cov_inv)
        term3 = tf.reshape(tf.matmul(term2, tf.reshape(mean_diff, (self.nnodes, -1, 1))), [self.nnodes])
        term4 = prior.log_det_cov()
        term5 = self.log_det_cov()

        return 0.5*(term1 + term3 - self.nparams + term4 - term5)

class MVNPosterior(FactorisedPosterior):
    """
    Multivariate Normal posterior distribution
    """

    def __init__(self, posts, **kwargs):
        FactorisedPosterior.__init__(self, posts, **kwargs)
        self.suppress_nan = kwargs.get("suppress_nan", True)

        # The full covariance matrix is formed from the Cholesky decomposition
        # to ensure that it remains positive definite.
        #
        # To achieve this, we have to create PxP tensor variables for
        # each parameter vertex, but we then extract only the lower triangular
        # elements and train only on these. The diagonal elements
        # are constructed by the FactorisedPosterior
        if kwargs.get("init", None):
            # We are initializing from an existing posterior.
            # The FactorizedPosterior will already have extracted the mean and
            # diagonal of the covariance matrix - we need the Cholesky decomposition
            # of the covariance to initialize the off-diagonal terms
            self.log.info(" - Initializing posterior covariance from input posterior")
            _mean, cov = kwargs["init"]
            covar_init = tf.linalg.cholesky(cov)
        else:
            covar_init = tf.zeros([self.nnodes, self.nparams, self.nparams], dtype=TF_DTYPE)

        self.off_diag_vars_base = tf.Variable(covar_init, validate_shape=False)

    def build(self):
        FactorisedPosterior.build(self)

        if self.suppress_nan:
            self.off_diag_vars = tf.where(tf.math.is_nan(self.off_diag_vars_base), tf.zeros_like(self.off_diag_vars_base), self.off_diag_vars_base)
        else:
            self.off_diag_vars = self.off_diag_vars_base
        self.off_diag_cov_chol = tf.linalg.set_diag(tf.linalg.band_part(self.off_diag_vars, -1, 0),
                                                    tf.zeros([self.nnodes, self.nparams]),
                                                    name='%s_off_diag_cov_chol' % self.name)

        # Combine diagonal and off-diagonal elements into full matrix
        self.cov_chol = tf.add(tf.linalg.diag(self.std), self.off_diag_cov_chol,
                               name='%s_cov_chol' % self.name)

        # Form the covariance matrix from the chol decomposition
        self.cov = tf.matmul(tf.transpose(self.cov_chol, perm=(0, 2, 1)), self.cov_chol,
                             name='%s_cov' % self.name)

    def log_det_cov(self):
        """
        Determinant of a matrix can be calculated from the Cholesky decomposition which may
        be faster and more stable than tf.linalg.matrix_determinant
        """
        return tf.multiply(2.0, tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.cov_chol)), axis=1))

    def sample(self, nsamples):
        # Use the 'reparameterization trick' to return the samples
        eps = tf.random.normal((self.nnodes, self.nparams, nsamples), 0, 1, dtype=TF_DTYPE, name="eps")

        # NB self.cov_chol is the Cholesky decomposition of the covariance matrix
        # so plays the role of the std.dev.
        tiled_mean = tf.tile(tf.reshape(self.mean, [self.nnodes, self.nparams, 1]),
                             [1, 1, nsamples])
        sample = tf.add(tiled_mean, tf.matmul(self.cov_chol, eps), name="%s_sample" % self.name)
        return sample

    def entropy(self, _samples=None):
        entropy = tf.identity(-0.5 * self.log_det_cov(), name="%s_entropy" % self.name)
        return entropy

    def state(self):
        return list(FactorisedPosterior.state(self)) + [self.off_diag_vars]

    def set_state(self, state):
        ops = list(FactorisedPosterior.set_state(self, state[:-1]))
        ops += [self.off_diag_vars_base.assign(state[-1], validate_shape=False)]
        return ops
