"""
VABY_SVB: Experimental parameter priors
"""

import numpy as np
import tensorflow as tf

from vaby.utils import TF_DTYPE
from .prior import NormalPrior, ParameterPrior

class FabberMRFSpatialPrior(NormalPrior):
    """
    ParameterPrior designed to mimic the 'M' type spatial prior in Fabber.
    
    Note that this uses update equations for ak which is not in the spirit of the stochastic
    method. 'Native' SVB MRF spatial priors are also defined which simply treat the spatial
    precision parameter as an inference variable.

    This code has been verified to generate the same ak estimate given the same input as
    Fabber, however in practice it does not optimize to the same value. We don't yet know
    why.
    """

    def __init__(self, data_model, mean, var, idx=None, post=None, **kwargs):
        """
        :param mean: Tensor of shape [W] containing the prior mean at each parameter vertex
        :param var: Tensor of shape [W] containing the prior variance at each parameter vertex
        :param post: Posterior instance
        """
        NormalPrior.__init__(self, data_model, mean, var, **kwargs)

        # Save the original vertexwise mean and variance - the actual prior mean/var
        # will be calculated from these and also the spatial variation in neighbour nodes
        self.fixed_mean = self.mean
        self.fixed_var = self.var

        # Set up spatial smoothing parameter calculation from posterior and neighbour lists
        self._setup_ak(post, self.nn)

        # Set up prior mean/variance
        self._setup_mean_var(post, self.nn)

    def __str__(self):
        return "Spatial MRF prior (%f, %f)" % (self.scalar_mean, self.scalar_var)

    def _setup_ak(self, post, nn):
        # This is the equivalent of CalculateAk in Fabber
        #
        # Some of this could probably be better done using linalg
        # operations but bear in mind this is one parameter only

        self.sigmaK = tf.linalg.diag_part(post.cov)[:, self.idx] # [W]
        self.wK = post.mean[:, self.idx] # [W]
        self.num_nn = tf.sparse.reduce_sum(self.nn, axis=1) # [W]

        # Sum over nodes of parameter variance multiplied by number of 
        # nearest neighbours for each vertex
        trace_term = tf.reduce_sum(self.sigmaK * self.num_nn) # [1]

        # Sum of nearest and next-nearest neighbour mean values
        self.sum_means_nn = tf.reshape(tf.sparse.sparse_dense_matmul(self.nn, tf.reshape(self.wK, (-1, 1))), (-1,)) # [W]
        
        # vertex parameter mean multipled by number of nearest neighbours
        wknn = self.wK * self.num_nn # [W]

        swk = wknn - self.sum_means_nn # [W]

        term2 = tf.reduce_sum(swk * self.wK)# [1]

        scale, shape = 1.0, 10
        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 1/scale)
        hk = tf.multiply(tf.cast(self.nnodes, TF_DTYPE), 0.5) + shape
        self.ak = gk * hk

    def _setup_mean_var(self, post, nn):
        # This is the equivalent of ApplyToMVN in Fabber
        contrib_nn = 8*self.sum_means_nn # [W]
        
        spatial_mean = contrib_nn / (8*self.num_nn)
        spatial_prec = self.num_nn * self.ak

        self.var = 1 / (1/self.fixed_var + spatial_prec)
        #self.var = self.fixed_var
        self.mean = self.var * spatial_prec * spatial_mean
        #self.mean = self.fixed_mean + self.ak

class MRF2SpatialPrior(ParameterPrior):
    """
    ParameterPrior which performs adaptive spatial regularization based on the 
    contents of neighbouring nodes using the Markov Random Field method

    This uses the same formalism as the Fabber 'M' type spatial prior but treats the ak
    as a parameter of the optimization. It differs from MRFSpatialPrior by using the
    PDF formulation of the PDF rather than the matrix formulation (the two are equivalent
    but currently we keep both around for checking that they really are!)

    FIXME currently this does not work unless sample size=1
    """

    def __init__(self, data_model, mean, var, idx=None, post=None, nn=None, **kwargs):
        ParameterPrior.__init__(self, data_model)
        self.mean = tf.fill([self.nnodes], mean)
        self.var = tf.fill([self.nnodes], var)
        self.std = tf.sqrt(self.var)

        # We need the number of samples to implement the log PDF function
        self.sample_size = kwargs.get("sample_size", 5)

        # Set up spatial smoothing parameter calculation from posterior and neighbour lists
        self.log_ak = tf.Variable(-5.0, dtype=TF_DTYPE)
        self.ak = tf.exp(self.log_ak)
        self.vars = {"log_ak" : self.log_ak}

    def mean_log_pdf(self, samples):
        samples = tf.reshape(samples, (self.nnodes, -1)) # [W, N]
        self.num_nn = tf.sparse.reduce_sum(self.nn, axis=1) # [W]

        expanded_nn = tf.sparse.concat(2, [tf.sparse.reshape(self.nn, (self.nnodes, self.nnodes, 1))] * self.sample_size)
        xj = expanded_nn * tf.reshape(samples, (self.nnodes, 1, -1))
        #xi = tf.reshape(tf.sparse.to_dense(tf.sparse.reorder(self.nn)), (self.nnodes, self.nnodes, 1)) * tf.reshape(samples, (1, self.nnodes, -1))
        xi = expanded_nn * tf.reshape(samples, (1, self.nnodes, -1))
        #xi = tf.sparse.transpose(xj, perm=(1, 0, 2)) 
        neg_xi = tf.SparseTensor(xi.indices, -xi.values, dense_shape=xi.dense_shape )
        dx2 = tf.square(tf.sparse.add(xj, neg_xi))
        sdx = tf.sparse.reduce_sum(dx2, axis=0) # [W, N]
        term1 = tf.identity(0.5*self.log_ak)
        term2 = tf.identity(-self.ak * sdx / 4)
        log_pdf = term1 + term2  # [W, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.nnodes]) # [W]
        return mean_log_pdf

    def __str__(self):
        return "MRF2 spatial prior"

class ConstantMRFSpatialPrior(NormalPrior):
    """
    ParameterPrior which performs adaptive spatial regularization based on the 
    contents of neighbouring nodes using the Markov Random Field method

    This is equivalent to the Fabber 'M' type spatial prior
    """

    def __init__(self, data_model, mean, var, idx=None, **kwargs):
        """
        :param mean: Tensor of shape [W] containing the prior mean at each parameter vertex
        :param var: Tensor of shape [W] containing the prior variance at each parameter vertex
        :param post: Posterior instance
        """
        NormalPrior.__init__(self, data_model, mean, var, **kwargs)

        # Save the original vertexwise mean and variance - the actual prior mean/var
        # will be calculated from these and also the spatial variation in neighbour nodes
        self.fixed_mean = self.mean
        self.fixed_var = self.var
        self.vars = {}

    def __str__(self):
        return "Spatial MRF prior (%f, %f) - const" % (self.scalar_mean, self.scalar_var)

    def update_ak(self, post_mean, post_cov):
        # This is the equivalent of CalculateAk in Fabber
        #
        # Some of this could probably be better done using linalg
        # operations but bear in mind this is one parameter only

        self.sigmaK = post_cov[:, self.idx, self.idx] # [W]
        self.wK = post_mean[:, self.idx] # [W]
        self.num_nn = np.sum(self.nn, axis=1) # [W]

        # Sum over nodes of parameter variance multiplied by number of 
        # nearest neighbours for each vertex
        trace_term = np.sum(self.sigmaK * self.num_nn) # [1]

        # Sum of nearest and next-nearest neighbour mean values
        self.sum_means_nn = np.matmul(self.nn, np.reshape(self.wK, (-1, 1))) # [W]
        
        # vertex parameter mean multipled by number of nearest neighbours
        wknn = self.wK * self.num_nn # [W]

        swk = wknn - self.sum_means_nn # [W]

        term2 = np.sum(swk * self.wK) # [1]

        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 0.1)
        hk = float(self.nnodes) * 0.5 + 1.0
        self.ak = gk * hk
        self.log.debug("MRF2: ak=%f", self.ak)

    def _setup_mean_var(self, post_mean, post_cov):
        # This is the equivalent of ApplyToMVN in Fabber
        contrib_nn = 8*self.sum_means_nn # [W]
        
        spatial_mean = contrib_nn / (8*self.num_nn)
        spatial_prec = self.num_nn * self.ak

        self.var = 1 / (1/self.fixed_var + spatial_prec)
        #self.var = self.fixed_var
        self.mean = self.var * spatial_prec * spatial_mean
        #self.mean = self.fixed_mean + self.ak
