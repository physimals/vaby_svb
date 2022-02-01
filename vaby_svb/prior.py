"""
VABY_SVB: Parameter priors
"""

import numpy as np
import tensorflow as tf

from vaby import DataModel
from vaby.utils import LogBase, scipy_to_tf_sparse, TF_DTYPE, NP_DTYPE
from vaby.dist import Normal

def get_prior(idx, param, data_model, data_space=DataModel.MODEL_SPACE, **kwargs):
    """
    Factory method to return a vertexwise prior
    """
    prior = None
    if isinstance(param.prior_dist, Normal):
        if data_space == DataModel.MODEL_SPACE:
            nnodes = data_model.model_space.size
        else:
            nnodes = data_model.data_space.size

        if param.prior_type == "N":
            prior = NormalPrior(data_model, idx, param.prior_dist, nnodes=nnodes, **kwargs)
        elif param.prior_type == "M":
            prior = MRFSpatialPrior(data_model, idx, param.prior_dist, **kwargs)
        #elif param.prior_type == "M2":
        #    prior = MRF2SpatialPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        #elif param.prior_type == "Mfab":
        #    prior = FabberMRFSpatialPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "A":
            prior = ARDPrior(data_model, idx, param.prior_dist, **kwargs)

    if prior is not None:
        return prior
    else:
        raise ValueError("Can't create prior type %s for distribution %s - unrecognized combination" % (param.prior_type, param.prior_dist))

class ParameterPrior(LogBase):
    """
    Base class for a prior

    All priors must define the following attributes:

    :attr mean: Mean value [W] or [1]
    :attr var: Variance [W] or [1]
    """

    def __init__(self, data_model, idx, nnodes=None, **kwargs):
        """
        Constructor

        Subclasses must call this constructor, as well as define any
        constant tensors and any Variable tensors. Tensors which depend
        on Variables must be defined in ``build()``

        :param data_model: DataModel object
        """
        LogBase.__init__(self)
        self.data_model = data_model
        self.idx = idx
        if nnodes:
            self.nnodes = nnodes
        else:
            self.nnodes = self.data_model.model_space.size

    def build(self, post):
        """
        Define tensors that depend on any Variables in the prior or on the current
        state of the posterior
        
        This must not be done in the constructor. This method only needs to be
        implemented for priors that have inferrable variables

        :param post: Posterior object
        """
        pass

    @property
    def is_gaussian(self):
        return isinstance(self, NormalPrior)

    def mean_log_pdf(self, samples):
        """
        :param samples: A tensor of shape [W, P, S] where W is the number
                        of parameter nodes, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples

        :return: A tensor of shape [W] where W is the number of parameter nodes
                 containing the mean log PDF of the parameter samples
                 provided
        """
        raise NotImplementedError()

    def log_det_cov(self):
        raise NotImplementedError()

class NormalPrior(ParameterPrior):
    """
    Fixed Gaussian prior for a single model parameter
    """

    def __init__(self, data_model, idx, prior_dist, **kwargs):
        """
        """
        ParameterPrior.__init__(self, data_model, idx, **kwargs)
        self.scalar_mean = prior_dist.mean
        self.scalar_var = prior_dist.var

        self.mean = tf.fill([self.nnodes], NP_DTYPE(self.scalar_mean))
        self.var = tf.fill([self.nnodes], NP_DTYPE(self.scalar_var))
        self.vars = []

    def mean_log_pdf(self, samples):
        """
        Mean log PDF for normal distribution

        Note that ``term1`` is a constant offset when the prior variance is fixed and hence
        in earlier versions of the code this was neglected, along with other constant offsets
        such as factors of pi. However when this code is inherited by spatial priors and ARD
        the variance is no longer fixed and this term must be included.
        """
        dx = tf.subtract(samples, tf.reshape(self.mean, [self.nnodes, 1, 1])) # [W, 1, N]
        z = tf.math.divide(tf.square(dx), 
                           tf.reshape(self.var, [self.nnodes, 1, 1])) # [W, 1, N]
        term1 = -0.5*tf.math.log(tf.reshape(self.var, [self.nnodes, 1, 1]))
        term2 = -0.5*z
        log_pdf = term1 + term2 # [W, 1, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.nnodes]) # [W]
        return mean_log_pdf

    def __str__(self):
        return "Non-spatial prior (%f, %f)" % (self.scalar_mean, self.scalar_var)

class MRFSpatialPrior(ParameterPrior):
    """
    ParameterPrior which performs adaptive spatial regularization based on the 
    contents of neighbouring nodes using the Markov Random Field method
    """

    def __init__(self, data_model, idx, prior_dist, **kwargs):
        """
        """
        ParameterPrior.__init__(self, data_model, idx)
        self.q1 = kwargs.get('gamma_q1', 1.0)
        self.q2 = kwargs.get('gamma_q2', 10)
        self.log.debug("gamma q1", self.q1)
        self.log.debug("gamma q2", self.q2)

        # Laplacian matrix
        self.laplacian = scipy_to_tf_sparse(data_model.model_space.laplacian)

        # Laplacian matrix with diagonal zeroed
        offdiag_mask = self.laplacian.indices[:, 0] != self.laplacian.indices[:, 1]
        self.laplacian_nodiag = tf.SparseTensor(
            indices=self.laplacian.indices[offdiag_mask],
            values=self.laplacian.values[offdiag_mask], 
            dense_shape=[self.nnodes, self.nnodes]
        ) # [W, W] sparse

        # Diagonal of Laplacian matrix [W]
        diag_mask = self.laplacian.indices[:, 0] == self.laplacian.indices[:, 1]
        self.laplacian_diagonal = -self.laplacian.values[diag_mask]

        # Set up spatial smoothing parameter - infer the log so always positive
        self.num_aks = data_model.model_space.num_strucs
        self.sub_strucs = data_model.model_space.parts
        self.slices = data_model.model_space.slices
        ak_init = tf.fill([self.num_aks], NP_DTYPE(kwargs.get("ak", 1e-5)))
        if kwargs.get("infer_ak", True):
            self.log_ak = tf.Variable(np.log(ak_init), dtype=TF_DTYPE)
            self.vars = [self.log_ak]
        else:
            self.log_ak = tf.constant(np.log(ak_init), dtype=TF_DTYPE)
            self.vars = []

    def __str__(self):
        return "MRF spatial prior"

    def build(self, post):
        """
        self.ak is a nodewise tensor containing the relevant ak for each node
        (one per sub-structure)
        """
        aks_nodewise = []
        for struc_idx, struc in enumerate(self.sub_strucs):
            aks_nodewise.append(tf.fill([struc.nnodes], tf.exp(self.log_ak[struc_idx])))
        self.ak = tf.concat(aks_nodewise, 0) # [W]

        # For the spatial mean we essentially need the (weighted) average of 
        # nearest neighbour mean values. This does not involve the current posterior
        # mean at the voxel itself!
        # This is the equivalent of ApplyToMVN in Fabber
        node_mean = tf.expand_dims(post.mean[:, self.idx], 1) # [W]
        node_nn_total_weight = tf.sparse.reduce_sum(self.laplacian_nodiag, axis=1) # [W]
        spatial_mean = tf.sparse.sparse_dense_matmul(self.laplacian_nodiag, node_mean) # [W]
        spatial_mean = tf.squeeze(spatial_mean, 1)
        spatial_mean = spatial_mean / node_nn_total_weight # [W]
        spatial_prec = node_nn_total_weight * self.ak # [W]

        #self.var = 1 / (1/init_variance + spatial_prec)
        self.var = 1 / spatial_prec # [W]
        self.mean = self.var * spatial_prec * spatial_mean # [W]

    def mean_log_pdf(self, samples):
        r"""
        mean log PDF for the MRF spatial prior.

        This is calculating:

        :math:`\log P = \frac{1}{2} \log \phi - \frac{\phi}{2}\underline{x^T} D \underline{x}`
        """

        # Method 1: x * (D @ x), existing approach gives a node-wise quantity 
        # which is NOT the mathmetical SSD per node (but the sum across all 
        # nodes is equivalent)

        def _calc_xDx(D, x):
            Dx = tf.sparse.sparse_dense_matmul(D, x) # [W,N]
            xDx = x * Dx # [W,N]
            return xDx 

        samples = tf.reshape(samples, (self.nnodes, -1)) # [W,N]
        xDx = _calc_xDx(self.laplacian, samples) # [W, N]
        half_log_ak = 0.5 * self.log_ak # [T]
        half_ak_xDx = 0.5 * tf.expand_dims(self.ak, -1) * xDx # [W, N]
        mean_log_p = 0
        for struc_idx, struc in enumerate(self.sub_strucs):
            log_p = half_log_ak[struc_idx] + half_ak_xDx[self.slices[struc_idx]]
            mean_log_p += tf.reduce_mean(log_p)

        # Optional extra: cost from gamma prior on ak. 
        mean_log_p += tf.reduce_sum(((self.q1-1) * self.log_ak) - tf.exp(self.log_ak) / self.q2)
        return mean_log_p

class ARDPrior(NormalPrior):
    """
    Automatic Relevance Determination prior
    """
    def __init__(self, data_model, idx, prior_dist, **kwargs):
        """
        """
        NormalPrior.__init__(self, data_model, idx, prior_dist, **kwargs)
        self.fixed_var = prior_dist.var
        self.mean = tf.fill((self.nnodes,), NP_DTYPE(0))
        
        # Set up inferred precision parameter phi - infer the log so always positive
        # FIXME should we use hardcoded default_phi or the supplied variance?
        default_phi = np.full((self.nnodes, ), np.log(1e-12))
        self.log_phi = tf.Variable(default_phi, dtype=TF_DTYPE)
        self.vars = [self.log_phi]

    def __str__(self):
        return "ARD prior"

    def build(self, post):
        self.phi = tf.clip_by_value(tf.exp(self.log_phi), 0, 1e6)
        self.var = 1/self.phi

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

        q1, q2 = 1.0, 10
        gk = 1 / (0.5 * trace_term + 0.5 * term2 + 1/q1)
        hk = tf.multiply(tf.cast(self.nnodes, TF_DTYPE), 0.5) + q2
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
        self.vars = [self.log_ak]

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
        self.vars = []

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

class FactorisedPrior(LogBase):
    """
    Prior for a collection of parameters where there is no prior covariance

    In this case the mean log PDF can be summed from the contributions of each
    parameter
    """

    def __init__(self, priors, **kwargs):
        LogBase.__init__(self)
        self.priors = priors
        self.nparams = len(priors)
        self.vars = sum([p.vars for p in priors], [])

    def build(self, post):
        for prior in self.priors:
            prior.build(post)

        means = [prior.mean for prior in self.priors]
        variances = [prior.var for prior in self.priors]
        self.mean = tf.stack(means, axis=-1)
        self.var = tf.stack(variances, axis=-1)
        self.std = tf.sqrt(self.var)

        # Define a diagonal covariance matrix for convenience
        self.cov = tf.linalg.diag(self.var)

    def mean_log_pdf(self, samples):
        nnodes = tf.shape(samples)[0]

        mean_log_pdf = tf.zeros([nnodes], dtype=TF_DTYPE)
        for idx, prior in enumerate(self.priors):
            param_samples = tf.slice(samples, [0, idx, 0], [-1, 1, -1])
            param_logpdf = prior.mean_log_pdf(param_samples)
            mean_log_pdf = tf.add(mean_log_pdf, param_logpdf)
        return mean_log_pdf
    
    def log_det_cov(self):
        """
        Determinant of diagonal matrix is product of diagonal entries
        """
        return tf.reduce_sum(tf.math.log(self.var), axis=1)
