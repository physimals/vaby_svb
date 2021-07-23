"""
VABY_AVB - Definition of prior distribution
"""
import numpy as np

from toblerone.utils import is_symmetric, is_nsd
import tensorflow as tf

from vaby.utils import LogBase, TF_DTYPE, scipy_to_tf_sparse
from vaby.dist import Normal

PRIOR_TYPE_NONSPATIAL = "N"
PRIOR_TYPE_SPATIAL_MRF = "M"

def get_prior(param, data_model, **kwargs):
    """
    Factory method to return a vertexwise prior
    """
    prior = None
    if isinstance(param.prior_dist, Normal):
        if param.prior_type == "N":
            prior = NormalPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "M":
            prior = MRFSpatialPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "M2":
            prior = MRF2SpatialPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "Mfab":
            prior = FabberMRFSpatialPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)
        elif param.prior_type == "A":
            prior = ARDPrior(data_model, param.prior_dist.mean, param.prior_dist.var, **kwargs)

    if prior is not None:
        return prior
    else:
        raise ValueError("Can't create prior type %s for distribution %s - unrecognized combination" % (param.prior_type, param.prior_dist))

class Prior(LogBase):
    """
    Base class for a prior, defining methods that must be implemented
    """

    def __init__(self, data_model=None):
        """
        Constructor

        Subclasses must call this constructor, as well as define any
        constant tensors and any Variable tensors. Tensors which depend
        on Variables must be defined in ``build()``

        :param data_model: DataModel object
        """
        super().__init__() 
        self.data_model = data_model

        if data_model is not None: 
            # TODO: currently this NN tensor is only required for the Fabber
            # priors (not the full SVB prior) - remove? This is also the only
            # reason we require the adj_matrix on the DataModel 
            self.nn = tf.SparseTensor(
                indices=np.array(
                    [data_model.adj_matrix.row, 
                    data_model.adj_matrix.col]).T,
                values=data_model.adj_matrix.data, 
                dense_shape=data_model.adj_matrix.shape, 
            )

            # Vol and surface have a single laplcian, hybrid has 2 
            # (for surface and volume, listed in that order). Pad
            # the former case into a size-1 list 
            #if not data_model.is_hybrid: 
            lap = [data_model.laplacian]
            #else: 
            #    lap = data_model.laplacian 

            # Check sign convention on Laplacian
            for l in lap: 
                diags = l.tocsr()[np.diag_indices(l.shape[0])]
                if (diags > 0).any():
                    raise ValueError("Sign convention on Laplacian matrix: " +
                    "diagonal elements should be negative, off-diag positive.")
                assert is_nsd(l), 'Laplacian not NSD'
                assert is_symmetric(l), 'Laplacian not symmetric'

            #if data_model.is_hybrid: 
            #    self.surf_laplacian = scipy_to_tf_sparse(lap[0])
            #    self.vol_laplacian = scipy_to_tf_sparse(lap[1])
            #else: 
            self.laplacian = scipy_to_tf_sparse(lap[0])
        
    @property
    def is_gaussian(self):
        return isinstance(self, NormalPrior)

    @property
    def size(self):
        return self.data_model.n_nodes

    def build(self):
        """
        Define tensors that depend on any Variables in the prior
        
        This must not be done in the constructor. Only needs to be
        implemented for priors that have inferrable variables
        """
        pass

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

class NormalPrior(Prior):
    """
    Prior based on a nodewise univariate normal distribution
    """

    def __init__(self, data_model, mean, var, **kwargs):
        """
        :param mean: Prior mean value
        :param var: Prior variance
        """
        Prior.__init__(self, data_model)
        self.name = kwargs.get("name", "NormalPrior")
        self.scalar_mean = mean
        self.scalar_var = var
        self.mean = tf.fill([self.size], mean, name="%s_mean" % self.name)
        self.var = tf.fill([self.size], var, name="%s_var" % self.name)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)
        self.vars = []

    def mean_log_pdf(self, samples):
        """
        Mean log PDF for normal distribution

        Note that ``term1`` is a constant offset when the prior variance is fixed and hence
        in earlier versions of the code this was neglected, along with other constant offsets
        such as factors of pi. However when this code is inherited by spatial priors and ARD
        the variance is no longer fixed and this term must be included.
        """
        dx = tf.subtract(samples, tf.reshape(self.mean, [self.size, 1, 1])) # [W, 1, N]
        z = tf.math.divide(tf.square(dx), 
                           tf.reshape(self.var, [self.size, 1, 1])) # [W, 1, N]
        term1 = -0.5*tf.math.log(tf.reshape(self.var, [self.size, 1, 1]))
        term2 = -0.5*z
        log_pdf = term1 + term2 # [W, 1, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.size]) # [W]
        return mean_log_pdf

    def __str__(self):
        return "Non-spatial prior (%f, %f)" % (self.scalar_mean, self.scalar_var)

class FabberMRFSpatialPrior(NormalPrior):
    """
    Prior designed to mimic the 'M' type spatial prior in Fabber.
    
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
        NormalPrior.__init__(self, data_model, mean, var, name="FabberMRFSpatialPrior", **kwargs)
        self.idx = idx

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
        hk = tf.multiply(tf.cast(self.size, TF_DTYPE), 0.5) + q2
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

class MRFSpatialPrior(Prior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring nodes using the Markov Random Field method

    This uses the same formalism as the Fabber 'M' type spatial prior but treats the ak
    as a parameter of the optimization.
    """

    def __init__(self, data_model, mean, var, idx=None, post=None, **kwargs):
        Prior.__init__(self, data_model)
        self.name = kwargs.get("name", "MRFSpatialPrior")
        self.mean = tf.fill([self.size], mean, name="%s_mean" % self.name)
        self.var = tf.fill([self.size], var, name="%s_var" % self.name)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

        self.q1 = kwargs.get('gamma_q1', 1.0)
        self.q2 = kwargs.get('gamma_q2', 10)
        print("gamma q1", self.q1)
        print("gamma q2", self.q2)

        # Set up spatial smoothing parameter calculation from posterior and neighbour lists
        # We infer the log of ak.
        ak_init = kwargs.get("ak", 1e-5)
        if kwargs.get("infer_ak", True):
            #if self.data_model.is_hybrid: 
            #    self.logak = [
            #        tf.Variable(np.log(ak_init), name="log_ak_surf", dtype=TF_DTYPE),
            #        tf.Variable(np.log(ak_init), name="log_ak_vol", dtype=TF_DTYPE),
            #    ]
            #else:
            self.logak = tf.Variable(np.log(ak_init), name="log_ak", dtype=TF_DTYPE)
            self.vars = [self.logak]
        else:
            self.logak = tf.constant(np.log(ak_init), name="log_ak", dtype=TF_DTYPE)
            self.vars = []

    def build(self):
        # Convert from log space to real number space 
        #if self.data_model.is_hybrid: 
        #    self.ak = [
        #        tf.exp(self.logak[0]), 
        #        tf.exp(self.logak[1])
        #    ]
        #else:
        tf.exp(self.logak)

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

        samples = tf.reshape(samples, (self.size, -1)) # [W,N]

        #if self.data_model.is_hybrid: 
        #    surf_samples = samples[self.data_model.surf_slicer]
        #    vol_samples = samples[self.data_model.vol_slicer]
        #
        #    xDx_s = _calc_xDx(self.surf_laplacian, surf_samples)
        #    xDx_v = _calc_xDx(self.vol_laplacian, vol_samples)
        #
        #    ak_xDx_s = tf.identity(0.5 * self.ak[0] * xDx_s, name="half_ak_xDx_s")
        #    ak_xDx_v = tf.identity(0.5 * self.ak[1] * xDx_v, name="half_ak_xDx_v")
        #
        #    logP_s = tf.reduce_mean((0.5 * self.logak[0]) + ak_xDx_s)
        #    logP_v = tf.reduce_mean((0.5 * self.logak[1]) + ak_xDx_v)
        #
        #    gamma_s = (((self.q1-1) * self.logak[0]) - self.ak[0] / self.q2)
        #    gamma_v = (((self.q1-1) * self.logak[1]) - self.ak[1] / self.q2)
        #
        #    return logP_s + logP_v + gamma_s + gamma_v
        #else: 
        xDx = _calc_xDx(self.laplacian, samples)
        log_ak = tf.identity(0.5 * self.logak, name="log_ak")
        half_ak_xDx = tf.identity(0.5 * self.ak * xDx, name="half_ak_xDx")
        logP = log_ak + half_ak_xDx
        mean_logP = tf.reduce_mean(logP)

        # Optional extra: cost from gamma prior on ak. 
        mean_logP += (((self.q1-1) * self.logak) - self.ak / self.q2)
        return mean_logP

    def __str__(self):
        return "MRF spatial prior"

class ARDPrior(NormalPrior):
    """
    Automatic Relevance Determination prior
    """
    def __init__(self, data_model, mean, var, **kwargs):
        NormalPrior.__init__(self, data_model, mean, var, **kwargs)
        self.name = kwargs.get("name", "ARDPrior")
        self.fixed_var = self.var
        
        # Set up inferred precision parameter phi
        self.logphi = tf.Variable(tf.math.log(1/self.fixed_var), name="log_phi", dtype=TF_DTYPE)
        self.vars = [self.logphi]

    def build(self):
        self.phi = tf.exp(self.logphi)
        self.var = 1/self.phi
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

    def __str__(self):
        return "ARD prior"

class MRF2SpatialPrior(Prior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring nodes using the Markov Random Field method

    This uses the same formalism as the Fabber 'M' type spatial prior but treats the ak
    as a parameter of the optimization. It differs from MRFSpatialPrior by using the
    PDF formulation of the PDF rather than the matrix formulation (the two are equivalent
    but currently we keep both around for checking that they really are!)

    FIXME currently this does not work unless sample size=1
    """

    def __init__(self, data_model, mean, var, idx=None, post=None, nn=None, **kwargs):
        Prior.__init__(self, data_model)
        self.name = kwargs.get("name", "MRF2SpatialPrior")
        self.mean = tf.fill([self.size], mean, name="%s_mean" % self.name)
        self.var = tf.fill([self.size], var, name="%s_var" % self.name)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

        # We need the number of samples to implement the log PDF function
        self.sample_size = kwargs.get("sample_size", 5)

        # Set up spatial smoothing parameter calculation from posterior and neighbour lists
        self.logak = tf.Variable(-5.0, name="log_ak", dtype=TF_DTYPE)
        self.ak = tf.exp(self.logak)
        self.vars = [self.logak]

    def mean_log_pdf(self, samples):
        samples = tf.reshape(samples, (self.size, -1)) # [W, N]
        self.num_nn = tf.sparse.reduce_sum(self.nn, axis=1) # [W]

        expanded_nn = tf.sparse.concat(2, [tf.sparse.reshape(self.nn, (self.size, self.size, 1))] * self.sample_size)
        xj = expanded_nn * tf.reshape(samples, (self.size, 1, -1))
        #xi = tf.reshape(tf.sparse.to_dense(tf.sparse.reorder(self.nn)), (self.size, self.size, 1)) * tf.reshape(samples, (1, self.size, -1))
        xi = expanded_nn * tf.reshape(samples, (1, self.size, -1))
        #xi = tf.sparse.transpose(xj, perm=(1, 0, 2)) 
        neg_xi = tf.SparseTensor(xi.indices, -xi.values, dense_shape=xi.dense_shape )
        dx2 = tf.square(tf.sparse.add(xj, neg_xi), name="dx2")
        sdx = tf.sparse.reduce_sum(dx2, axis=0) # [W, N]
        term1 = tf.identity(0.5*self.logak, name="term1")
        term2 = tf.identity(-self.ak * sdx / 4, name="term2")
        log_pdf = term1 + term2  # [W, N]
        mean_log_pdf = tf.reshape(tf.reduce_mean(log_pdf, axis=-1), [self.size]) # [W]
        return mean_log_pdf

    def __str__(self):
        return "MRF2 spatial prior"

class ConstantMRFSpatialPrior(NormalPrior):
    """
    Prior which performs adaptive spatial regularization based on the 
    contents of neighbouring nodes using the Markov Random Field method

    This is equivalent to the Fabber 'M' type spatial prior
    """

    def __init__(self, data_model, mean, var, idx=None, **kwargs):
        """
        :param mean: Tensor of shape [W] containing the prior mean at each parameter vertex
        :param var: Tensor of shape [W] containing the prior variance at each parameter vertex
        :param post: Posterior instance
        """
        NormalPrior.__init__(self, data_model, mean, var, name="MRFSpatialPrior", **kwargs)
        self.idx = idx

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
        hk = float(self.size) * 0.5 + 1.0
        self.ak = gk * hk
        self.log.info("%s: ak=%f", self.name, self.ak)

    def _setup_mean_var(self, post_mean, post_cov):
        # This is the equivalent of ApplyToMVN in Fabber
        contrib_nn = 8*self.sum_means_nn # [W]
        
        spatial_mean = contrib_nn / (8*self.num_nn)
        spatial_prec = self.num_nn * self.ak

        self.var = 1 / (1/self.fixed_var + spatial_prec)
        #self.var = self.fixed_var
        self.mean = self.var * spatial_prec * spatial_mean
        #self.mean = self.fixed_mean + self.ak

class FactorisedPrior(Prior):
    """
    Prior for a collection of parameters where there is no prior covariance

    In this case the mean log PDF can be summed from the contributions of each
    parameter
    """

    def __init__(self, priors, **kwargs):
        Prior.__init__(self)
        self.priors = priors
        self.name = kwargs.get("name", "FactPrior")
        self.nparams = len(priors)
        self.vars = sum([p.vars for p in priors], [])

    def build(self):
        for prior in self.priors:
            prior.build()

        means = [prior.mean for prior in self.priors]
        variances = [prior.var for prior in self.priors]
        self.mean = tf.stack(means, axis=-1)
        self.var = tf.stack(variances, axis=-1)
        self.std = tf.sqrt(self.var, name="%s_std" % self.name)

        # Define a diagonal covariance matrix for convenience
        self.cov = tf.linalg.diag(self.var, name='%s_cov' % self.name)

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
        return tf.reduce_sum(tf.math.log(self.var), axis=1, name='%s_log_det_cov' % self.name)
