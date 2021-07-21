"""
VABY_SVB - Stochastic Bayesian inference of a nonlinear model

Infers:
    - Posterior mean values of model parameters
    - A posterior covariance matrix (which may be diagonal or a full
      positive-definite matrix)

The general order for tensor dimensions is:
    - Voxel indexing (V=number of voxels / W=number of parameter nodes)
    - Parameter indexing (P=number of parameters)
    - Sample indexing (S=number of samples)
    - Data point indexing (B=batch size, i.e. number of time points
      being trained on, in some cases T=total number of time points
      in full data)

This ordering is chosen to allow the use of TensorFlow batch matrix
operations. However it is inconvenient for the model which would like
to be able to index input by parameter. For this reason we transpose
when calling the model's ``evaluate`` function to put the P dimension
first.

The parameter nodes, W, are the set of points on which parameters are defined
and will be output. They may be voxel centres, or surface element nodes. The
data voxels, V, on the other hand are the points on which the data to be fitted to
is defined. Typically this will be volumetric voxels as that is what most
imaging experiments output as raw data.

In many cases, W will be the same as V since we are inferring volumetric parameter
maps from volumetric data. However we might alternatively want to infer surface
based parameter maps but keep the comparison to the measured volumetric data. In
this case V and W will be different. The key point at which this difference is handled
is the model evaluation which takes parameters defined on W and outputs a prediction
defined on V.

V and W are currently identical but may not be in the future. For example
we may want to estimate parameters on a surface (W=number of surface 
nodes) using data defined on a volume (V=number of voxels).

Ideas for per voxel/vertex convergence:

    - Maintain vertex_mask as member. Initially all ones
    - Mask nodes when generating samples and evaluating model. The
      latent cost will be over unmasked nodes only.
    - PROBLEM: need reconstruction cost defined over full voxel set
      hence need to project model evaluation onto all voxels. So
      masked nodes still need to keep their previous model evaluation
      output
    - Define criteria for masking nodes after each epoch
    - PROBLEM: spatial interactions make per-voxel convergence difficult.
      Maybe only do full set convergence in this case (like Fabber)
"""
import time
import six

import numpy as np
import tensorflow as tf

from vaby.utils import LogBase, TF_DTYPE

from .noise import NoiseParameter
from .prior import FactorisedPrior, get_prior
from .posterior import FactorisedPosterior, MVNPosterior, get_posterior

class Svb(LogBase):
    """
    Stochastic Bayesian model fitting

    :ivar model: Model instance to be fitted to some data
    :ivar prior: svb.prior.Prior instance defining the prior parameter distribution
    :ivar post: svb.posterior.Posterior instance defining the posterior parameter distribution
    :ivar params: Sequence of Parameter instances of parameters to infer. This includes the model
                  parameters and the noise parameter(s)
    """
    def __init__(self, data_model, fwd_model, **kwargs):
        LogBase.__init__(self)

        # The data model
        self.data_model = data_model

        # The model to use for inference and time points
        self.model = fwd_model

        # Expect tpts to have a dimension for voxelwise variation even if it is the same for all voxels
        self.tpts = self.model.tpts()
        if self.tpts.ndim == 1:
            self.tpts = self.tpts.reshape(1, -1)

        # Determine number of voxels and timepoints and check consistent
        if self.tpts.shape[0] > 1 and self.tpts.shape[0] != self.data_model.n_nodes:
            raise ValueError("Time points has %i nodes, but data has %i" % (self.tpts.shape[0], self.data_model.n_nodes))
        if self.tpts.shape[1] != self.data_model.n_tpts:
            raise ValueError("Time points has length %i, but data has %i volumes" % (self.tpts.shape[1], self.data_model.n_tpts))

        # For debug purposes 
        self.latent_only = kwargs.get('latent_only', False)
        self.reconstruction_only = kwargs.get('reconstruction_only', False)

        # All the parameters to infer - model parameters. 
        # Noise is defined as a separate parameter in "voxel" space 
        # (never "node" - surface - as may be the case for the model) 
        self.params = list(fwd_model.params)
        self.noise_param = NoiseParameter()
        self._nparams = len(self.params) + 1
        self._infer_covar = kwargs.get("infer_covar", False)
        self.mean_1, self.covar_1 = None, None

        # Create prior and posterior 
        self._create_prior_post(**kwargs)

    @property
    def n_model_params(self):
        """Number of model parameters, exlcuding noise
        """
        return self._nparams - 1

    @property
    def n_all_params(self):
        """Number of paramters, including noise
        """
        return self._nparams

    @property
    def nnodes(self):
        """
        Number of positions at which *model* parameters will be estimated
        """
        return self.data_model.n_nodes

    @property
    def nvoxels(self):
        """Number of data voxels that will be used for estimation 
        """
        return self.data_model.n_unmasked_voxels
 
    def _create_input_tensors(self):
        """
        Tensorflow input required for training
        
        x will have shape VxB where B is the batch size and V the number of voxels
        xfull is the full data so will have shape VxT where N is the full time size
        tpts_train will have shape 1xB or VxB depending on whether the timeseries is voxel
        dependent (e.g. in 2D multi-slice readout)
        
        NB we don't know V, B and T at this stage so we set placeholder variables
        self.nvoxels and self.nt_full and use validate_shape=False when creating
        tensorflow Variables
        """
        # Full data - we need this during training to correctly scale contributions
        # to the cost
        self.data_full = tf.constant(self.data_model.data_flat, dtype=TF_DTYPE, name="data_full")

        # Number of time points in full data - known at runtime
        self.nt_full = self.data_model.n_tpts

        # Initial learning rate
        #self.initial_lr = tf.placeholder(TF_DTYPE, shape=[])

        # Counters to keep track of how far through the full set of optimization steps
        # we have reached
        #self.global_step = tf.train.create_global_step()
        #self.num_steps = tf.placeholder(TF_DTYPE, shape=[])

        # Optional learning rate decay - to disable simply set decay rate to 1.0
        #self.lr_decay_rate = tf.placeholder(TF_DTYPE, shape=[])
        #self.learning_rate = tf.train.exponential_decay(
        #    self.initial_lr,
        #    self.global_step,
        #    self.num_steps,
        #    self.lr_decay_rate,
        #    staircase=False,
        #)

        # Amount of weight given to latent loss in cost function (0-1)
        #self.latent_weight = tf.placeholder(TF_DTYPE, shape=[])

        # Initial number of samples per parameter for the sampling of the posterior distribution
        #self.initial_ss = tf.placeholder(tf.int32, shape=[])

        # Optional increase in the sample size - to disable set factor to 1.0
        #self.ss_increase_factor = tf.placeholder(TF_DTYPE, shape=[])
        #self.sample_size = tf.cast(tf.round(tf.train.exponential_decay(
        #    tf.cast(self.initial_ss, TF_DTYPE),
        #    self.global_step,
        #    self.num_steps,
        #    self.ss_increase_factor,
        #    staircase=False,
        #    #tf.to_float(self.initial_ss) * self.ss_increase_factor,
        #    #power=1.0,
        #)), tf.int32)

    def _create_prior_post(self, **kwargs):
        """
        Create voxelwise prior and posterior distribution tensors
        """
        self.log.info("Setting up prior and posterior")

        # Create posterior distributions for model parameters
        # Note this can be initialized using the actual data
        gaussian_posts, nongaussian_posts, all_posts = [], [], []
        for idx, param in enumerate(self.params):    
            post = get_posterior(idx, param, self.tpts, 
                self.data_model, init=self.data_model.post_init, 
                **kwargs
            )
            if post.is_gaussian:
                gaussian_posts.append(post)
            else:
                nongaussian_posts.append(post)
            all_posts.append(post)

        # The noise posterior is defined separate to model posteriors in 
        # the voxel data space 
        self.noise_post = get_posterior(idx+1, self.noise_param,
                                        self.tpts, self.data_model, 
                                        init=self.data_model.post_init, **kwargs)

        if self._infer_covar:
            self.log.info(" - Inferring covariances (correlation) between %i Gaussian parameters" % len(gaussian_posts))
            if nongaussian_posts:
                self.log.info(" - Adding %i non-Gaussian parameters" % len(nongaussian_posts))
                self.post = FactorisedPosterior([MVNPosterior(gaussian_posts, **kwargs)] + nongaussian_posts, name="post", **kwargs)
            else:
                self.post = MVNPosterior(gaussian_posts, name="post", init=self.data_model.post_init, **kwargs)

        else:
            self.log.info(" - Not inferring covariances between parameters")
            self.post = FactorisedPosterior(all_posts, name="post", **kwargs)

        # Create prior distribution - note this can make use of the posterior e.g.
        # for spatial regularization
        all_priors = []
        for idx, param in enumerate(self.params):            
            all_priors.append(get_prior(param, self.data_model, idx=idx, post=self.post, **kwargs))
        self.prior = FactorisedPrior(all_priors, name="prior", **kwargs)

        # As for the noise posterior, the prior is defined seperately to the
        # model ones, and again in voxel data space  
        self.noise_prior = get_prior(self.noise_param, self.data_model, 
                                     idx=idx+1, post=self.noise_param.post_dist)

        # If all of our priors and posteriors are Gaussian we can use an analytic expression for
        # the latent loss - so set this flag to decide if this is possible
        self.analytic_latent_loss = (np.all([ p.is_gaussian for p in all_priors ]) 
                                     and not nongaussian_posts 
                                     and not kwargs.get("force_num_latent_loss", False)
                                     and False)   # FIXME always disabled for now 
        if self.analytic_latent_loss:
            self.log.info(" - Using analytical expression for latent loss since prior and posterior are Gaussian")
        else:
            self.log.info(" - Using numerical calculation of latent loss")

        # Report summary of priors/posterior for each parameter
        for idx, param in enumerate(self.params):
            self.log.info(" - %s", param)
            self.log.info("   - Prior: %s %s", param.prior_dist, all_priors[idx])
            self.log.info("   - Posterior: %s %s", param.post_dist, all_posts[idx])

        self.log.info(" - Noise")
        self.log.info("   - Prior: %s %s", self.noise_param.prior_dist, self.noise_prior)
        self.log.info("   - Posterior: %s %s", self.noise_param.post_dist, self.noise_post)

    def _get_model_prediction(self, param_samples, tpts):
        """
        Get a model prediction for the data batch being processed for each
        sample from the posterior

        :param param_samples: Tensor [W x P x S] containing samples from the posterior.
                              This is does not include noise samples. S is the number 
                              of samples (not always the same as the batch size). 

        :return Tensor [V x S x B]. B is the batch size, so for each voxel and sample
                we return a prediction which can be compared with the data batch
        """
        model_samples, model_means, model_vars = [], [], []
        for idx, param in zip(range(param_samples.shape[1]), self.params):
            int_samples = param_samples[:, idx, :]
            int_means = self.post.mean[:, idx]
            int_vars = self.post.var[:, idx]
            
            # Transform the underlying Gaussian samples into the values required by the model
            # denoted by the prefix int_ -> ext_, determined by each parameter's 
            # underlying distribution. 
            #
            # The sample parameter values tensor also needs to be reshaped to [P x W x S x 1] so
            # the time values from the data batch will be broadcasted and a full prediction
            # returned for every sample
            model_samples.append(tf.expand_dims(param.post_dist.transform.ext_values(int_samples), -1))
            ext_means, ext_vars = param.post_dist.transform.ext_moments(int_means, int_vars)
            model_means.append(ext_means)
            model_vars.append(ext_vars)

        # The timepoints tensor has shape [V x B] or [1 x B]. It needs to be reshaped
        # to [V x 1 x B] or [1 x 1 x B] so it can be broadcast across each of the S samples
        sample_tpts = tf.expand_dims(tpts, 1)
        
        # Produce a model prediction for each set of transformed parameter samples passed in 
        # Model prediction has shape [W x S x B]
        self.model_samples = model_samples
        self.sample_predictions = self.model.evaluate(model_samples, sample_tpts)

        # Define convenience tensors for querying the model-space sample, means and prediction. 
        # Modelfit_nodes has shape [W x B]. Produce a current "best estimate" prediction from 
        # model_means. This is distinct to producing a prediction for each samples. 
        self.model_means = tf.identity(model_means)
        self.model_vars = tf.identity(model_vars)

        return self.sample_predictions

    def _cost(self, data, tpts, sample_size):
        """
        Create the loss optimizer which will minimise the cost function

        The loss is composed of two terms:

        1. log likelihood. This is a measure of how likely the data are given the
           current posterior, i.e. how well the data fit the model using
           the inferred parameters.

        2. The latent loss. This is a measure of how closely the posterior fits the
           prior
        """
        # Generate a set of model parameter samples from the posterior [W x P x B]
        # Generate a set of noise parameter samples from the posterior [W x 2 x B]
        # They are in the 'internal' form required by the distributions themselves
        param_samples_int = self.post.sample(sample_size)
        noise_samples_int = self.noise_post.sample(sample_size)

        # Part 1: Reconstruction loss
        #
        # This deals with how well the parameters replicate the data and is 
        # defined as the log-likelihood of the data (given the parameters).
        #
        # This is calculated using the noise model. For each prediction generated
        # from the samples, the difference between it and the original data is 
        # taken to be an estimate of the noise ('residual' loss). This noise estimate
        # is compared with the current noise posterior in each voxel (via samples
        # drawn from the posterior and calculation of the log likelihoood). 
        # 
        # If the current model parameters are poor, then the model prediction
        # will not reproduce the data well. The high residual loss between the
        # predictions and the data will be interpreted as a high level of noise. 
        # If these noise levels are highly unlikely given the current noise 
        # parameters, then the log-likelihood will be low, and in turn the 
        # free energy (the quantity we wish to maximise) will be reduced. 

        # For each set of parameter samples, produce a model prediction with 
        # shape [W x S x B]. Project these into voxel space for comparison 
        # with the data. Note that we pass the total number of time points 
        # as we need to scale this term correctly when the batch size is not
        # the full data size. 
        model_prediction = self._get_model_prediction(param_samples_int, tpts)
        model_prediction_voxels = self.data_model.nodes_to_voxels_ts(model_prediction)

        # FIXME for some reason the below code to evaluate model fit returns 
        # an error currently - related to while loops and sparse matrices
        # param_means = tf.expand_dims(tf.transpose(self.model_means), 2)
        # modelfit_nodes = self._get_model_prediction(param_means)
        # modelfit_voxels = self.data_model.nodes_to_voxels(modelfit_nodes[...,0])

        # Convert the noise samples from from internal to external representation.
        # Save the current moments of the noise posterior. 
        transformer = self.noise_param.post_dist.transform
        noise_samples_ext = tf.squeeze(transformer.ext_values(noise_samples_int))
        noise_mean_ext, noise_var_ext = transformer.ext_moments(
                                            self.noise_post.mean, self.noise_post.var)
        self.noise_mean = noise_mean_ext
        self.noise_var = noise_var_ext

        # Reconstruction loss calculated here: the log-likelihood of the difference between 
        # the data and the model predictions. Calculation of the log-likelihood requires 
        # the properties of the noise distribution, which we get from the noise paramter
        # samples. Finally, we average this over voxels. 
        reconstr_loss = self.noise_param.log_likelihood(data, 
                            model_prediction_voxels, noise_samples_ext, self.data_model.n_tpts)
        self.reconstr_loss = reconstr_loss
        self.mean_reconstr_cost = tf.reduce_mean(self.reconstr_loss, name="mean_reconstr_cost")

        # Part 2: Latent loss
        #
        # This penalises parameters which are far from the prior
        # If both the prior and posterior are represented by an MVN we can calculate an analytic
        # expression for this cost. If not, we need to do it numerically using the posterior
        # sample obtained earlier. Note that the mean log pdf of the posterior based on sampling
        # from itself is just the distribution entropy so we allow it to be calculated without
        # sampling.
        if self.analytic_latent_loss: 
            # FIXME: this will not be able to handle mixed voxel/node priors. 
            latent_loss = tf.identity(self.post.latent_loss(self.prior), name="latent_loss")
        else:

            # Latent loss is calculated over model parameters and noise parameters separately 
            # Note that these may be sized differently (one over nodes, the other voxels)
            self.param_latent_loss = tf.subtract(
                                    self.post.entropy(param_samples_int), 
                                    self.prior.mean_log_pdf(param_samples_int), 
                                    name="param_latent_loss")
            self.noise_latent_loss = tf.subtract(
                                    self.noise_post.entropy(noise_samples_int), 
                                    self.noise_prior.mean_log_pdf(noise_samples_int),
                                    name="noise_latent_loss")

        # Reduce the latent costs over their voxel/node dimension to give an average.
        # This deals with the situation where they may be sized differently. The overall
        # latent cost is then the sum of the two averages 
        self.latent_weight = 1 # FIXME
        self.mean_latent_loss = tf.add(
                tf.reduce_mean(self.latent_weight * self.param_latent_loss), 
                tf.reduce_mean(self.latent_weight * self.noise_latent_loss),
                name="mean_latent_cost")
                
        # We have the possibility of gradually introducing the latent loss 
        # via the latent_weight variable. This is based on the theory that you 
        # should let the model fit the data first and then allow the fit to
        # be perturbed by the priors.
        if self.latent_weight == 0:
            raise NotImplementedError("Variable latent cost not implemented")

        # Overall mean cost is taken as the sum of (mean voxel reconstruction loss) 
        # and the sum ((average latent over model) + (average latent over nosie))
        if self.latent_only: 
            self.log.info("Debug: latent cost only")
            self.mean_cost = self.mean_latent_loss
        elif self.reconstruction_only: 
            self.log.info("Debug: reconstruction cost only")
            self.mean_cost = self.mean_reconstr_cost
        else: 
            self.mean_cost = tf.add(self.mean_reconstr_cost, self.mean_latent_loss)

        return self.mean_cost
    
    def _optimize(self):
        # Set up ADAM to optimise over mean cost as defined above. 
        # It is also possible to optimize the total cost but this makes it harder to compare with
        # variable numbers of voxels
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimize = self.optimizer.minimize(self.mean_cost, global_step=self.global_step)

    def cost(self, data, tpts, sample_size):
        self.prior.build()
        self.post.build()
        self.noise_prior.build()
        self.noise_post.build()
        return self._cost(data, tpts, sample_size)

    def fit_batch(self):
        """
        Train model based on mini-batch of input data.

        :return: Tuple of (param_latent, noise_latent, reconstruction) losses for batch
        """
        _, param_latent, noise_latent, reconstr = self.evaluate(self.optimize, 
                    self.param_latent_loss, self.noise_latent_loss, self.reconstr_loss)
        return param_latent, noise_latent, reconstr

    def state(self):
        """
        Get the current state of the optimization.

        This can be used to restart from a previous state if a numerical error occurs
        """
        return self.evaluate(self.post.state())
        
    def set_state(self, state):
        """
        Set the state of the optimization

        :param state: State as returned by the ``state()`` method
        """
        self.evaluate(self.post.set_state(state))

    def _extract_ak(self):
        """
        Spatial smoothing ak values. Returned in order of 
        [param1_surf, param2_surf, ...], [param1_vol, ...]
        Empty lists will be returned if the mode is not 
        applicable (eg, surf = [] in pure volume mode)
        """
        sak, vak = [], []
        for idx, prior in enumerate(self.prior.priors):
            try: 
                if self.data_model.is_hybrid: 
                    aks = [ self.evaluate(a) for a in prior.ak ]
                    sak.append(aks[0]); vak.append(aks[1])
                elif self.data_model.is_volumetric: 
                    vak.append(self.evaluate(prior.ak))
                else: 
                    sak.append(self.evaluate(prior.ak))
            except: 
                pass 
        return np.array(sak), np.array(vak) 

        
    def train(self, batch_size=None, sequential_batches=False,
              epochs=100, fit_only_epochs=0, display_step=10,
              learning_rate=0.1, lr_decay_rate=1.0,
              sample_size=None, ss_increase_factor=1.0,
              revert_post_trials=50, revert_post_final=True,
              **kwargs):
        """
        Train the graph to infer the posterior distribution given timeseries data

        Optional arguments:

        :param batch_size: Batch size to use when training model. Need not be a factor of T, however if not
                           batches will not all be the same size. If not specified, data size is used (i.e.
                           no mini-batch optimization)
        :param sequential_batches: If True, form batches from consecutive time points rather than strides
        :param epochs: Number of training epochs
        :param fit_only_epochs: If specified, this number of epochs will be restricted to fitting only
                                and ignore prior information. In practice this means only the
                                reconstruction loss is considered not the latent cost
        :param display_step: How many steps to execute for each display line
        :param learning_rate: Initial learning rate
        :param lr_decay_rate: When adjusting the learning rate, the factor to reduce it by
        :param sample_size: Number of samples to use when estimating expectations over the posterior
        :param ss_increase_factor: Factor to increase the sample size by over the epochs
        :param revert_post_trials: How many epoch to continue for without an improvement in the mean cost before
                                   reverting the posterior to the previous best parameters
        :param revert_post_final: If True, revert to the state giving the best cost achieved after the final epoch
        """
        data = self.data_model.data_flat
        n_voxels = self.data_model.n_voxels
        n_nodes = self.data_model.n_nodes
        n_tpts = self.data_model.n_tpts

        # Determine number of batches and sample size
        if batch_size is None:
            batch_size = n_tpts
        n_batches = int(np.ceil(float(n_tpts) / batch_size))
        if sample_size is None:
            sample_size = batch_size

        # Create arrays to store cost and parameter histories, mean and voxelwise
        training_history = {
            "mean_cost" : np.zeros([epochs+1]),
            "reconstruction_cost" : np.zeros([n_voxels, epochs+1]),
            "param_latent_loss" : np.zeros([n_nodes, epochs+1]), 
            "noise_latent_loss" : np.zeros([n_voxels, epochs+1]),
            "node_params" : np.zeros([n_nodes, epochs+1, self.n_model_params]),
            "mean_node_params" : np.zeros([epochs+1, self.n_model_params]),
            "noise_params": np.zeros([n_voxels, epochs+1]),
            "mean_noise_params": np.zeros([epochs+1]),
            "ak" : {
                "surf": np.zeros([epochs+1, self.n_model_params]),
                "vol": np.zeros([epochs+1, self.n_model_params]),
            }, 
            "runtime" : np.zeros([epochs+1]),
        }

        # Training cycle
        #self.feed_dict = {
        #    self.data_full : data,
        #    self.num_steps : epochs*n_batches,
        #    self.initial_lr : learning_rate,
        #    self.lr_decay_rate : lr_decay_rate,
        #    self.initial_ss : sample_size,
        #    self.ss_increase_factor : ss_increase_factor,
        #    self.data_train: data,
        #    self.tpts_train : self.tpts,
        #    self.latent_weight : 1.0,
        #}
        self.cost(data, self.tpts, sample_size)

        trials, best_cost, best_state = 0, 1e12, None
        latent_weight = 0

        # Each epoch passes through the whole data but it may do this in 'batches' so there may be
        # multiple training iterations per epoch, one for each batch
        self.log.info("Starting inference...")
        self.log.info(" - Number of training epochs: %i", epochs)
        self.log.info(" - %i voxels of %i time points (processed in %i batches of target size %i)" , n_voxels, n_tpts, n_batches, batch_size)
        self.log.info(" - Projector defines %d model nodes:", self.data_model.n_nodes)
        self.log.info(" - Initial learning rate: %.5f (decay rate %.3f)", learning_rate, lr_decay_rate)
        self.log.info(" - Initial sample size: %i (increase factor %.3f)", sample_size, ss_increase_factor)
        if revert_post_trials > 0:
            self.log.info(" - Posterior reversion after %i trials", revert_post_trials)

        initial_cost = self.mean_cost.numpy().mean()
        initial_param_means = np.mean(self.model_means.numpy().T, axis=0)
        initial_param_vars = np.mean(self.post.var.numpy(), axis=0)
        initial_noise = np.array([self.noise_mean.numpy(),
                                  self.noise_var.numpy()]).T

        initial_latent = np.mean(self.param_latent_loss.numpy()) + np.mean(self.noise_latent_loss.numpy())
        initial_reconstr = np.mean(self.reconstr_loss.numpy())
        start_time = time.time()

        state_str = (" - Start 0000: Means: %s, Variances: %s, mean cost %.4g (latent %.4g, reconstr %.4g)" 
                        % (initial_param_means, initial_param_vars, initial_cost, initial_latent, initial_reconstr))
        self.log.info(state_str)

        return
        # Potential for a bug here: don't use range(1, epochs+1) as it 
        # interacts badly with training the ak parameter. 
        for epoch in range(epochs):
            try:
                err = False
                total_param_latent = np.zeros([n_nodes])
                total_noise_latent = np.zeros([n_voxels])
                total_reconstr = np.zeros([n_voxels])

                if epoch == fit_only_epochs:
                    # Once we have completed fit_only_epochs of training we will allow the latent cost to have
                    # an impact and reset the best cost accordingly. By default this happens on the first epoch
                    latent_weight = 1.0
                    trials, best_cost = 0, 1e12

                # Iterate over training batches - note that there may be only one
                for i in range(n_batches):
                    if sequential_batches:
                        # Batches are defined by sequential data time points
                        index = i*batch_size
                        if i == n_batches - 1:
                            # Batch size may not be an exact factor of the number of time points
                            # so make the last batch the right size so all of the data is used
                            batch_size += n_tpts - n_batches * batch_size
                        batch_data = data[:, index:index+batch_size]
                        batch_tpts = self.tpts[:, index:index+batch_size]
                    else:
                        # Batches are defined by constant strides through the data time points
                        # This automatically handles case where number of time point does not
                        # exactly divide into batches
                        batch_data = data[:, i::n_batches]
                        batch_tpts = self.tpts[:, i::n_batches]

                    # Perform a training iteration using batch data
                    self.feed_dict.update({
                        self.data_train: batch_data,
                        self.tpts_train : batch_tpts,
                        self.latent_weight : latent_weight,
                    })
                    param_latent, noise_latent, reconstruction = self.fit_batch()
                    total_param_latent += param_latent / n_batches
                    total_noise_latent += noise_latent / n_batches 
                    total_reconstr += reconstruction / n_batches

            except tf.OpError:
                self.log.exception("Numerical error fitting batch")
                err = True

            # End of epoch
            # Extract model parameter estimates
            current_lr, current_ss = self.evaluate(self.learning_rate, self.sample_size)
            param_means = self.evaluate(self.model_means).T # [W, P]
            param_vars = self.evaluate(self.post.var) # [W, P]

            # Extract noise parameter estimates. 
            # The noise_mean is actually the 'best' estimate of noise variance, 
            # whereas noise_var is the variance of the noise vairance - yes, confusing!
            # We generally only care about the former 
            noise_params = np.array([ self.evaluate(self.noise_mean), 
                                      self.evaluate(self.noise_var) ]).T
            mean_noise_params = noise_params.mean(0)

            # Assembele total, mean and median costs. 
            # Note that param_latent is sized according to nodes, 
            # noise_latent is sized according to voxels, and these may be different
            mean_total_latent = total_param_latent.mean() + total_noise_latent.mean()
            mean_total_reconst = total_reconstr.mean()
            mean_total_cost = mean_total_latent + mean_total_reconst
            median_total_latent = np.median(total_param_latent) + np.median(total_noise_latent)
            median_total_reconst = np.median(total_reconstr)
            median_total_cost = median_total_latent + median_total_reconst # approx
            
            # Store costs (all are over batches of this epoch): 
            # mean total cost (reconstruction + latent)
            # reconstruction cost of each voxel 
            # parameter latent cost over nodes 
            # noise latent cost over voxels 
            training_history["mean_cost"][epoch] = mean_total_cost
            training_history["reconstruction_cost"][:, epoch] = total_reconstr
            training_history["param_latent_loss"][:, epoch] = total_param_latent
            training_history["noise_latent_loss"][:, epoch] = total_noise_latent

            # Store parameter estimates: 
            # mean across nodes of model parameter posterior distributions 
            # mean of model parameter posterior distribution for each node 
            # mean across voxels of (mean of noise variance posterior)
            # mean of noise variance posterior in each voxel 
            training_history["mean_node_params"][epoch, :] = param_means.mean(0)
            training_history["node_params"][:, epoch, :] = param_means
            training_history["noise_params"][:, epoch] = noise_params[:,0]
            sak, vak = self._extract_ak()
            if sak.size: training_history["ak"]["surf"][epoch,:] = sak 
            if vak.size: training_history["ak"]["vol"][epoch,:] = vak 

            if err or np.isnan(mean_total_cost) or np.isnan(param_means).any():
                # Numerical errors while processing this epoch. Revert to best saved params if possible
                if best_state is not None:
                    self.set_state(best_state)
                outcome = "Revert - Numerical errors"
            elif mean_total_cost < best_cost:
                # There was an improvement in the mean cost - save the current state of the posterior
                outcome = "Saving"
                best_cost = mean_total_cost
                best_state = self.state()
                trials = 0
            else:
                # The mean cost did not improve. 
                if revert_post_trials > 0:
                    # Continue until it has not improved for revert_post_trials epochs and then revert 
                    trials += 1
                    if trials < revert_post_trials:
                        outcome = "Trial %i" % trials
                    elif best_state is not None:
                        self.set_state(best_state)
                        outcome = "Revert"
                        trials = 0
                    else:
                        outcome = "Continue - No best state"
                        trials = 0
                else:
                    outcome = "Not saving"

            if (epoch % display_step == 0) and (epoch > 0):
                first_line = (("mean/median cost %.4g/%.4g, (latent %.4g, reconstr %.4g)") 
                                % (mean_total_cost, median_total_cost, mean_total_latent,
                                    mean_total_reconst))

                space_strings = []
                with np.printoptions(precision=3):
                    if vol_inds is not None: 
                        vmean = param_means[vol_inds,:].mean(0)
                        vvar = param_vars[vol_inds,:].mean(0)
                        space_strings.append(
                            "Volume: param means %s, param vars %s, ak %s"
                            % (vmean, vvar, vak))
                    if surf_inds is not None: 
                        smean = param_means[surf_inds,:].mean(0)
                        svar = param_vars[surf_inds,:].mean(0)
                        space_strings.append(
                            "Surface: param means %s, param vars %s, ak %s" 
                                            % (smean, svar, sak))
                    if subcort_inds is not None: 
                        submean = param_means[subcort_inds,:].mean(0)
                        subvar = param_vars[subcort_inds,:].mean(0)
                        space_strings.append("ROIs: param means %s, param vars %s" 
                                            % (submean, subvar))
                    end_str = ("Noise mean/var %s, lr %.4g, ss %.4g" 
                                % (mean_noise_params, current_lr, current_ss))
                state_str = ("\n"+10*" ").join((first_line, *space_strings, end_str))
                self.log.info(" - Epoch %04d: %s - %s", epoch, state_str, outcome)

            epoch_end_time = time.time()
            training_history["runtime"][epoch] = float(epoch_end_time - start_time)

        self.log.info(" - End of inference. ")
        if revert_post_final and best_state is not None:
            # At the end of training we revert to the state with best mean cost and write a final history step
            # with these values. Note that the cost may not be as reported earlier as this was based on a
            # mean over the training batches whereas here we recalculate the cost for the whole data set.
            self.log.info(" - Reverting to best batch-averaged cost")
            self.set_state(best_state)

        self.feed_dict[self.data_train] = data
        self.feed_dict[self.tpts_train] = self.tpts
        param_means = self.evaluate(self.model_means).T # [W, P]
        noise_params = np.array([ self.evaluate(self.noise_mean), 
                                  self.evaluate(self.noise_var) ]).T
        mean_noise_params = noise_params.mean(0)

        with np.printoptions(precision=3):
            final_str = ["   Best batch-averaged cost: %.4g" % best_cost]
            mean_str = []
            if vol_inds is not None: 
                mean_str.append("%s in volume" % param_means[vol_inds,:].mean(0))
            if surf_inds is not None: 
                mean_str.append("%s on surface" % param_means[surf_inds,:].mean(0))
            if subcort_inds is not None: 
                mean_str.append("%s in ROIs" % param_means[subcort_inds,:].mean(0))
            final_str.append("Final parameter means: " + ", ".join(mean_str))
            final_str.append("Final noise variance in volume: %.4g" % mean_noise_params[0])

        self.log.info(("\n"+10*" ").join(final_str))
        training_history["mean_node_params"][-1, :] = param_means.mean(0)
        training_history["node_params"][:, -1, :] = param_means
        training_history["mean_noise_params"][-1] = mean_noise_params[0]
        training_history["noise_params"][:, -1] = noise_params[:,0]
        sak, vak = self._extract_ak()
        if sak.size: training_history["ak"]["surf"][epoch,:] = sak 
        if vak.size: training_history["ak"]["vol"][epoch,:] = vak 

        # Return training history
        return training_history
