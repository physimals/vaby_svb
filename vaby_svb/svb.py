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

import numpy as np
import tensorflow as tf

from vaby.utils import InferenceMethod, TF_DTYPE

from .noise import NoiseParameter
from .prior import FactorisedPrior, get_prior
from .posterior import FactorisedPosterior, MVNPosterior, get_posterior

class Svb(InferenceMethod):
    """
    Stochastic Bayesian model fitting
    """

    def __init__(self, data_model, fwd_model, **kwargs):
        InferenceMethod.__init__(self, data_model, fwd_model, **kwargs)

    def run(self, epochs=100, learning_rate=0.1, **kwargs):
        """
        Run stochastic VB fitting
        
        Optional arguments:

        :param batch_size: Batch size to use when training model. Need not be a factor of T, however if not
                           batches will not all be the same size. If not specified, data size is used (i.e.
                           no mini-batch optimization)
        :param sequential_batches: If True, form batches from consecutive time points rather than strides
        :param epochs: Number of training epochs
        :param fit_only_epochs: If specified, this number of epochs will be restricted to fitting only
                                and ignore prior information. In practice this means only the
                                reconstruction cost is considered not the latent cost
        :param display_step: How many steps to execute for each display line
        :param learning_rate: Initial learning rate
        :param lr_decay_rate: When adjusting the learning rate, the factor to reduce it by
        :param sample_size: Number of samples to use when estimating expectations over the posterior
        :param ss_increase_factor: Factor to increase the sample size by over the epochs
        :param revert_post_trials: How many epoch to continue for without an improvement in the mean cost before
                                   reverting the posterior to the previous best parameters
        :param revert_post_final: If True, revert to the state giving the best cost achieved after the final epoch
        """
        self.log.info("Starting SVB inference")

        # Optional parameters that are only used occasionally
        display_step = kwargs.get("display_step", 10)
        lr_decay_rate = kwargs.get("lr_decay_rate", 1.0)
        ss_increase_factor = kwargs.get("ss_increase_factor", 1.0)
        self.revert_post_trials = kwargs.get("revert_post_trials", 50)
        revert_post_final = kwargs.get("revert_post_final", True)
        record_history = kwargs.get("record_history", False)
        self.history = {}

        # All the parameters to infer - model parameters. 
        # Noise is defined as a separate parameter in "voxel" space 
        # (never "node" - surface - as may be the case for the model) 
        self.noise_param = NoiseParameter()
        self._infer_covar = kwargs.get("infer_covar", False)
        self.mean_1, self.covar_1 = None, None

        # Create prior and posterior 
        self._create_prior_post(**kwargs)

        # Determine number of batches and sample size
        batch_size = kwargs.get("batch_size", self.n_tpts)
        n_batches = int(np.ceil(float(self.n_tpts) / batch_size))
        sample_size = kwargs.get("sample_size", 5)
        
        # Optional variable weighting of latent and reconstruction costs
        self.latent_weight = kwargs.get("latent_weight", 1.0)
        self.reconst_weight = kwargs.get("reconst_weight", 1.0)
        fit_only_epochs = kwargs.get("fit_only_epochs", 0)
        if fit_only_epochs > 0:
            self.latent_weight = 0.0

        # If required, create arrays to store cost and parameter histories
        if record_history:
            self.history = {
                "mean_cost" : np.zeros([epochs+1]),
                "reconst_cost" : np.zeros([self.n_voxels, epochs+1]),
                "latent_cost" : np.zeros([self.n_nodes, epochs+1]),
                "model_mean" : np.zeros([self.n_nodes, epochs+1, self.n_params]),
                "model_mean_mean" : np.zeros([epochs+1, self.n_params]),
                "noise_mean": np.zeros([self.n_voxels, epochs+1]),
                "runtime" : np.zeros([epochs+1]),
            }

        # Each epoch passes through the whole data but it may do this in 'batches' so there may be
        # multiple training iterations per epoch, one for each batch
        self.log.info(" - Number of training epochs: %i", epochs)
        self.log.info(" - %i voxels of %i time points (processed in %i batches of target size %i)" , self.n_voxels, self.n_tpts, n_batches, batch_size)
        self.log.info(" - Initial learning rate: %.5f", learning_rate)
        if lr_decay_rate < 1.0:
            self.log.info(" - Learning rate decay: %.3f", lr_decay_rate)
        self.log.info(" - Initial sample size: %i", sample_size)
        if ss_increase_factor > 1.0:
            self.log.info(" - Sample size increase factor: %.3f", ss_increase_factor)
        if self.revert_post_trials > 0:
            self.log.info(" - Posterior reversion after %i trials", self.revert_post_trials)

        # Log initial state
        self.cost(self.data, self.tpts, sample_size)
        self.epoch_latent = self.mean_latent_cost.numpy()
        self.epoch_reconst = self.mean_reconst_cost.numpy()
        self.epoch_cost = self.mean_cost.numpy().mean()
        self._log_epoch(0)

        # Main training loop
        #
        # Potential for a bug here: don't use range(1, epochs+1) as it 
        # interacts badly with training the ak parameter. 
        self.start_time = time.time()
        self.trials, self.best_cost, self.best_state = 0, self.epoch_cost, None
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(epochs):
            try:
                self.epoch_err = False
                self.epoch_latent, self.epoch_reconst, self.epoch_cost = 0, 0, 0

                if fit_only_epochs and epoch == fit_only_epochs:
                    # Once we have completed fit_only_epochs of training we will allow the latent cost to have
                    # an impact and reset the best cost accordingly.
                    self.latent_weight = 1.0
                    self.trials, self.best_cost = 0, 1e12

                # Iterate over training batches - note that there may be only one
                for i in range(n_batches):
                    if kwargs.get("sequential_batches", False):
                        # Batches are defined by sequential data time points
                        index = i*batch_size
                        if i == n_batches - 1:
                            # Batch size may not be an exact factor of the number of time points
                            # so make the last batch the right size so all of the data is used
                            batch_size += self.n_tpts - n_batches * batch_size
                        batch_data = self.data[:, index:index+batch_size]
                        batch_tpts = self.tpts[:, index:index+batch_size]
                    else:
                        # Batches are defined by constant strides through the data time points
                        # This automatically handles case where number of time point does not
                        # exactly divide into batches
                        batch_data = self.data[:, i::n_batches]
                        batch_tpts = self.tpts[:, i::n_batches]

                    # Perform a training iteration using batch data
                    all_vars = self.post.vars + self.prior.vars + self.noise_post.vars + self.noise_prior.vars
                    with tf.GradientTape(persistent=False) as t:
                        ecost = self.cost(batch_data, batch_tpts, sample_size)
                    gradients = t.gradient(ecost, all_vars)

                    # Apply the gradients
                    optimizer.apply_gradients(zip(gradients, all_vars))

                    self.epoch_latent += self.mean_latent_cost.numpy() / n_batches
                    self.epoch_reconst += self.mean_reconst_cost.numpy() / n_batches
                    self.epoch_cost += self.mean_cost.numpy() / n_batches

            except tf.errors.OpError:
                self.log.exception("Numerical error fitting batch")
                self.epoch_err = True

            # End of epoch
            if record_history:
                self._record_history(epoch)
            outcome = self._check_improvement()

            if (epoch+1) % display_step == 0:
                self._log_epoch(epoch+1, outcome)

        self.log.info(" - End of inference. ")
        if revert_post_final and self.best_state is not None:
            # At the end of training we revert to the state with best mean cost and write a final history step
            # with these values. Note that the cost may not be as reported earlier as this was based on a
            # mean over the training batches whereas here we recalculate the cost for the whole data set.
            self.log.info(" - Reverting to best batch-averaged cost")
            #self.set_state(best_state)

        # Record final history as it might have reverted and hence be different    
        if record_history:
            self._record_history(epoch)
        self.finalize()

    def _record_history(self, epoch):
        # Extract model and noise parameter estimates
        # The noise_mean is actually the 'best' estimate of noise variance, 
        # whereas noise_var is the variance of the noise vairance - yes, confusing!
        model_mean = self.model_mean.numpy().T # [W, P]
        noise_mean = self.noise_mean.numpy() # [V]

        # Store costs
        self.history["mean_cost"][epoch] = self.epoch_cost
        self.history["reconst_cost"][:, epoch] = self.epoch_reconst
        self.history["latent_cost"][:, epoch] = self.epoch_latent

        # Store parameter estimates:
        self.history["model_mean_mean"][epoch, :] = model_mean.mean(0)
        self.history["model_mean"][:, epoch, :] = model_mean
        self.history["noise_mean"][:, epoch] = noise_mean
        #sak, vak = self._extract_ak()
        #if sak.size: history["ak"]["surf"][epoch,:] = sak 
        #if vak.size: history["ak"]["vol"][epoch,:] = vak

        self.history["runtime"][epoch] = float(time.time() - self.start_time)

    def _check_improvement(self):
        if self.epoch_err or np.isnan(self.epoch_cost) or np.isnan(self.model_mean.numpy()).any():
            # Numerical errors while processing this epoch. Revert to best saved params if possible
            #if best_state is not None:
            #    self.set_state(best_state)
            outcome = "Revert - Numerical errors"
        elif self.epoch_cost < self.best_cost:
            # There was an improvement in the mean cost - save the current state of the posterior
            outcome = "Saving"
            best_cost = self.epoch_cost
            #best_state = self.state()
            trials = 0
        else:
            # The mean cost did not improve. 
            if self.revert_post_trials > 0:
                # Continue until it has not improved for revert_post_trials epochs and then revert 
                self.trials += 1
                if self.trials < self.revert_post_trials:
                    outcome = "Trial %i" % self.trials
                elif self.best_state is not None:
                    #self.set_state(best_state)
                    outcome = "Revert"
                    trials = 0
                else:
                    outcome = "Continue - No best state"
                    trials = 0
            else:
                outcome = "Not saving"
        
        return outcome

    def _log_epoch(self, epoch, outcome=""):
        self.log.info(" - Epoch %04d - Outcome: %s" % (epoch, outcome))
        self.log.info("   - Mean: %s" % self.model_mean.numpy().mean(1))
        self.log.info("   - Variance: %s" % self.model_var.numpy().mean(1))
        self.log.info("   - Cost: %.4g (latent %.4g, reconst %.4g)" % (self.epoch_cost, self.epoch_latent, self.epoch_reconst))

    def _create_input_tensors(self):
        """
        Tensorflow input required for training
        
        x will have shape VxB where B is the batch size and V the number of voxels
        xfull is the full data so will have shape VxT where N is the full time size
        tpts_train will have shape 1xB or VxB depending on whether the timeseries is voxel
        dependent (e.g. in 2D multi-slice readout)
        
        NB we don't know V, B and T at this stage so we set placeholder variables
        self.n_voxels and self.n_tpts and use validate_shape=False when creating
        tensorflow Variables
        """
        # Full data - we need this during training to correctly scale contributions
        # to the cost
        self.data_full = tf.constant(self.data_model.data_space.srcdata.flat, dtype=TF_DTYPE, name="data_full")

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

        # Amount of weight given to latent cost in cost function (0-1)
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
        # the latent cost - so set this flag to decide if this is possible
        self.analytic_latent_cost = (np.all([ p.is_gaussian for p in all_priors ]) 
                                     and not nongaussian_posts 
                                     and not kwargs.get("force_num_latent_cost", False)
                                     and False)   # FIXME always disabled for now 
        if self.analytic_latent_cost:
            self.log.info(" - Using analytical expression for latent cost since prior and posterior are Gaussian")
        else:
            self.log.info(" - Using numerical calculation of latent cost")

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
        model_samples, model_mean, model_var = [], [], []
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
            model_mean.append(ext_means)
            model_var.append(ext_vars)

        # The timepoints tensor has shape [V x B] or [1 x B]. It needs to be reshaped
        # to [V x 1 x B] or [1 x 1 x B] so it can be broadcast across each of the S samples
        sample_tpts = tf.expand_dims(tpts, 1)
        
        # Produce a model prediction for each set of transformed parameter samples passed in 
        # Model prediction has shape [W x S x B]
        self.model_samples = model_samples
        self.sample_predictions = self.fwd_model.evaluate(model_samples, sample_tpts)

        # Define convenience tensors for querying the model-space sample, means and prediction. 
        # Modelfit_nodes has shape [W x B]. Produce a current "best estimate" prediction from 
        # model_mean. This is distinct to producing a prediction for each samples. 
        self.model_mean = tf.identity(model_mean)
        self.model_var = tf.identity(model_var)
        self.modelfit = self.fwd_model.evaluate(tf.expand_dims(self.model_mean, -1), tpts)

        return self.sample_predictions

    def _cost(self, data, tpts, sample_size):
        """
        Create the cost optimizer which will minimise the cost function

        The cost is composed of two terms:

        1. log likelihood. This is a measure of how likely the data are given the
           current posterior, i.e. how well the data fit the model using
           the inferred parameters.

        2. The latent cost. This is a measure of how closely the posterior fits the
           prior
        """
        # Generate a set of model parameter samples from the posterior [W x P x B]
        # Generate a set of noise parameter samples from the posterior [W x 2 x B]
        # They are in the 'internal' form required by the distributions themselves
        param_samples_int = self.post.sample(sample_size)
        noise_samples_int = self.noise_post.sample(sample_size)

        # Part 1: Reconstruction cost
        #
        # This deals with how well the parameters replicate the data and is 
        # defined as the log-likelihood of the data (given the parameters).
        #
        # This is calculated using the noise model. For each prediction generated
        # from the samples, the difference between it and the original data is 
        # taken to be an estimate of the noise ('residual' cost). This noise estimate
        # is compared with the current noise posterior in each voxel (via samples
        # drawn from the posterior and calculation of the log likelihoood). 
        # 
        # If the current model parameters are poor, then the model prediction
        # will not reproduce the data well. The high residual cost between the
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
        model_prediction_voxels = self.data_model.model_to_data(model_prediction)

        # Convert the noise samples from from internal to external representation.
        # Save the current moments of the noise posterior. 
        transformer = self.noise_param.post_dist.transform
        noise_samples_ext = tf.squeeze(transformer.ext_values(noise_samples_int))
        noise_mean_ext, noise_var_ext = transformer.ext_moments(
                                            self.noise_post.mean, self.noise_post.var)
        self.noise_mean = noise_mean_ext
        self.noise_var = noise_var_ext

        # Reconstruction cost calculated here: the log-likelihood of the difference between 
        # the data and the model predictions. Calculation of the log-likelihood requires 
        # the properties of the noise distribution, which we get from the noise paramter
        # samples. Finally, we average this over voxels. 
        reconst_cost = self.noise_param.log_likelihood(data, 
                            model_prediction_voxels, noise_samples_ext, self.n_tpts)
        self.reconst_cost = reconst_cost
        self.mean_reconst_cost = tf.reduce_mean(self.reconst_cost, name="mean_reconst_cost")

        # Part 2: Latent cost
        #
        # This penalises parameters which are far from the prior
        # If both the prior and posterior are represented by an MVN we can calculate an analytic
        # expression for this cost. If not, we need to do it numerically using the posterior
        # sample obtained earlier. Note that the mean log pdf of the posterior based on sampling
        # from itself is just the distribution entropy so we allow it to be calculated without
        # sampling.
        if self.analytic_latent_cost: 
            # FIXME: this will not be able to handle mixed voxel/node priors. 
            latent_cost = tf.identity(self.post.latent_cost(self.prior), name="latent_cost")
        else:

            # Latent cost is calculated over model parameters and noise parameters separately 
            # Note that these may be sized differently (one over nodes, the other voxels)
            self.param_latent_cost = tf.subtract(
                                    self.post.entropy(param_samples_int), 
                                    self.prior.mean_log_pdf(param_samples_int), 
                                    name="param_latent_cost")
            self.noise_latent_cost = tf.subtract(
                                    self.noise_post.entropy(noise_samples_int), 
                                    self.noise_prior.mean_log_pdf(noise_samples_int),
                                    name="noise_latent_cost")

        # Reduce the latent costs over their voxel/node dimension to give an average.
        # This deals with the situation where they may be sized differently. The overall
        # latent cost is then the sum of the two averages 
        self.mean_latent_cost = tf.add(
            tf.reduce_mean(self.param_latent_cost), 
            tf.reduce_mean(self.noise_latent_cost),
            name="mean_latent_cost")

        # We have the possibility of modifying the weighting of the latent and
        # reconstruction costs. This can be used for debugging, also there is
        # a theory that fitting can sometimes be improved by gradually increasing
        # the latent cost - i.e. let the model fit the data first and then allow 
        # the fit to be perturbed by the priors.
        self.mean_latent_cost = self.mean_latent_cost * self.latent_weight
        self.mean_reconst_cost = self.mean_reconst_cost * self.reconst_weight

        # Overall mean cost is taken as the sum of (mean voxel reconstruction cost) 
        # and the sum ((average latent over model) + (average latent over noise))
        self.mean_cost = tf.add(self.mean_reconst_cost, self.mean_latent_cost)

        return self.mean_cost
    
    def cost(self, data, tpts, sample_size):
        self.prior.build()
        self.post.build()
        self.noise_prior.build()
        self.noise_post.build()
        return self._cost(data, tpts, sample_size)

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
