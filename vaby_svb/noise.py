"""
Noise model
"""
import tensorflow as tf
   
from vaby.parameter import Parameter
from vaby import dist
from vaby.utils import TF_DTYPE

class NoiseParameter(Parameter):
    """
    Noise parameter providing Gaussian (white) noise. Note that by default, 
    noise is always assumed to be volumetric in origin (data_space="voxel"). 
    """

    def __init__(self, data_space="voxel", **kwargs):
        Parameter.__init__(self, "noise",
                           prior=dist.LogNormal(1.0, 2e5),
                           post=dist.LogNormal(1.0, 1.02),
                           post_init=self._init_noise,
                           data_space=data_space,
                           **kwargs)

    def _init_noise(self, _param, _t, data):
        _data_mean, data_var = tf.nn.moments(tf.constant(data), axes=1)
        return tf.where(tf.math.less(data_var, 1), tf.ones_like(data_var), data_var), None

    def log_likelihood(self, data, pred, noise_var, nt):
        """
        Calculate the log-likelihood of the data

        :param data: Tensor of shape [V, B]
        :param pred: Model prediction tensor with shape [V, S, B]
        :param noise: Noise parameter samples tensor with shape [V, S]
        :return: Tensor of shape [V] containing mean log likelihood of the 
                 data at each voxel with respect to the noise parameters
        """
        nvoxels = tf.shape(data)[0]
        batch_size = tf.shape(data)[1]
        sample_size = tf.shape(pred)[1]

        # Possible to get zeros when using surface projection, in which case replace them
        # with ones (required for log transforms)
        noise_var = tf.where(tf.equal(noise_var, 0), tf.ones_like(noise_var), noise_var)
        log_noise_var = tf.math.log(noise_var)

        # Expand the dimensions of data to match that of predictions (insert samples axis)
        data = tf.tile(tf.reshape(data, [nvoxels, 1, batch_size]), [1, sample_size, 1])

        # Squared differences between data and predictions, shape [V, S, B],
        # aka residuals between data and prediction. Sum along the batch dimension 
        square_diff = tf.square(data - pred)
        sum_square_diff = tf.reduce_sum(square_diff, axis=-1)

        # Since we are processing only a batch of the data at a time, we need to scale the 
        # sum of squared differences term correctly. 'nt' is already the full data
        # size, so we multiply up the residuals for this batch by (nt / batch) size, ie, 
        # scale it as if the batch contained all the data 
        scale = tf.divide(tf.cast(nt, TF_DTYPE), tf.cast(batch_size, TF_DTYPE))

        # Log likelihood has shape [V, S]
        log_likelihood = 0.5 * (log_noise_var * tf.cast(nt, TF_DTYPE) +
                                scale * sum_square_diff / noise_var)

        # Mean over samples - reconstr_loss has shape [V]
        return tf.reduce_mean(log_likelihood, axis=1)
