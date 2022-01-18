"""
Example of usage of the AVB framework to infer a single exponential decay
model.

This uses the Python classes directly to infer the parameters for a single
instance of noisy data constructed as a Numpy array.
"""
import sys
import logging

import numpy as np

from vaby_svb import Svb
import vaby

# Uncomment line below to start the random number generator off with the same seed value
# each time, for repeatable results
#np.random.seed(0)

# Ground truth parameters
PARAMS_TRUTH = [42, 0.5]
NOISE_PREC_TRUTH = 0.1
NOISE_VAR_TRUTH = 1/NOISE_PREC_TRUTH
NOISE_STD_TRUTH = np.sqrt(NOISE_VAR_TRUTH)
print("Ground truth: a=%f, r=%f, noise=%f (precision)" % (PARAMS_TRUTH[0], PARAMS_TRUTH[1], NOISE_PREC_TRUTH))
# Create single exponential model
model = vaby.get_model_class("exp")(None)

# Observed data samples are generated by Numpy from the ground truth
# Gaussian distribution. Reducing the number of samples should make
# the inference less 'confident' - i.e. the output variances for
# MU and BETA will increase
N = 100
DT = 0.02
t = np.array([float(t)*DT for t in range(N)])
DATA_CLEAN = model.evaluate(PARAMS_TRUTH, t).numpy()
DATA_NOISY = DATA_CLEAN + np.random.normal(0, NOISE_STD_TRUTH, [N])
print("Time values:")
print(t)
print("Data samples (clean):")
print(DATA_CLEAN)
print("Data samples (noisy):")
print(DATA_NOISY)
data_model = vaby.DataModel(DATA_NOISY)

# Run Fabber as a comparison if desired
#import os
#import nibabel as nib
#niidata = DATA_NOISY.reshape((1, 1, 1, N))
#nii = nib.Nifti1Image(niidata, np.identity(4))
#nii.to_filename("data_noisy.nii.gz")
#os.system("fabber_exp --data=data_noisy --print-free-energy --output=fabberout --dt=%.3f --model=exp --num-exps=1 --method=vb --noise=white --overwrite --debug" % DT)

# Log to stdout
logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
logging.getLogger().addHandler(handler)

# Run AVB inference
fwd_model = vaby.get_model_class("exp")(data_model, dt=DT)
avb = Svb(data_model, fwd_model)
avb.run(epochs=300, learning_rate=0.1, sample_size=10, debug="--debug" in sys.argv)
