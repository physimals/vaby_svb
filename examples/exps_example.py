"""
Example inferring multiple exponential decay models arranged into a
4D voxelwise image.

This example uses the main() interface as used by the command line
application to simplify running the inference and saving the output
"""
import sys

import numpy as np
import nibabel as nib

import vaby 

# Uncomment line below to start the random number generator off with the same seed value
# each time, for repeatable results.
#np.random.seed(0)

# Ground truth parameters
PARAMS_TRUTH = [42, 0.5]
NOISE_PREC_TRUTH = 0.1
NOISE_VAR_TRUTH = 1/NOISE_PREC_TRUTH
NOISE_STD_TRUTH = np.sqrt(NOISE_VAR_TRUTH)

# Observed data samples are generated by Numpy from the ground truth
# Gaussian distribution. Reducing the number of samples should make
# the inference less 'confident' - i.e. the output variances for
# MU and BETA will increase
N = 100
DT = 2.0 / N
NX, NY, NZ = 10, 10, 10
t = np.array([float(t)*DT for t in range(N)])
params_voxelwise = np.tile(np.array(PARAMS_TRUTH)[..., np.newaxis, np.newaxis], (1, NX*NY*NZ, 1))
temp_model = vaby.get_model_class("exp")(None, dt=DT)
DATA_CLEAN = temp_model.evaluate(params_voxelwise, t).numpy()
DATA_NOISY = DATA_CLEAN + np.random.normal(0, NOISE_STD_TRUTH, DATA_CLEAN.shape)
niidata = DATA_NOISY.reshape((NX, NY, NZ, N))
nii = nib.Nifti1Image(niidata, np.identity(4))
nii.to_filename("data_exp_noisy.nii.gz")

# Run Fabber as a comparison if desired
#import os
#os.system("fabber_exp --data=data_exp_noisy  --max-iterations=20 --output=exps_example_fabber_out --dt=%.3f --model=exp --num-exps=1 --method=vb --noise=white --overwrite" % DT)

options = {
    "method" : "svb",
    "dt" : DT,
    "save_mean" : True,
    #"save_free_energy" : True,
    #"save_model_fit" : True,
    "save_log" : True,
    "log_stream" : sys.stdout,
    "epochs" : 1000,
    "learning_rate" : 0.1,
    "sample_size" : 5,
}

runtime, avb = vaby.run("data_exp_noisy.nii.gz", "exp", "exps_example_out", **options)
