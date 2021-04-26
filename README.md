# A lesion-level deep learning approach to predict enhancing lesions from non-enhanced MR images in Multiple Sclerosis

This is the code accompanying our manuscript "A lesion-level deep learning approach to predict enhancing lesions from non-enhanced MR images in Multiple Sclerosis".

> **generate_patches_proval.py** is the script responsible for creating 16x16x16 patches (suitable as input to training / prediction) from 3D NIfTI images (co-registered FLAIR and T1w plus a binary segmentation map). A T1w image with contrast enhancement is optional and only used to gather data on contrast enhancement (50th and 75th percentile of lesion intensity on subtraction [t1c-t1] images).

> **densenet3d_ms.py** is the script reponsible for definining the architecture and training a classifier from the patches generated above. By default, training is run 5 times over 250 epochs each run, with three models saved (final model, highest valid auc, highest valid f1 score), yielding a total of 15 models. The model is written in Keras / Tf2.4
