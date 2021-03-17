import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import random
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy import ndimage
from tensorflow.keras.utils import Sequence
import gc
import pandas as pd

tf.config.threading.set_inter_op_parallelism_threads(24)
tf.config.threading.set_intra_op_parallelism_threads(24)

"""
3D DenseNet for prediction of ce in MS lesion patches
"""

#Hyperparameters
blocks_lst = [6, 12, 18] #
k = 32. # Growth rate
reduction = 0.5 #Decrease of feature maps at each transition
dropout_rate = 0. # 0. to disable
batch_size = 100
n_epochs = 250

def dense_block(layer_, blocks):
	"""A dense block.
	# Arguments
		layer_: input tensor.
		blocks: integer, the number of building blocks.
	# Returns
		output tensor for the block.
	"""
	for i in range(blocks):
		layer_ = conv_block(layer_, k)
	return layer_

def transition_block(layer_):
	"""A transition block.
	# Arguments
		layer_: input tensor.
	# Returns
		output tensor for the block.
	"""
	num_feature_maps = layer_.shape[-1] # The value of 'm'
	layer_ = tf.keras.layers.BatchNormalization() (layer_)
	layer_ = tf.keras.layers.LeakyReLU() (layer_)
	layer_ = tf.keras.layers.Conv3D(int(num_feature_maps * reduction), kernel_size=1, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer='l2') (layer_)
	if dropout_rate > 0:
		layer_ = tf.keras.layers.Dropout(rate=dropout_rate) (layer_)
	return layer_

def conv_block(input_, growth_rate):
	"""A building block for a dense block.
	# Arguments
		input_: input tensor.
		growth_rate: float, growth rate at dense layers.
	# Returns
		Output tensor for the block.
	"""

	layer_ = tf.keras.layers.BatchNormalization() (input_)
	layer_ = tf.keras.layers.LeakyReLU() (layer_)
	layer_ = tf.keras.layers.Conv3D(4 * growth_rate, kernel_size=1, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer='l2') (layer_) #This is the bottleneck layer, i.e. kernel_size=1

	layer_ = tf.keras.layers.BatchNormalization() (layer_)
	layer_ = tf.keras.layers.LeakyReLU() (layer_)
	layer_ = tf.keras.layers.Conv3D(growth_rate, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False, kernel_regularizer='l2') (layer_)
	if dropout_rate > 0:
		layer_ = tf.keras.layers.Dropout(rate=dropout_rate) (layer_)

	layer_ = tf.keras.layers.Concatenate()([input_, layer_])

	return layer_

def DenseNet(blocks, input_shape):
	"""Instantiates the DenseNet architecture.
	# Arguments
		blocks: numbers of building blocks for the four dense layers.
	# Returns
		A Keras model instance.
	"""
	img_input = tf.keras.layers.Input(shape=input_shape)

	layer = dense_block(img_input, blocks[0])
	layer = transition_block(layer)
	layer = dense_block(layer, blocks[1])
	layer = transition_block(layer)
	layer = dense_block(layer, blocks[2])
	#layer = transition_block(layer)
	#layer = dense_block(layer, blocks[3])

	layer = tf.keras.layers.BatchNormalization() (layer)
	layer = tf.keras.layers.LeakyReLU() (layer)

	layer = tf.keras.layers.GlobalAveragePooling3D() (layer)
	output = tf.keras.layers.Dense(1, activation='sigmoid') (layer)

	model = tf.keras.Model(img_input, output, name='densenet_3d')

	return model

class DenseSequence_3D(Sequence):
	"""
	Custom data generator for 3D-DenseNet
	"""
	def __init__(self, path, batch_size, augment=True):
		"""
		path: string pointing to the directory where the images (in subfolders "enhancing/" and "non_enhancing/") are stored.
		augment: Bool whether to use augmentation (train) or not (valid/test)
		"""
		self.path = path
		enhancing = os.listdir(path + "/enhancing/")
		self.enhancing = [datei for datei in enhancing if "-f2" in datei]
		non_enhancing = os.listdir(path + "/non_enhancing/")
		self.non_enhancing = [datei for datei in non_enhancing if "-f2" in datei]
		self.batch_size = batch_size
		self.augment = augment

	def __len__(self):
		return (len(self.non_enhancing)*2) // self.batch_size #Because we oversample the minority class (enhancing) to match 1:1

	def __getitem__(self, idx):
		"""
		Loads a list of batch_size images (evenly sampled between (non-)enhancing)
		Returns a rank-5-array [batch_size,width,heigth,depth,channels] and a y_vector of len(batch_size) [0,1]
		"""
		y_vector = []
		img_list = []

		for i in range(self.batch_size):
			if random.random() > 0.5:
				random_file = random.choice(self.enhancing)
				f2_file = nib.load(self.path + "/enhancing/" + random_file)
				t1_file = nib.load(self.path + "/enhancing/" + random_file.replace("-f2","-t1"))
				y_vector.append(1.)
			else:
				random_file = random.choice(self.non_enhancing)
				f2_file = nib.load(self.path + "/non_enhancing/" + random_file)
				t1_file = nib.load(self.path + "/non_enhancing/" + random_file.replace("-f2","-t1"))
				y_vector.append(0.)

			f2 = f2_file.get_fdata()
			f2 = np.divide(f2,f2.max())

			t1 = t1_file.get_fdata()
			t1 = np.divide(t1,t1.max())

			if self.augment: #Now with scipy-based custom augmentations
				#Augmentation hyperparameters
				max_angle = 45 #Maximum rotation in degrees
				max_sigma = 1. #For the guassian filter

				#Morphological augmentations
				if random.random() > 0.25:
					angle = random.randint(-max_angle,max_angle)
					axes = random.sample((0,1,2),2)
					f2 = ndimage.rotate(f2,angle=angle,axes=axes,reshape=False,order=1,mode="constant",cval=0.0)
					t1 = ndimage.rotate(t1,angle=angle,axes=axes,reshape=False,order=1,mode="constant",cval=0.0)

				if random.random() > 0.25:
					axis = random.sample((0,1,2),1)
					f2 = np.flip(f2,axis=axis)
					t1 = np.flip(t1,axis=axis)

				if random.random() > 0.25:
					axis = random.sample((0,1,2),1)
					f2 = np.flip(f2,axis=axis)
					t1 = np.flip(t1,axis=axis)

				#Intensity augmentations
				if random.random() > 0.25:
					sigma = random.uniform(0,max_sigma)
					f2 = ndimage.gaussian_filter(f2,sigma=sigma)
					t1 = ndimage.gaussian_filter(t1,sigma=sigma)

			img_image = np.stack((f2,t1),axis=-1)
			img_image = np.expand_dims(img_image,axis=(0))
			img_list.append(img_image)

		return np.concatenate(img_list,axis=0), np.array(y_vector)

learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
	initial_learning_rate=0.01,
	decay_steps=n_epochs,
	end_learning_rate=0.0001,
	power=0.9)

class FreeMemOnEpochEnd(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		gc.collect()

class SaveBestModel(tf.keras.callbacks.Callback):
	def __init__(self,ctr):
		self.best_f1 = 0.
		self.best_auc = 0.
		self.ctr = ctr

	def on_epoch_end(self, epoch, logs=None):
		f1_ = 2 / ( ((logs["val_precision"]+0.000000000001) ** -1) + ((logs["val_recall"]+0.000000000001) ** -1) )
		
		if (f1_ > self.best_f1) and (epoch > 5):
			self.model.save("/home/benewiestler/DenseNet3D_MS_highest_valid_f1_run" + str(self.ctr) + ".h5")      
			self.best_f1 = f1_

		if (logs["val_auc"] > self.best_auc) and (epoch > 5):
			self.model.save("/home/benewiestler/DenseNet3D_MS_highest_valid_auc_run" + str(self.ctr) + ".h5")      
			self.best_auc = logs["val_auc"]

for ctr in range(5):

	model = DenseNet(blocks_lst,(16,16,16,2))

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
					loss=tf.keras.losses.BinaryCrossentropy(), metrics=["AUC","Precision","Recall"])
	mdl_history = model.fit(DenseSequence_3D("/mnt/Drive1/bene/ms_patches_new/16/train/",batch_size=batch_size,augment=True),
							validation_data=DenseSequence_3D("/mnt/Drive1/bene/ms_patches_new/16/valid/",batch_size=batch_size,augment=False),
							epochs=n_epochs,callbacks=[FreeMemOnEpochEnd(),SaveBestModel(ctr+3)])
	pd.DataFrame(mdl_history.history).to_csv("/home/benewiestler/MS_DenseNet3D_history_run" + str(ctr) + ".csv")
	model.save("/home/benewiestler/DenseNet3D_MS_final_run" + str(ctr) + ".h5")