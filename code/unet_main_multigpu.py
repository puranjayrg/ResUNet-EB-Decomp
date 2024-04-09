#--------
# Imports
#--------
import matplotlib
from tensorflow.python.keras.callbacks import ModelCheckpoint
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.preprocessing import normalize

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

import healpy as hp
import scipy as sc
import h5py
import pathlib

import segmentation_models as sm

keras.backend.set_image_data_format('channels_last')

print('Imports complete', flush=True)

#-------------
# Data Loading
#-------------
res_path = pathlib.Path("/home/puranjay/projects/def-hinshaw/puranjay/results/resunet_realb/sm/run1cut_11-04-22/250ep")
data_path = pathlib.Path("/home/puranjay/projects/def-hinshaw/puranjay/data")

# tune_path = pathlib.Path("/home/puranjay/projects/def-hinshaw/puranjay/results/resunet_realb/sm/run6_10-03-22/hptuner")
# resfolder = pathlib.Path("/home/puranjay/projects/def-hinshaw/puranjay/results/resunet_realb")

res_path.mkdir(parents=True, exist_ok=True)

tf.config.list_physical_devices('GPU')

print('Loading training + test data...', flush=True)

with h5py.File(os.path.join(data_path, 'traindata_1patch20deg_ns1024_12k5.h5'), 'r') as f:
    print('Opened h5 file, now getting data...', flush=True)
    dset_xtrain = f.get('train/x_train')
    dset_ytrain = f.get('train/y_train')
    dset_xtest = f.get('test/x_test')
    dset_ytest = f.get('test/y_test')
    for name in f:
        print(name)
    print('Dsets obtained, now saving as numpy arrays...', flush=True)
    x_trainingdata = dset_xtrain[:]
    y_trainingdata = dset_ytrain[:]
    x_test = dset_xtest[:]
    y_test = dset_ytest[:]
    print('Everything saved, with loop end, file should auto close now', flush=True)
print(x_trainingdata.shape, flush=True)
print(y_trainingdata.shape, flush=True)
print(x_test.shape, flush=True)
print(y_test.shape, flush=True)
 
x_train, x_val, y_train, y_val = train_test_split(x_trainingdata, y_trainingdata, 
                                                  test_size=0.1, random_state=2)
print('Training data loaded', flush=True)

#--------------
# OPTIONAL 
# Preprocessing
#--------------

# print('Now preprocess the training data...', flush=True)

# def ppnorm(x):
#     ppmu = np.mean(x, axis=(0,1,2))
#     ppstd = np.std(x, axis=(0,1,2))

#     x = (x - ppmu)/ppstd
#     return x

# x_train = ppnorm(x_train)
# y_train = ppnorm(y_train)
# x_val = ppnorm(x_val)
# y_val = ppnorm(y_val)

#x_train = preprocess_input(x_train)
#x_val = preprocess_input(x_val)
#y_train = preprocess_input(y_train)
#y_val = preprocess_input(y_val)

print('Training data preprocessed (no preprocessing)', flush=True)
# print('Here\'s what it looks like:\nxtr:{}\nytr:{}\nxv:{}\nyv:{}'.format(x_train[0], y_train[0], x_val[0], y_val[0]), flush=True)


# Vars for naming results
datasize = x_trainingdata.shape[0] + x_test.shape[0] # Total training data size (train + val + test)
ns = 1024 # N_side for the input data

print('Data size (used for naming results) is: {}'.format(datasize), flush=True)



#----------------------
# OPTIONAL 
# Simple Model definition
# Used for testing 
# NOT the final model
#----------------------
# print('Now defining ResUNet...', flush=True)

# def resi_block(x_in, n_filters, kernel_size=(7,7), padding="same", act='selu', strides=1, first=False, input_shape=(256,256,2)):
    
#     if first:
#         x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, strides=strides, input_shape=input_shape)(x_in)
#         x = BatchNormalization()(x)
#         x = Activation(act)(x)
#         x = Dropout(0.3)(x)
#         x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, strides=1)(x)
#     else:
#         x = BatchNormalization()(x_in)
#         x = Activation(act)(x)
#         #Skip above 2 if it's the very first encoding block
#         x = Dropout(0.3)(x)
#         x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, strides=strides)(x)
#         x = BatchNormalization()(x)
#         x = Activation(act)(x)
#         x = Dropout(0.3)(x)
#         x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding=padding, strides=1)(x)
    
#     #residual shortcuts
#     sc = Conv2D(filters=n_filters, kernel_size=(1,1), padding=padding, strides=strides, input_shape=input_shape)(x_in)
#     sc = BatchNormalization()(sc)

#     x = Add()([x, sc])

#     return x


# def upsampconcat(x, x_enc):
#     u = UpSampling2D((2, 2))(x)
#     c = Concatenate()([u, x_enc])
#     return c

# def ResUNet():
#     filter_list = [64, 128, 256]
#     print(filter_list[0])
#     input_shape = (256, 256, 2)
#     input_maps = Input(input_shape)

#     enc1 = resi_block(input_maps, n_filters=filter_list[0], kernel_size=(7,7), strides=1, first=True, input_shape=input_shape)
#     enc2 = resi_block(enc1, n_filters=filter_list[1], kernel_size=(7,7), strides=2)
#     enc3 = resi_block(enc2, n_filters=filter_list[2], kernel_size=(7,7), strides=2)

#     bridge = resi_block(enc3, n_filters=filter_list[2], kernel_size=(7,7), strides=2)

#     upskip1 = upsampconcat(bridge, enc3)
#     dec1 = resi_block(upskip1, n_filters=filter_list[2], kernel_size=(7,7), strides=1)
#     upskip2 = upsampconcat(dec1, enc2)
#     dec2 = resi_block(upskip2, n_filters=filter_list[1], kernel_size=(7,7), strides=1)
#     upskip3 = upsampconcat(dec2, enc1)
#     dec3 = resi_block(upskip3, n_filters=filter_list[0], kernel_size=(7,7), strides=1)

#     output_maps = Conv2D(2, (7,7), padding='same', strides=1, activation='linear')(dec3)

#     model = Model(input_maps, output_maps)
#     return model

print('clearing session..', flush=True)
K.clear_session()


#-----------------------------
# Model Definition & Compiling
#-----------------------------

# Hyperparams settings
# ---
n_ep = 250
b_size = 32
run = '1cut'

# Callbacks
# ---
es_patience = 10
lr_fac = 0.5 # halve the lr
lr_patience = 3

csv_logger = CSVLogger(os.path.join(res_path, 
                                    'unetsm_training_smallb_run_{}.log'.format(run)),
                                      separator=',', append=False)

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                          min_delta=0, 
                                          patience=es_patience, 
                                          restore_best_weights=True)

droplr = ReduceLROnPlateau(monitor='val_loss', 
                           mode='min', 
                           factor=lr_fac, 
                           patience = lr_patience, 
                           verbose=1)

best_ep_mae = ModelCheckpoint(os.path.join(res_path, 
                            "unetsm_{}ep_{}_ns{}_bestwts_mae.h5".format(n_ep, datasize, ns)), 
                            monitor='val_MAE',
                            mode='min',
                            save_best_only=True,
                            verbose=1)

best_ep_mse = ModelCheckpoint(os.path.join(res_path, 
                            "unetsm_{}ep_{}_ns{}_bestwts_mse.h5".format(n_ep, datasize, ns)), 
                            monitor='val_MSE',
                            mode='min',
                            save_best_only=True,
                            verbose=1)

callbacks = [csv_logger, best_ep_mae, best_ep_mse]
# ---
# Callbacks defined, now compiling model


# Using tuned hyperparameters from kerastuner Hyperband search
# ---
print('Setting sm backbone...', flush=True)
BACKBONE = 'inceptionresnetv2' # Optimal BACKBONE obtained from hyperparameter tuning

print('Backbone is:{}'.format(BACKBONE), flush=True)

lr_init = 0.0033455 # Optimal lr obtained from hyperparameter tuning

print('Now defining UNet from segmentation_models...', flush=True)

# Setting up multi-GPUs for training
# Using 4 NVidia V100L Volta GPUs in TF mirrored strategy
# ---
strategy = tf.distribute.MirroredStrategy() 
with strategy.scope():
    model = sm.Unet(BACKBONE, 
                    input_shape=(256, 256, 2), 
                    encoder_weights=None, 
                    classes=2, 
                    activation='linear')
    
    opt = Adam(learning_rate=lr_init)

    model.compile(optimizer=opt, loss='MSE', metrics=['MAE', 'MSE'])

    print('Model compiled', flush=True)

# Saving model architecture
plot_model(model, to_file=os.path.join(res_path, 'resunet_arch.png'), show_layer_names=True, show_shapes=True)
plot_model(model, to_file=os.path.join('/home/puranjay/ubc/research/1ebsep', 'resunet_arch_fixed.png'), show_layer_names=True, show_shapes=True)
print('Model architecture graph saved', flush=True)

    
#---------------
# Model Training
#---------------

print('Starting training now...', flush=True)

hist = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    batch_size=b_size,
    epochs=n_ep,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

print('training completed', flush=True)

model.save(os.path.join(res_path, 'resunet_{}ep_{}_ns{}_model.h5'.format(n_ep, datasize, ns)))
model.save_weights(os.path.join(res_path, 'resunet_{}ep_{}_ns{}_finalwts.h5'.format(n_ep, datasize, ns)))

#-----------------------------
# Model Evaluation and Testing
#-----------------------------

# No need to save loss since I am CSV logging all loss vals, 
# (essesntially saving hist.history object as a CSV)
# Can get loss curve from there.

# Plotting loss curves
# ---
fig2 = plt.figure()
plt.plot(hist.history['loss']);
plt.xlabel('Epoch');
plt.ylabel('Loss');
fig2.savefig(os.path.join(res_path, "resunet_{}ep_{}_ns{}_loss.png".format(n_ep, datasize, ns)))

fig3 = plt.figure()
plt.plot(hist.history['val_loss']);
plt.xlabel('Epoch');
plt.ylabel('Validation Loss');
fig3.savefig(os.path.join(res_path, "resunet_{}ep_{}_ns{}_valloss.png".format(n_ep, datasize, ns)))


# Prediction on Test Set
# ---
print('Predicting on test data, final then best...', flush=True)

score1 = model.evaluate(x_test, y_test, verbose=2)
print('Score and acc for ResUNet on *final* weights is:{}'.format(score1), flush=True)

print('Now load the best MAE weights...', flush=True)
model.load_weights(os.path.join(res_path, "unetsm_{}ep_{}_ns{}_bestwts_mae.h5".format(n_ep, datasize, ns)))

score2 = model.evaluate(x_test, y_test, verbose=2)
print('Score and acc for ResUNet on *best* weights is:{}'.format(score2), flush=True)

print('Now predict using best weights...', flush=True)

decoded_imgs = model.predict(x_test)

residuals = y_test - decoded_imgs # actual - predicted
#should have shape (ntest, 256,256, 2)

np.save(os.path.join(res_path, 'yhatrealb_resunet_ns{}_20deg_{}_mae_bestwts.npy'.format(ns, datasize)), decoded_imgs)


#-----------------
# Plotting Results
#-----------------

# shuffle before plotting so that random test examples are picked for plotting, 
# instead of having all 4 patches from a single realization.

qmin0 = np.amin(np.concatenate((x_test[:10,:,:,0], y_test[:10,:,:,0])))
qmax0 = np.amax(np.concatenate((x_test[:10,:,:,0], y_test[:10,:,:,0])))

umin0 = np.amin(np.concatenate((x_test[:10,:,:,1], y_test[:10,:,:,1])))
umax0 = np.amax(np.concatenate((x_test[:10,:,:,1], y_test[:10,:,:,1])))

qmin1 = np.amin(np.concatenate((y_test[:10,:,:,0])))
qmax1 = np.amax(np.concatenate((y_test[:10,:,:,0])))

umin1 = np.amin(np.concatenate((y_test[:10,:,:,1])))
umax1 = np.amax(np.concatenate((y_test[:10,:,:,1])))



# Q plot
# ---
print('Starting Q plot...', flush=True)

n = 10
fig4, axes = plt.subplots(nrows=3, ncols=10, figsize=(25,8))
for i, ax in enumerate(axes.flat):
    if i < 10:
        im = ax.imshow(x_test[i,:,:,0], vmin=qmin0, vmax=qmax0)
    elif i < 20:
        im = ax.imshow(y_test[i-10,:,:,0], vmin=qmin1, vmax=qmax1)
    else:
        im = ax.imshow(decoded_imgs[i-20,:,:,0], vmin=qmin1, vmax=qmax1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig4.subplots_adjust(right=0.95)
cbar_ax = fig4.add_axes([0.97, 0.2, 0.02, 0.6]) # [left bot width height]
fig4.colorbar(im, cax=cbar_ax)

fig4.savefig(os.path.join(res_path, 
                        "unetsm_{}ep_{}_ns{}_qbw.png".format(n_ep, datasize, ns)),
                        bbox_inches='tight')

# U plot
# ---
print('Starting U plot...', flush=True)
n = 10
fig5, axes = plt.subplots(nrows=3, ncols=10, figsize=(25,8))
for i, ax in enumerate(axes.flat):
    if i < 10:
        im = ax.imshow(x_test[i,:,:,1], vmin=umin0, vmax=umax0)
    elif i < 20:
        im = ax.imshow(y_test[i-10,:,:,1], vmin=umin1, vmax=umax1)
    else:
        im = ax.imshow(decoded_imgs[i-20,:,:,1], vmin=umin1, vmax=umax1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig5.subplots_adjust(right=0.95)
cbar_ax = fig5.add_axes([0.97, 0.2, 0.02, 0.6]) # [left bot width height]
fig5.colorbar(im, cax=cbar_ax)

fig5.savefig(os.path.join(res_path, 
                        "unetsm_{}ep_{}_ns{}_ubw.png".format(n_ep, datasize, ns)),
                        bbox_inches='tight')

#---------------

qmin2 = np.amin(np.concatenate((y_test[:10,:,:,0])))
qmax2 = np.amax(np.concatenate((y_test[:10,:,:,0])))

umin2 = np.amin(np.concatenate((y_test[:10,:,:,1])))
umax2 = np.amax(np.concatenate((y_test[:10,:,:,1])))

print('Starting Q plot 2, only B...', flush=True)

n = 10
fig6, axes = plt.subplots(nrows=3, ncols=10, figsize=(25,8))
for i, ax in enumerate(axes.flat):
    if i < 10:
        im = ax.imshow(y_test[i,:,:,0], vmin=qmin2, vmax=qmax2)
    elif i < 20:
        im = ax.imshow(decoded_imgs[i-10,:,:,0], vmin=qmin2, vmax=qmax2)
    else:
        im = ax.imshow(residuals[i-20,:,:,0], vmin=qmin2, vmax=qmax2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig6.subplots_adjust(right=0.95)
cbar_ax = fig6.add_axes([0.97, 0.2, 0.02, 0.6]) 
fig6.colorbar(im, cax=cbar_ax)

fig6.savefig(os.path.join(res_path, 
                        "unetsm_{}ep_{}_ns{}_qbw_onlyb.png".format(n_ep, datasize, ns)),
                        bbox_inches='tight')

print('Starting U plot 2, only B...', flush=True)
n = 10
fig7, axes = plt.subplots(nrows=3, ncols=10, figsize=(25,8))
for i, ax in enumerate(axes.flat):
    if i < 10:
        im = ax.imshow(y_test[i,:,:,1], vmin=umin2, vmax=umax2)
    elif i < 20:
        im = ax.imshow(decoded_imgs[i-10,:,:,1], vmin=umin2, vmax=umax2)
    else:
        im = ax.imshow(residuals[i-20,:,:,1], vmin=umin2, vmax=umax2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig7.subplots_adjust(right=0.95)
cbar_ax = fig7.add_axes([0.97, 0.2, 0.02, 0.6])
fig7.colorbar(im, cax=cbar_ax)
fig7.savefig(os.path.join(res_path, 
                        "unetsm_{}ep_{}_ns{}_ubw_onlyb.png".format(n_ep, datasize, ns)),
                        bbox_inches='tight')

print('Plots saved, execution complete!', flush=True)



