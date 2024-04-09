#!/home/puranjay/mlenv/bin/python
print('py script started', flush=True)

#--------
# Imports
#--------

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
import scipy as sc

import sys
import os

import h5py
from tqdm import tqdm

data_path = "/home/puranjay/projects/def-hinshaw/puranjay/data"

print('imports complete', flush=True)


#-------------
# Data Loading
#-------------

l_camb, tt_camb, ee_camb, bb_camb, te_camb = np.loadtxt(os.path.join(data_path, "camb_47773334_totcls.dat"), dtype='float', unpack=True)
print("camb data loaded", flush=True) 


#------------------------------
# C_ell List Creation 
# Obtain 256x256 cartesian maps 
# From this data
#------------------------------

ns_orig = 2048
npix = hp.nside2npix(ns_orig)
ps_fac = (l_camb*(l_camb + 1.))/(2.*np.pi)
cl_list_all = np.zeros((4, 2201))
cl_list_all[:, 2:] = np.array((1./ps_fac)*[tt_camb, ee_camb, bb_camb, te_camb])

cl_list_bb = np.zeros(cl_list_all.shape)
cl_list_ee = np.zeros(cl_list_all.shape)
cl_list_eebb = np.zeros(cl_list_all.shape)

cl_list_bb[2] = cl_list_all[2]
cl_list_ee[[1], :] = cl_list_all[[1], :]
cl_list_eebb[[1,2], :] = cl_list_all[[1,2], :]

print("cl lists made", flush=True)

ns = 1024
npix = hp.nside2npix(ns)
indices= np.arange(npix)
resol = hp.nside2resol(ns)
resol_arcmin = hp.nside2resol(ns, arcmin=True)
print("nside is {:d}".format(ns), flush=True)

#----------------
# Data Parameters
#----------------

nmaps = ntrain = 12500 # Total train + test data size
npatches = 1*nmaps     # How many 20 deg x 20deg patches do we take from 1 full sky map realization?

print("number of maps to be generated is {:d}".format(nmaps), flush=True)
print("number of patches to be generated is {:d} (= ntrain)".format(npatches), flush=True)

imgpix = 256 # No. of pixels in resulting 2d images
input_shape = (npatches,imgpix,imgpix)
print('img dim is:{}, and total patch array shape is:{}'.format(imgpix, input_shape), flush=True)

#-----------------
# Dataset Creation
#-----------------

f = h5py.File(os.path.join(data_path, 'traintest_data_1patch20deg_ns1024_12k5.h5'), 'w-')
#w- : create file, fail if exists

dset_xtr = f.create_dataset('train/x_train', shape=(11250,256,256,2), dtype='float32', chunks=True, 
                        compression='gzip', maxshape=(None, 256, 256, 2))
dset_ytr = f.create_dataset('train/y_train', shape=(11250,256,256,2), dtype='float32', chunks=True, 
                        compression='gzip', maxshape=(None, 256, 256, 2))
dset_eetr = f.create_dataset('train/ee_patches', shape=(11250,256,256,2), dtype='float32', chunks=True, 
                        compression='gzip', maxshape=(None, 256, 256, 2))

dset_xtest = f.create_dataset('test/x_test', shape=(1250,256,256,2), dtype='float32', chunks=True, 
                        compression='gzip', maxshape=(None, 256, 256, 2))
dset_ytest = f.create_dataset('test/y_test', shape=(1250,256,256,2), dtype='float32', chunks=True, 
                        compression='gzip', maxshape=(None, 256, 256, 2))
dset_eetest = f.create_dataset('test/ee_patches', shape=(1250,256,256,2), dtype='float32', chunks=True, 
                        compression='gzip', maxshape=(None, 256, 256, 2))

dset_xtr.shape
dset_ytr.shape
dset_eetr.shape
dset_xtest.shape
dset_ytest.shape
dset_eetest.shape
f

#--------------------------
# Longitude/Latitude
# From where square patches
# are taken
#--------------------------

lon1 = np.linspace(-190,80,4)
lon2 = np.linspace(-170,100,4)

lonlist = np.asarray([(x1,x2) for x1,x2 in zip(lon1, lon2)])
latlist = np.tile(np.linspace(-10,10,2), 4).reshape(4,2)

# In case of 4 patches from each full-sky map 
lonra = lonlist[2]
latra = latlist[2]

# In case of 1 patch from each full-sky map 
lonra2 = [10., 30.]
latra2 = [10., 30.]

print('Sanity check your lonra:{},latra:{}'.format(lonra, latra), flush=True)
print('Sanity check your lonra:{},latra:{}'.format(lonra2, latra2), flush=True)

chunk_size = (1250,256,256,2)
ebpatch_chunk = np.zeros(chunk_size).astype('float32')
bpatch_chunk = np.zeros(chunk_size).astype('float32')
epatch_chunk = np.zeros(chunk_size).astype('float32')

n_chunk = chunk_size[0]
n_iter = 10 # total patches generated = n_chunk * n_iter

print('so far so good, starting map+patch generation loop: (ntrain={} times)'.format(ntrain), flush=True)


# --------------------------
# Main Patch Generation Loop
# --------------------------

#Overwrire one single map {tqu} variable to save working memory

#Count variables for E, B, and E+B patches
cnt_eb = 0
cnt_b = 0
cnt_e = 0

print('Progress: 0 out of {:d} patches to be made...'.format(3*npatches), flush=True)

for k in tqdm(range(n_iter)):
# This outer loop will write 10% patches to file at a time

    for i in tqdm(range(n_chunk)):
        # This inner loop will genenrate those chunk of patches to write, 10% at a time
        # larger chunk size for faster performance, one at a time is too many I/O operations

        # Obtain {T, Q, U} maps from C_ell lists 
        [t_e, q_e, u_e] = hp.synfast(cl_list_ee, nside=ns, pol=True, fwhm=resol, new=True)
        [t_b, q_b, u_b] = hp.synfast(cl_list_bb, nside=ns, pol=True, fwhm=resol, new=True)

        # Obtain {T, Q, U} maps containing both E & B-modes
        t_eb = t_e + t_b
        q_eb = q_e + q_b
        u_eb = u_e + u_b
        # Now {t_eb, q_eb, u_eb} contains the complete E + B map. 

        # Take out 4 (or 1) patch(es) from it and throw it away, no need to save the map

        # In case of 1 patch per full-sky realization, remove the lonra/latra loop entirely
        # Currently 1 patch per realization 
        #    for lonra, latra in zip(lonlist, latlist):

        # Generate patches, i.e., 2D projections of our 20deg x 20deg regions of the full sky   

        #First for E+B maps...
        ebpatch_chunk[i,:,:,0] = hp.cartview(q_eb, xsize=imgpix, lonra=lonra, latra=latra, 
                                             fig=0, return_projected_map=True, 
                                             cbar=False, notext=True).astype('float32');
        plt.close(0);
        plt.clf();

        ebpatch_chunk[i,:,:,1] = hp.cartview(u_eb, xsize=imgpix,lonra=lonra, latra=latra, 
                                             fig=1, return_projected_map=True, 
                                             cbar=False, notext=True).astype('float32');
        plt.close(1);
        plt.clf();
        
        cnt_eb += 1
        if cnt_eb%(n_chunk//2) == 0:
            print('{:d} out of {:d} E+B patches made...'.format(cnt_eb, npatches), flush=True)
        
        plt.close(plt.gcf());
        plt.close('all');

        # Then for B maps...
        bpatch_chunk[i,:,:,0] = hp.cartview(q_b, xsize=imgpix, fig=0, lonra=lonra, latra=latra,
                                            return_projected_map=True, cbar=False, notext=True).astype('float32');
        plt.close(0);
        plt.clf();
        bpatch_chunk[i,:,:,1] = hp.cartview(u_b, xsize=imgpix, fig=1, lonra=lonra, latra=latra,
                                            return_projected_map=True, cbar=False, notext=True).astype('float32');
        plt.close(1);
        plt.clf();
        
        cnt_b+=1
        if cnt_b%(n_chunk//2) == 0:
            print('{:d} out of {:d} pure B patches made...'.format(cnt_b, npatches), flush=True)

        plt.close(plt.gcf());
        plt.close('all');
        

        # And finally for E maps...
        epatch_chunk[i,:,:,0] = hp.cartview(q_e, xsize=imgpix, fig=0, lonra=lonra, latra=latra,
                                            return_projected_map=True, cbar=False, notext=True).astype('float32');
        plt.close(0);
        plt.clf();
        epatch_chunk[i,:,:,1] = hp.cartview(u_e, xsize=imgpix, fig=1, lonra=lonra, latra=latra,
                                            return_projected_map=True, cbar=False, notext=True).astype('float32');
        plt.close(0);
        plt.clf();
        cnt_e+=1
        if cnt_e%(n_chunk//2) == 0:
            print('{:d} out of {:d} pure E patches made...'.format(cnt_e, npatches), flush=True)    
        
        plt.close(plt.gcf());
        plt.close('all');


    # Now the patch_chunk np.arrays have generated patches stored in them, save to h5 file 
    # Do in chunks of 10% for speed, and save last chunk as test set,
    # instead of appending to train set
    if k == n_iter-1:
        f['test/x_test'][:n_chunk] = ebpatch_chunk
        f['test/y_test'][:n_chunk] = bpatch_chunk
        f['test/ee_patches'][:n_chunk] = epatch_chunk
    else:
        f['train/x_train'][k*n_chunk:(k+1)*n_chunk] = ebpatch_chunk
        f['train/y_train'][k*n_chunk:(k+1)*n_chunk] = bpatch_chunk
        f['train/ee_patches'][k*n_chunk:(k+1)*n_chunk] = epatch_chunk

f.close()
print('File manually closed', flush=True)
print('Datasets successfully saved!', flush=True)
