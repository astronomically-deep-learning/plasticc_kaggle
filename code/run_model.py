import sys
import numpy as np

import time

import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


import warnings
warnings.simplefilter('ignore')

def load_keras_model():

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model_Conv = load_model('../data/model_CNN.h5')

    return model_Conv

def load_data(filename):

    dtype = [('object_id','i8'), ('mjd','f8'), ('passband','i8'), ('flux','f8'), ('flux_err','f8'), ('detected','i8')]

    data = np.genfromtxt(filename, dtype=dtype, delimiter=',')

    return data


def run_band(series, deltaT_per_pixel):

    MJD = series['mjd']
    # Modified Julian Date

    F = series['flux']
    # flux

    F_err = series['flux_err']
    # flux error


    Duration_MJD = np.max(MJD) - np.min(MJD)
    pixels_image = int(np.ceil( Duration_MJD / deltaT_per_pixel ))
    # number of pixels that the current image will cover in the resampled image

    # So we don't go over the maximum number of pixels
    pixels_image = np.min([pixels_image, n_pixels])


    #
    # MJD   = np.array(MJD)
    # F     = np.array(F)
    # F_err = np.array(F_err)

    F_positive = F + abs(np.min(F))
    # shifting values up to have only positive arrays


    F_binned = np.zeros((n_pixels))
    # binned flux series


    # > Binning series - for band b:

    for p in range(n_pixels):
    # p = pixel index

        indexes_bin = np.where( (p*deltaT_per_pixel <= (MJD - MJD[0])) & ((MJD - MJD[0]) < (p+1)*deltaT_per_pixel) )[0]
        # indexes of data points (of series i) within current pixel (bin)

        F_binned[p] = float(np.mean( F_positive[indexes_bin] ))
        # flux within current pixel

        np.nan_to_num(F_binned,0)
        # (in case no data points were binned, the average is 'nan')


    # > Converting series to image - for band b:
    data_i = np.zeros((n_pixels,n_pixels))
    # flux ratio matrix for source i
    # NOTE: organizing data in a 2D matrix with shape [ len(array) ,  len(array) , channels(= n_bands) ]


    # Row index
    for j in range(n_pixels):  data_i[j,:] = ( F_binned[j] / F_binned )


    # > Mirroring matrix across the diagonal:
    indexes_lower = np.tril_indices(data_i.shape[0], -1)
    # indexes of lower part of matrix
    data_i[indexes_lower] = data_i.T[indexes_lower]
    # mirroring


    # > Removing 'inf' and 'nan':
    data_i[np.isinf(data_i)] = 0
    np.nan_to_num(data_i,0)


    # > Normalizing image:
    # NOTE: At this point, value are only negative or 0
    data_i_norm = (data_i + abs(np.min(data_i))) / np.max(data_i + abs(np.min(data_i)))



    return data_i_norm
    # loading image in corresponding unique (object ID, band) slot


# # Load pickled model
model_Conv = load_keras_model()


n_pixels = 32
n_bands  = 6

Duration_MAX = 1094.00239999999
# Duration_MAX = np.max(Duration_MJDs)
# longest duration of time series [MDJ]


def run_model(filebase):

    start_time = time.time()

    path_to_real_data = "../data/"

    filepath = path_to_real_data + filebase + ".npy"
    data_in = np.load(filepath)


#     # Load data from .csv files
#     data_in = load_data(filepath)

    # List of unique objects
    obj_array = np.unique(data_in['object_id'])
    N_objs = len(obj_array)

    print('Number of individual targets: ' + str(N_objs))


    deltaT_per_pixel = Duration_MAX / n_pixels
    # time interval within a pixel of the new, resampled images [MJD/pixel]
    # NOTE: this is defined with respect to the largest duration
    # NOTE: final images will have 64 x 64 pixels

    data_2D = np.zeros( (N_objs, n_pixels, n_pixels, n_bands) )
    labels_unique = np.zeros(N_objs)
    # data array < n_targets , image(x,y), bands >
    # NOTE: The first dimension corresponds to the number of targets <indexed as IDs_unique>


    # Iterate through objects and bands - convert luminosities to 2D data
    bands = np.arange(6)
    for i, obj_ID in enumerate(obj_array):

        if i%100 == 0: print(i, time.time() - start_time)

        idx = np.where(data_in[data_in['object_id'] == obj_ID])

        data_obj = data_in[idx]


        for b in range(n_bands):

            if len(data_obj[data_obj['passband'] == b]) == 0: continue

            data_2D[i,:,:,b] = run_band(data_obj[data_obj['passband'] == b], deltaT_per_pixel)

        labels_unique[i] = obj_ID


    # Run predictions
    predictions = model_Conv.predict(data_2D)

    # reshaping arrays to combine tables
    ids_mod = obj_array.reshape(N_objs,1)
    class99 = np.zeros((N_objs,1))

    # have to re-arrange the band order (actually move band 6 at position 5 to position 1)
    # to follow the format of submission



    final_array = np.concatenate((ids_mod, predictions[:, [5,0,1,2,3,4,6,7,8,9,10,11,12,13]], class99), axis=1)

    np.savetxt(filebase + "_classified.csv", final_array, fmt='%d'+(',%1.4f'*15))


run_model(sys.argv[1])
