
#############
# EDIT HERE #
#

path_to_data      = "training_data_v1/"
path_to_real_data = "test_data/"

#model_filename = "classification_PLAsTiCC_v2.1_Ker_Conv.sav"

#
#############

#######################
# Importing Libraries #
#######################
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets, utils, metrics
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import glob
import sys

import warnings
warnings.simplefilter('ignore')



#########################
# Loading Training data #
#########################


files = sorted(glob.glob(path_to_data + "/*.npy"))

#restore:
n_files = len(files)
#debug:n_files = 1000


# > Loading series / labels:
# <indexed as ID array>
    
IDs   = []
# ID number (without "id-")
bands = []
# band name (without "band-") [0 .. 5]

series = []
labels = []

Duration_MJDs = []
# array of duration of light curves

for i in range(n_files):

    
    tmp = np.load(files[i])

    IDs.append(files[i].split("/")[-1].split("_")[0].strip("id-"))
    bands.append(int(files[i].split("/")[-1].split("_")[1].strip("band-")))

    band = int(bands[-1])
    # current band index [0 .. 6]
    
    series.append(tmp)
    labels.append(files[i].split("/")[-1].split("-")[-1].strip(".npy"))

    
    # > Loading single data files:
    #
    # Data format:
    #
    # Modified Juilian Date, Flux, err_Flux, Detected (flag)

    MJD = series[-1][:,0]
    # Modified Julian Date
    
    #F = series[-1][:,1]
    # flux
        
    #F_err = series[-1][:,2]
    # flux error
    
    
    Duration_MJDs.append(MJD[-1] - MJD[0])


IDs_unique = np.unique(IDs)
# list of target IDs, without repetitions

labels_unique = ['' for x in range(len(IDs_unique))]
# list of target labels, without repetitions <indexed as IDs_unique>

print('Number of individual targets: ' + str(len(IDs_unique)))



###############################
# Converting series to images #
###############################

# The conversion happens in two steps:
#
# 1) Splitting a time series over a number of pixels (n_pixels)
#
#    The pixel number is chosen arbitrarily, and corresponds to the maximum time duration
#    (i.e. to the duration of the longest series).
#    Hence, a pixel corresponds to a given time interval.
#    The number of pixels (= time) covered by a series is calculated accordingly to its duration.
#    The series are block-averaged (binned) within each time interval (pixel); empty pixels are attributed a 0.
#
# 2) Converting data into a matrix of flux "ratios"
#
#    The ratios of the fluxes of each pixel with respect to the fluxes of all the other pixels
#    are organized in a n_pixels x n_pixels matrix, of the type:
#
#   [1             , (F_1 / F_2)  , ..       , ..       , (F_1 / F_N)  ]
#   [(F_2 / F_1)   , 1            , ..       , ..       , (F_2 / F_N)  ]
#   [..            , ..           , 1        , ..       , ..           ]
#   [(F_N-1 / F_1) , ..           , ..       , 1        , (F_N-1 / F_N)]
#   [(F_N / F_1)   , ..           , ..       , ..       , 1            ]

n_pixels = 32
n_bands  = 6

Duration_MAX = np.max(Duration_MJDs)
# longest duration of time series [MDJ]

deltaT_per_pixel = Duration_MAX / n_pixels
# time interval within a pixel of the new, resampled images [MJD/pixel]
# NOTE: this is defined with respect to the largest duration
# NOTE: final images will have 64 x 64 pixels

data = np.zeros( (len(IDs_unique) , n_pixels , n_pixels , n_bands ) )
# data array < n_targets , image(x,y), bands >
# NOTE: The first dimension corresponds to the number of targets <indexed as IDs_unique>



for i,serie in enumerate(series):

    band = int(bands[i])
    # current band index [0 .. 6]

    pixels_image = int(np.ceil( Duration_MJDs[i] / deltaT_per_pixel ))
    # number of pixels that the current image will cover in the resampled image

    MJD = series[i][:,0]
    # Modified Julian Date
    
    F = series[i][:,1]
    # flux
    
    
    F_err = series[i][:,2]
    # flux error
    
    MJD   = np.array(MJD)
    F     = np.array(F)
    F_err = np.array(F_err)
    
    F_positive = F + abs(np.min(F))
    # shifting values up to have only positive arrays
    
    F_binned = np.zeros((n_pixels))
    # binned flux series
    
    
    # > Binning series - for band b:
      
    for p in range(pixels_image):
    # p = pixel index 

        (indexes_bin,) = np.where( (p*deltaT_per_pixel <= (MJD - MJD[0])) & ((MJD - MJD[0]) < (p+1)*deltaT_per_pixel) )
        # indexes of data points (of series i) within current pixel (bin)
        
        F_binned[p] = float(np.mean( F_positive[indexes_bin] ))
        # flux within current pixel
        
        np.nan_to_num(F_binned,0)
        # (in case no data points were binned, the average is 'nan')

        
    # > Converting series to image - for band b:
        
    data_i = np.zeros((n_pixels,n_pixels,n_bands))
    # flux ratio matrix for source i
    # NOTE: organizing data in a 2D matrix with shape [ len(array) ,  len(array) , channels(= n_bands) ]
 

    for j in range(n_pixels):
    # j = row index
      
        data_i[j,:,band] = ( F_binned[j] / F_binned )

        
    # > Mirroring matrix across the diagonal:
    
    data_i_band = data_i[:,:,band]
    indexes_lower = np.tril_indices(data_i_band.shape[0], -1)
    # indexes of lower part of matrix
    data_i_band[indexes_lower] = data_i_band.T[indexes_lower]
    # mirroring


    # > Removing 'inf' and 'nan':
    data_i_band[np.isinf(data_i_band)] = 0
    np.nan_to_num(data_i_band,0)
    
    
    # > Normalizing image:
    # NOTE: At this point, value are only negative or 0
    data_i_band_norm = (data_i_band + abs(np.min(data_i_band))) / np.max(data_i_band + abs(np.min(data_i_band)))


    # > Organizing data in proper structure:
    
    (t,) = np.where(IDs_unique == IDs[i])
    t = t[0]
    # t = index of target in array of unique IDs
    # NOTE: manipulation is because np.where returns a 1-value array instead of a variable

    labels_unique[t] = labels[i]
    # NOTE: This is over-written 6 times (once per band)
    #       But it doesn't matter since all bands share the same classification 

    band = int(bands[i])
    # NOTE: "bands" is indexed as "IDs"

    data[t,:,:,band] = data_i_band_norm
    # loading image in corresponding unique (object ID, band) slot  
  
print('Shape of data: ' + str(data.shape))


#################################
# Creating samples for training #
#################################

# Shuffle the samples
shuffled_indexes = np.arange(len(IDs_unique))
np.random.shuffle(shuffled_indexes)

# To reduce the sample size (for testing purposes):
# remove: shuffled_indexes = shuffled_indexes[0:1000]
# remove: n_samples = len(shuffled_indexes)

data = data[shuffled_indexes]
labels_unique = list(np.array(labels_unique)[shuffled_indexes])

n_samples = len(IDs_unique)

# Splitting in training, validation, and test samples:
data_train = data[:8 * n_samples // 10] # i.e. 80% training
labels_train = labels_unique[:8 * n_samples // 10]

data_valid = data[8 * n_samples // 10:9 * n_samples // 10] # i.e. 10% validation (80->90%)
labels_valid = labels_unique[8 * n_samples // 10:9 * n_samples // 10]

data_test = data[9 * n_samples // 10:] # i.e. 10% testing (90->100%)
labels_test = labels_unique[9 * n_samples // 10:]

#n_train_spiral = len([x for x in labels_train if x == 'spiral'])
#n_train_ellipt = len([x for x in labels_train if x == 'ellipt'])

#n_valid_spiral = len([x for x in labels_valid if x == 'spiral'])
#n_valid_ellipt = len([x for x in labels_valid if x == 'ellipt'])

#n_test_spiral = len([x for x in labels_test if x == 'spiral'])
#n_test_ellipt = len([x for x in labels_test if x == 'ellipt'])

print("Sample Summary")
print("________________________")
print("Total images     | %5s" % len(data))
print("-----------------|------")
print(" '-> Training    | %5s" % len(data_train))
#print("      '-> spiral | %5s (%.1f%%)" % (n_train_spiral , (n_train_spiral/len(data_train)*100.)))
#print("      '-> ellipt | %5s (%.1f%%)" % (n_train_ellipt , (n_train_ellipt/len(data_train)*100.)))
print("-----------------|------")
print(" '-> Validation  | %5s" % len(data_valid))
#print("      '-> spiral | %5s (%.1f%%)" % (n_valid_spiral , (n_valid_spiral/len(data_valid)*100.)))
#print("      '-> ellipt | %5s (%.1f%%)" % (n_valid_ellipt , (n_valid_ellipt/len(data_valid)*100.)))
print("-----------------|------")
print(" '-> Test        | %5s" % len(data_test))
#print("      '-> spiral | %5s (%.1f%%)" % (n_test_spiral , (n_test_spiral/len(data_test)*100.)))
#print("      '-> ellipt | %5s (%.1f%%)" % (n_test_ellipt , (n_test_ellipt/len(data_test)*100.)))

print('')
print('Compare these values with the accuracy of each classifier')
print('If accuracies are similar to the demographics, the classifier is only mirroring the data')

############################
# Building Keras CNN model #
############################

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.layers import Dropout

from tensorflow.contrib.layers import maxout

from keras.utils import np_utils
from keras import regularizers
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

n_bands = data_train.shape[-1]

# >> One hot encoding the class values to tranform the vector of class integers into a binary matrix:
int_enc = LabelEncoder()
labels_train_int = int_enc.fit_transform(labels_train)
labels_valid_int = int_enc.fit_transform(labels_valid)
labels_test_int  = int_enc.fit_transform(labels_test)

labels_train_int = np.expand_dims(labels_train_int, axis=1)
labels_valid_int = np.expand_dims(labels_valid_int, axis=1)
labels_test_int  = np.expand_dims(labels_test_int, axis=1)

# Replicating the classification for all the bands:
#labels_train_int = np.repeat(labels_train_int[:,np.newaxis], n_bands, 1)
#labels_valid_int = np.repeat(labels_valid_int[:,np.newaxis], n_bands, 1)
#labels_test_int  = np.repeat(labels_test_int[:,np.newaxis], n_bands, 1)

oh_enc = OneHotEncoder(sparse=False)
labels_train_ohe = oh_enc.fit_transform(labels_train_int)
labels_valid_ohe = oh_enc.fit_transform(labels_valid_int)
labels_train_ohe = oh_enc.fit_transform(labels_test_int)

# uniques, labels_valid = np.unique(labels_valid, return_inverse=True)
labels_train_cat = np_utils.to_categorical(labels_train_int)
labels_valid_cat = np_utils.to_categorical(labels_valid_int)
labels_test_cat  = np_utils.to_categorical(labels_test_int)

#n_classes = labels_valid_ohe.shape[1]
n_classes = 14
# must hard-code this one

print("Labels formats for convolutional layers:")

print("Train      int label format (?, ?, n_samples, n_channels)         | ", labels_train_int.shape)
print("Validation int label format (?, ?, n_samples, n_channels)         | ", labels_valid_int.shape)
print("Test       int label format (?, ?, n_samples, n_channels)         | ", labels_test_int.shape)

n_pixels = data_train.shape[1] 

# Formatting data for convolutional layer:
n_train_targets = data_train.shape[0]
n_valid_targets = data_valid.shape[0]
n_test_targets  = data_test.shape[0]

#labels_train_int_4D = np.expand_dims(labels_train_int   , axis=0)
#labels_train_int_4D = np.expand_dims(labels_train_int_4D, axis=0)
#labels_valid_int_4D = np.expand_dims(labels_valid_int   , axis=0)
#labels_valid_int_4D = np.expand_dims(labels_valid_int_4D, axis=0)
#labels_test_int_4D  = np.expand_dims(labels_test_int    , axis=0)
#labels_test_int_4D  = np.expand_dims(labels_test_int_4D , axis=0)


print("Data formats for convolutional layers:")

print("Train      4D data format (n_samples,size_x, size_y, n_channels) | ", data_train.shape)
print("Validation 4D data format (n_samples,size_x, size_y, n_channels) | ", data_valid.shape)
print("Test       4D data format (n_samples,size_x, size_y, n_channels) | ", data_test.shape)

#print("Train      4D label format (?, ?, n_samples, n_channels)         | ", labels_train_int_4D.shape)
#print("Validation 4D label format (?, ?, n_samples, n_channels)         | ", labels_valid_int_4D.shape)
#print("Test       4D label format (?, ?, n_samples, n_channels)         | ", labels_test_int_4D.shape)

# Trying with less bands:

n_bands = 6

data_train = data_train[:,:,:,:n_bands]
data_valid = data_valid[:,:,:,:n_bands]
data_test  = data_test [:,:,:,:n_bands]

if (n_bands == 1):
    data_train = np.expand_dims(data_train, axis=3)
    data_valid = np.expand_dims(data_valid, axis=3)
    data_test  = np.expand_dims(data_test, axis=3)


print("Data formats for convolutional layers - 1 band:")

print("Train      4D data format (n_samples,size_x, size_y, n_channels) | ", data_train.shape)
print("Validation 4D data format (n_samples,size_x, size_y, n_channels) | ", data_valid.shape)
print("Test       4D data format (n_samples,size_x, size_y, n_channels) | ", data_test.shape)


model_Conv = keras.Sequential([
                          keras.layers.Conv2D(8, kernel_size=(2,2), strides=(1,1), padding='valid',
                                              activation=tf.nn.relu, input_shape=(n_pixels,n_pixels,n_bands),
                                              data_format="channels_last"),
                          keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                          keras.layers.Conv2D(12, kernel_size=(2,2), strides=(1,1), padding='same',
                                              activation=tf.nn.relu),
                          keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                          keras.layers.Conv2D(16, kernel_size=(2,2), strides=(1,1), padding='same',
                                              activation=tf.nn.relu),
                          keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                          keras.layers.Flatten(),
                          keras.layers.Dense(64, activation=tf.nn.sigmoid),
                          keras.layers.Dropout(0.3),
                          keras.layers.Dense(n_classes, activation=tf.nn.sigmoid)
                          ])


model_Conv.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model_Conv.summary()

history_Conv = model_Conv.fit(data_train, labels_train_int, validation_data=(data_valid, labels_valid_int),
                    epochs=60, batch_size=128, verbose=2)


print('Number of different classes in train sample: ' + str(len(np.unique(labels_train))))
print('Number of different classes in valid sample: ' + str(len(np.unique(labels_valid))))
print('Number of different classes in test  sample: ' + str(len(np.unique(labels_test))))
print('')
print('WARNING: A display error can occur if the represented classes differ in the train/valid/test sets.')
print('         If so, increase the samples.')


# > Comparison with predictions:
labels_pred_float_Conv = model_Conv.predict(data_test)

labels_pred_Conv = int_enc.inverse_transform(labels_pred_float_Conv.argmax(1))
# reversing one hot encoding


print("Classification report for %s:\n%s\n"
      % (model_Conv, metrics.classification_report(labels_test, labels_pred_Conv)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_test, labels_pred_Conv))


#####################
# Running the model #
##################### 
#(for the real data set)


print()
print('========================')
print('   Applying the model')
print('========================')

files = sorted(glob.glob(path_to_real_data + "/*.npy"))
n_files = len(files)
IDs   = []
bands = []
series = []
labels = []
Duration_MJDs = []
for i in range(n_files):
    tmp = np.load(files[i])

#    print(files[i])
    IDs.append(files[i].split("/")[-1].split("_")[0].strip("id-"))
    bands.append(int(files[i].split('/')[-1].split('_')[1].split('-')[1].strip('.npy')))
    band = int(bands[-1])
    # current band index [0 .. 6]
    
    series.append(tmp)
    labels.append(files[i].split("/")[-1].split("-")[-1].strip(".npy"))

    
    # > Loading single data files:
    #
    # Data format:
    #
    # Modified Juilian Date, Flux, err_Flux, Detected (flag)

    MJD = series[-1][:,0]
    # Modified Julian Date
    
    #F = series[-1][:,1]
    # flux
        
    #F_err = series[-1][:,2]
    # flux error
    
    
    Duration_MJDs.append(MJD[-1] - MJD[0])


IDs_unique = np.unique(IDs)
# list of target IDs, without repetitions

labels_unique = ['' for x in range(len(IDs_unique))]
# list of target labels, without repetitions <indexed as IDs_unique>

print('Number of individual targets: ' + str(len(IDs_unique)))

n_pixels = 32
n_bands  = 6

Duration_MAX = np.max(Duration_MJDs)
# longest duration of time series [MDJ]

deltaT_per_pixel = Duration_MAX / n_pixels
# time interval within a pixel of the new, resampled images [MJD/pixel]
# NOTE: this is defined with respect to the largest duration
# NOTE: final images will have 64 x 64 pixels

data = np.zeros( (len(IDs_unique) , n_pixels , n_pixels , n_bands ) )
# data array < n_targets , image(x,y), bands >
# NOTE: The first dimension corresponds to the number of targets <indexed as IDs_unique>

for i,serie in enumerate(series):

    band = int(bands[i])
    # current band index [0 .. 6]

    pixels_image = int(np.ceil( Duration_MJDs[i] / deltaT_per_pixel ))
    # number of pixels that the current image will cover in the resampled image

    MJD = series[i][:,0]
    # Modified Julian Date
    
    F = series[i][:,1]
    # flux
    
    
    F_err = series[i][:,2]
    # flux error
    
    MJD   = np.array(MJD)
    F     = np.array(F)
    F_err = np.array(F_err)
    
    F_positive = F + abs(np.min(F))
    # shifting values up to have only positive arrays
    
    F_binned = np.zeros((n_pixels))
    # binned flux series
    
    
    # > Binning series - for band b:
      
    for p in range(pixels_image):
    # p = pixel index 

        (indexes_bin,) = np.where( (p*deltaT_per_pixel <= (MJD - MJD[0])) & ((MJD - MJD[0]) < (p+1)*deltaT_per_pixel) )
        # indexes of data points (of series i) within current pixel (bin)
        
        F_binned[p] = float(np.mean( F_positive[indexes_bin] ))
        # flux within current pixel
        
        np.nan_to_num(F_binned,0)
        # (in case no data points were binned, the average is 'nan')

        
    # > Converting series to image - for band b:
        
    data_i = np.zeros((n_pixels,n_pixels,n_bands))
    # flux ratio matrix for source i
    # NOTE: organizing data in a 2D matrix with shape [ len(array) ,  len(array) , channels(= n_bands) ]
 

    for j in range(n_pixels):
    # j = row index
      
        data_i[j,:,band] = ( F_binned[j] / F_binned )

        
    # > Mirroring matrix across the diagonal:
    
    data_i_band = data_i[:,:,band]
    indexes_lower = np.tril_indices(data_i_band.shape[0], -1)
    # indexes of lower part of matrix
    data_i_band[indexes_lower] = data_i_band.T[indexes_lower]
    # mirroring


    # > Removing 'inf' and 'nan':
    data_i_band[np.isinf(data_i_band)] = 0
    np.nan_to_num(data_i_band,0)
    
    
    # > Normalizing image:
    # NOTE: At this point, value are only negative or 0
    data_i_band_norm = (data_i_band + abs(np.min(data_i_band))) / np.max(data_i_band + abs(np.min(data_i_band)))


    # > Organizing data in proper structure:
    
    (t,) = np.where(IDs_unique == IDs[i])
    t = t[0]
    # t = index of target in array of unique IDs
    # NOTE: manipulation is because np.where returns a 1-value array instead of a variable

    labels_unique[t] = labels[i]
    # NOTE: This is over-written 6 times (once per band)
    #       But it doesn't matter since all bands share the same classification 

    band = int(bands[i])
    # NOTE: "bands" is indexed as "IDs"

    data[t,:,:,band] = data_i_band_norm
    # loading image in corresponding unique (object ID, band) slot  
    

predictions = model_Conv.predict(data)
ids = IDs_unique.astype(int)

# reshaping arrays to combine tables
ids_mod = ids.reshape(len(ids),1)
class99 = np.zeros(len(ids)).reshape(len(ids),1)

# have to re-arrange the band order (actually move band 6 at position 5 to position 1)
# to follow the format of submission

final_array = np.concatenate((ids_mod, predictions[:, [5,0,1,2,3,4,6,7,8,9,10,11,12,13]], class99), axis=1)

print(ids)
print(predictions)
print()
print(final_array)

np.savetxt("foo.csv", final_array, fmt='%d'+(',%1.4f'*15))

