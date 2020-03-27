import numpy as np
import os, keras, pickle, h5py, re
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import welch
from helper_functions import calc_psds, remove_zero_lfps, set_up_model, read_ids_parameters

DATA_DIR = os.path.join('../simulation_code/lfp_simulations_small/')
SAVE_DIR = os.path.join('./save/')

## Load LFPs
ids, labels = read_ids_parameters(os.path.join(DATA_DIR, 'id_parameters.txt'))
lfps = np.zeros((len(ids), 6, 2851), dtype=np.float32)
for j, i in enumerate(ids):
    with h5py.File(os.path.join(DATA_DIR, 'nest_output', i, 'LFP_firing_rate.h5')) as f:
        lfp = f['data'][:,150:]
    lfps[j] = lfp

lfps = np.array(lfps)
labels = np.array(labels)

## Get PSDs
fs, psds = calc_psds(lfps)
del lfps

## Remove cases of simulations where no neurons spiked
psds, labels = remove_zero_lfps(psds, labels)


test_psds = psds[:10000]
test_labels = labels[:10000]
training_psds = psds[10000:]
training_labels = labels[10000:]

## Hyperparams
batch_size = 100
epochs = 400
lr = 1e-3

x_train = training_psds
y_train = training_labels.copy()
x_test = test_psds
y_test = test_labels.copy()
x_train = np.swapaxes(x_train, 1,2)
x_test = np.swapaxes(x_test, 1,2)

y_test -= np.array([[0.8, 3.5, 0.05]])
y_test /= np.array([[3.2, 4.5, 0.35]])
y_train -= np.array([[0.8, 3.5, 0.05]])
y_train /= np.array([[3.2, 4.5, 0.35]])

## Set up and train model
full_model = set_up_model(x_train, lr, n_dense=128, output=3)

checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, 'model_small_parameterspace.h5'),
                                             monitor='val_loss',
                                             mode='min',
                                             save_best_only=True)

full_history = full_model.fit(x_train,
                              y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=2,
                              validation_data=(x_test, y_test),
                              callbacks=[checkpoint])

with open(os.path.join(SAVE_DIR, 'history_small_parameterspace.pkl'), 'wb') as f:
    pickle.dump(full_history.history, f)
