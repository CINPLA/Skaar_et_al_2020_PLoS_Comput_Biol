import numpy as np
import os, keras, pickle, h5py, re
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import welch
from helper_functions import calc_psds, remove_zero_lfps, set_up_model, read_ids_parameters

DATA_DIR = os.path.join('../simulation_code/lfp_simulations_large/')
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

## Split test data
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


## Set up and train full model
full_model = set_up_model(x_train, lr, n_dense=128, output=3)
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, 'model_full_parameterspace.h5'),
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

with open(os.path.join(SAVE_DIR, 'full_history_full_parameterspace.pkl'), 'wb') as f:
    pickle.dump(full_history.history, f)

## Set up and train single model for eta
eta_model = set_up_model(x_train, lr, n_dense=128, output=1)
eta_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, 'eta_model_full_parameterspace.h5'),
                                                 monitor='val_loss',
                                                 mode='min',
                                                 save_best_only=True)
eta_history = eta_model.fit(x_train,
                            y_train[:,0],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_data=(x_test, y_test[:,0]),
                            callbacks=[eta_checkpoint])
with open(os.path.join(SAVE_DIR, 'eta_history_full_parameterspace.pkl'), 'wb') as f:
    pickle.dump(eta_history.history, f)

## Set up and train single model for g
g_model = set_up_model(x_train, lr, n_dense=128, output=1)
g_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, 'g_model_full_parameterspace.h5'),
                                               monitor='val_loss',
                                               mode='min',
                                               save_best_only=True)

g_history = g_model.fit(x_train, y_train[:,1],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_test, y_test[:,1]),
                        callbacks=[g_checkpoint])
with open(os.path.join(SAVE_DIR, 'g_history_full_parameterspace.pkl'), 'wb') as f:
    pickle.dump(g_history.history, f)

## Set up and train single model for J
j_model = set_up_model(x_train, lr, n_dense=128, output=1)
j_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, 'j_model_full_parameterspace.h5'),
                                               monitor='val_loss',
                                               mode='min',
                                               save_best_only=True)

j_history = j_model.fit(x_train, y_train[:,2],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_test, y_test[:,2]),
                        callbacks=[j_checkpoint])

with open(os.path.join(SAVE_DIR, 'j_history_full_parameterspace.pkl'), 'wb') as f:
    pickle.dump(j_history.history, f)
