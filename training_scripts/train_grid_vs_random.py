import numpy as np
import os, keras, pickle, h5py, re
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import welch
from helper_functions import calc_psds, remove_zero_lfps, set_up_model, read_ids_parameters

DATA_DIR_GRID = os.path.join('../simulation_code/lfp_simulations_grid/')
DATA_DIR_RANDOM = os.path.join('../simulation_code/lfp_simulations_grid/')
DATA_DIR_TEST = os.path.join('../simulation_code/lfp_simulations_large/')

SAVE_DIR = os.path.join('./save/')

### GRID SAMPLED DATA
# Load LFPs
print('loading grid sampled data')
ids, grid_labels = read_ids_parameters(os.path.join(DATA_DIR_GRID, 'id_parameters.txt'))
lfps = np.zeros((len(ids), 6, 2851), dtype=np.float32)
for j, i in enumerate(ids):
    with h5py.File(os.path.join(DATA_DIR_GRID, 'nest_output', i, 'LFP_firing_rate.h5')) as f:
        lfp = f['data'][:,150:]
    lfps[j] = lfp

grid_labels = np.array(grid_labels)

## Get PSDs
fs, grid_psds = calc_psds(lfps)
del lfps

## Remove cases of simulations where no neurons spiked
psds, labels = remove_zero_lfps(grid_psds, grid_labels)


### RANDOMLY SAMPLED DATA
# Load LFPs
print('loading randomly sampled data')
ids, random_labels = read_ids_parameters(os.path.join(DATA_DIR_RANDOM, 'id_parameters.txt'))
lfps = np.zeros((len(ids), 6, 2851), dtype=np.float32)
for j, i in enumerate(ids):
    print('%d'%j, end='\r')
    with h5py.File(os.path.join(DATA_DIR_RANDOM, 'nest_output', i, 'LFP_firing_rate.h5')) as f:
        lfp = f['data'][:,150:]
    lfps[j] = lfp

print('converting')
random_labels = np.array(random_labels)

print('calc psd')
## Get PSDs
fs, random_psds = calc_psds(lfps)

print('remove zero')
## Remove cases of simulations where no neurons spiked
random_psds, random_labels = remove_zero_lfps(grid_psds, grid_labels)


## Load LFPs
print('loading test data')
ids, test_labels = read_ids_parameters(os.path.join(DATA_DIR_TEST, 'id_parameters.txt'))
lfps = np.zeros((10000, 6, 2851), dtype=np.float32)
for j, i in enumerate(ids[:10000]):
    with h5py.File(os.path.join(DATA_DIR_TEST, 'nest_output', i, 'LFP_firing_rate.h5')) as f:
        lfp = f['data'][:,150:]
    lfps[j] = lfp

test_labels = np.array(test_labels)

## Get PSDs
fs, test_psds = calc_psds(lfps)

## Remove cases of simulations where no neurons spiked
psds, labels = remove_zero_lfps(psds, labels)

## Split test data
test_psds = test_psds[:10000]
test_labels = test_labels[:10000]


def set_up_model(x_train, lr, output=1):
    keras.backend.clear_session()
    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(20, kernel_size=(12), strides=(1), activation='relu', use_bias=False)(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv1D(20, kernel_size=(4), strides=(1), activation='relu', use_bias=False)(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv1D(20, kernel_size=(4), strides=(1), activation='relu', use_bias=False)(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    predictions = keras.layers.Dense(output, activation='relu', use_bias=False)(x)
    model = keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['mse', 'mae'])
    print(model.summary())
    return model

batch_size = 100
epochs = 400
lr = 1e-3

x_train = grid_psds
y_train = grid_labels.copy()
x_test = test_psds
y_test = test_labels.copy()
x_train = np.swapaxes(x_train, 1,2)
x_test = np.swapaxes(x_test, 1,2)
y_test -= np.array([[0.8, 3.5, 0.05]])
y_test /= np.array([[3.2, 4.5, 0.35]])
y_train -= np.array([[0.8, 3.5, 0.05]])
y_train /= np.array([[3.2, 4.5, 0.35]])

grid_model = set_up_model(x_train, lr, output=3)
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, 'grid_sampling_model.h5'),
                                             monitor='val_loss',
                                             mode='min',
                                             save_best_only=True)

grid_history = grid_model.fit(x_train,
                              y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=2,
                              validation_data=(x_test, y_test),
                              callbacks=[checkpoint])

with open(os.path.join(SAVE_DIR, 'grid_sampling_history.pkl'), 'wb') as f:
    pickle.dump(grid_history.history, f)

x_train = random_psds
y_train = random_labels.copy()
x_test = test_psds
y_test = test_labels.copy()
x_train = np.swapaxes(x_train, 1,2)
x_test = np.swapaxes(x_test, 1,2)
y_test -= np.array([[0.8, 3.5, 0.05]])
y_test /= np.array([[3.2, 4.5, 0.35]])
y_train -= np.array([[0.8, 3.5, 0.05]])
y_train /= np.array([[3.2, 4.5, 0.35]])

random_model = set_up_model(x_train, lr, output=3)

checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, 'random_sampling_model.h5'),
                                             monitor='val_loss',
                                             mode='min',
                                             save_best_only=True)

random_history = random_model.fit(x_train, y_train,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_data=(x_test, y_test),
                                  callbacks=[checkpoint])

with open(os.path.join(SAVE_DIR, 'random_sampling_history.pkl'), 'wb') as f:
    pickle.dump(random_history.history, f)
