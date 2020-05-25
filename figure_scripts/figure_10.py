import numpy as np
import matplotlib.pyplot as plt
import os, h5py, keras
import matplotlib.gridspec as gridspec
from helper_functions import calc_psds, read_ids_parameters, remove_zero_lfps


TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2

RANDOM_MODEL_PATH = os.path.join('../training_scripts/save/model_full_parameterspace.h5')
GRID_MODEL_PATH = os.path.join('../training_scripts/save/grid_model.h5')

DATA_DIR_LARGE = os.path.join('../simulation_code/lfp_simulations_large/')
DATA_DIR_SMALL = os.path.join('../simulation_code/lfp_simulations_small/')
SAVE_DIR = os.path.join('./save/')

#### Load large parameterspace LFPs
ids, labels_large = read_ids_parameters(os.path.join(DATA_DIR_LARGE, 'id_parameters.txt'))

## Test sims
ids = ids[:10000]
labels_large = labels_large[:10000]

lfps = np.zeros((len(ids), 6, 2851), dtype=np.float32)
for j, i in enumerate(ids[:10000]):
    with h5py.File(os.path.join(DATA_DIR_LARGE, 'nest_output', i, 'LFP_firing_rate.h5')) as f:
        lfp = f['data'][:,150:]
    lfps[j] = lfp

lfps = np.array(lfps)
labels_large = np.array(labels_large)

## Get PSDs
fs, psds_large = calc_psds(lfps)
del lfps

## Remove cases of simulations where no neurons spiked
psds_large, labels_large = remove_zero_lfps(psds_large, labels_large)

psds_large = np.swapaxes(psds_large, 1,2)

## Rescale labels
labels_large_rescaled = labels_large - np.array([[0.8, 3.5, 0.05]])
labels_large_rescaled /= np.array([[3.2, 4.5, 0.35]])

## CNN model
def set_up_model(x_train, lr, n_dense=128, output=3):
    keras.backend.clear_session()
    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(20, kernel_size=(12), strides=(1),
                            activation='relu', use_bias=False)(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv1D(20, kernel_size=(4), strides=(1),
                            activation='relu', use_bias=False)(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv1D(20, kernel_size=(4), strides=(1),
                            activation='relu', use_bias=False)(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(n_dense, activation='relu')(x)
    x = keras.layers.Dense(n_dense, activation='relu')(x)
    predictions = keras.layers.Dense(output, activation=None, use_bias=False)(x)
    model = keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['mse', 'mae'])
    return model

## Get errors
full_model = set_up_model(psds_large, 0)
full_model.load_weights(RANDOM_MODEL_PATH)
preds = full_model.predict(psds_large)
random_errors = preds - labels_large_rescaled

grid_model = set_up_model(psds_large, 0)
grid_model.load_weights(GRID_MODEL_PATH)
grid_preds = grid_model.predict(psds_large)
grid_errors = grid_preds - labels_large_rescaled

## Create histograms
bins=np.linspace(-7, 7, 3000)
x = bins[1:] - (bins[-1]-bins[-2])/2

random_hist_eta, _ = np.histogram(random_errors[:,0], bins=bins)
random_hist_eta = random_hist_eta/random_hist_eta.sum()

random_hist_g, _ = np.histogram(random_errors[:,1], bins=bins)
random_hist_g = random_hist_g/random_hist_g.sum()

random_hist_j, _ = np.histogram(random_errors[:,2], bins=bins)
random_hist_j = random_hist_j/random_hist_j.sum()

grid_hist_eta, _ = np.histogram(grid_errors[:,0], bins=bins)
grid_hist_eta = grid_hist_eta/grid_hist_eta.sum()

grid_hist_g, _ = np.histogram(grid_errors[:,1], bins=bins)
grid_hist_g = grid_hist_g/grid_hist_g.sum()

grid_hist_j, _ = np.histogram(grid_errors[:,2], bins=bins)
grid_hist_j = grid_hist_j/grid_hist_j.sum()

## Create figure
lw=2.0
fig = plt.figure()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE, 1.8])
gs = gridspec.GridSpec(ncols=3, nrows=1, wspace=0.3, bottom=0.3 )
ax = [plt.subplot(gs[i]) for i in range(3)]

ax[0].step(x, random_hist_eta, color='royalblue', lw=lw)
ax[0].step(x, grid_hist_eta, color='orange', lw=lw)
ax[0].plot([grid_errors[:,0].mean()]*2, [0,1], color='orange', lw=1.0)
ax[0].plot([random_errors[:,0].mean()]*2, [0,1], color='royalblue', lw=1.0)
ax[0].set_title('$\eta$', fontdict={'fontsize': TITLE_FONT_SIZE})

ax[1].step(x, random_hist_g, color='royalblue', lw=lw, label='randomly\nsampled')
ax[1].step(x, grid_hist_g, color='orange', lw=lw, label='grid-\nsampled')
ax[1].plot([grid_errors[:,1].mean()]*2, [0,1], color='orange', lw=1.0)
ax[1].plot([random_errors[:,1].mean()]*2, [0,1], color='royalblue', lw=1.0)
ax[1].set_title('$g$', fontdict={'fontsize': TITLE_FONT_SIZE})

ax[2].step(x, random_hist_j, color='royalblue', lw=lw)
ax[2].step(x, grid_hist_j, color='orange', lw=lw)
ax[2].plot([grid_errors[:,2].mean()]*2, [0,1], color='orange', lw=1.0)
ax[2].plot([random_errors[:,2].mean()]*2, [0,1], color='royalblue', lw=1.0)
ax[2].set_title('$J$', fontdict={'fontsize': TITLE_FONT_SIZE})

ax[0].set_ylabel('error dist.', fontdict={'fontsize': LABEL_FONT_SIZE})
for a in ax:
    a.tick_params('both', labelsize=LABEL_FONT_SIZE)
    a.set_xlim(-0.1,0.1)
    a.set_ylim(0,0.3)
    a.set_xlabel('error', fontdict={'fontsize': LABEL_FONT_SIZE})

ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].set_yticklabels([])
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[2].set_yticklabels([])
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[1].legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(0.6, 0.5, 0.5, 0.5), framealpha=0)

# plt.savefig(os.path.join(fig_dir, 'Fig11.pdf'))
# plt.savefig(os.path.join(fig_dir, 'Fig11.eps'))
plt.show()
