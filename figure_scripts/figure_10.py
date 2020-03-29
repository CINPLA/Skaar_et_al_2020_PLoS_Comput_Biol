import numpy as np
import matplotlib.pyplot as plt
import os, keras, h5py
import matplotlib.gridspec as gridspec
from helper_functions import read_ids_parameters, calc_psds, remove_zero_lfps

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2

FULL_MODEL_PATH = os.path.join('../training_scripts/save/model_full_parameterspace.h5')
ETA_MODEL_PATH = os.path.join('../training_scripts/save/eta_model_full_parameterspace.h5')
G_MODEL_PATH = os.path.join('../training_scripts/save/g_model_full_parameterspace.h5')
J_MODEL_PATH = os.path.join('../training_scripts/save/j_model_full_parameterspace.h5')

DATA_DIR_LARGE = os.path.join('../simulation_code/lfp_simulations_large/')
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
full_model.load_weights(FULL_MODEL_PATH)
preds = full_model.predict(psds_large)
full_errors = preds - labels_large_rescaled

eta_model = set_up_model(psds_large, 0, output=1)
eta_model.load_weights(ETA_MODEL_PATH)
eta_preds = eta_model.predict(psds_large)
eta_errors = eta_preds.squeeze() - labels_large_rescaled[:10000,0]

g_model = set_up_model(psds_large, 0, output=1)
g_model.load_weights(G_MODEL_PATH)
g_preds = g_model.predict(psds_large)
g_errors = g_preds.squeeze() - labels_large_rescaled[:10000,1]

j_model = set_up_model(psds_large, 0, output=1)
j_model.load_weights(J_MODEL_PATH)
j_preds = j_model.predict(psds_large)
j_errors = j_preds.squeeze() - labels_large_rescaled[:10000,2]

## Create histograms
bins=np.linspace(-0.7, 0.7, 300)
x = bins[1:] - (bins[-1]-bins[-2])/2

full_hist_eta, _ = np.histogram(full_errors[:,0], bins=bins)
full_hist_eta = full_hist_eta/full_hist_eta.sum()

full_hist_g, _ = np.histogram(full_errors[:,1], bins=bins)
full_hist_g = full_hist_g/full_hist_g.sum()

full_hist_j, _ = np.histogram(full_errors[:,2], bins=bins)
full_hist_j = full_hist_j/full_hist_j.sum()

single_hist_eta, _ = np.histogram(eta_errors, bins=bins)
single_hist_eta = single_hist_eta/single_hist_eta.sum()

single_hist_g, _ = np.histogram(g_errors, bins=bins)
single_hist_g = single_hist_g/single_hist_g.sum()

single_hist_j, _ = np.histogram(j_errors, bins=bins)
single_hist_j = single_hist_j/single_hist_j.sum()

## Create figure
lw=2.0
fig = plt.figure()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE, 1.8])
gs = gridspec.GridSpec(ncols=3, nrows=1, wspace=0.3, bottom=0.3)
ax = [plt.subplot(gs[i]) for i in range(3)]

ax[0].step(x, full_hist_eta, color='royalblue', lw=lw)
ax[0].step(x, single_hist_eta, color='orange', lw=lw)
ax[0].plot([eta_errors.mean()]*2, [0,1], color='orange', lw=1.0)
ax[0].plot([full_errors[:,0].mean()]*2, [0,1], color='royalblue', lw=1.0)
ax[0].set_title('$\eta$', fontdict={'fontsize': TITLE_FONT_SIZE})

ax[1].step(x, full_hist_g, color='royalblue', lw=lw, label='combined\npredictions')
ax[1].step(x, single_hist_g, color='orange', lw=lw, label='single\npredictions')
ax[1].plot([g_errors.mean()]*2, [0,1], color='orange', lw=1.0)
ax[1].plot([full_errors[:,1].mean()]*2, [0,1], color='royalblue', lw=1.0)
ax[1].set_title('$g$', fontdict={'fontsize': TITLE_FONT_SIZE})

ax[2].step(x, full_hist_j, color='royalblue', lw=lw)
ax[2].step(x, single_hist_j, color='orange', lw=lw)
ax[2].plot([j_errors.mean()]*2, [0,1], color='orange', lw=1.0)
ax[2].plot([full_errors[:,2].mean()]*2, [0,1], color='royalblue', lw=1.0)
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
# plt.savefig(os.path.join(fig_dir, 'Fig10.pdf'))
# plt.savefig(os.path.join(fig_dir, 'Fig10.eps'))
plt.show()
