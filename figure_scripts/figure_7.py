import numpy as np
import matplotlib.pyplot as plt
import os, keras, h5py
from scipy.signal import welch
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
from helper_functions import calc_psds, read_ids_parameters, remove_zero_lfps

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2
LARGE_MODEL_PATH = os.path.join('../training_scripts/save/model_full_parameterspace.h5')
SMALL_MODEL_PATH = os.path.join('../training_scripts/save/model_small_parameterspace.h5')

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


#### Load small parameterspace LFPs
ids, labels_small = read_ids_parameters(os.path.join(DATA_DIR_SMALL, 'id_parameters.txt'))

## Test sims
ids = ids[:10000]
labels_small = labels_small[:10000]

lfps = np.zeros((len(ids), 6, 2851), dtype=np.float32)
for j, i in enumerate(ids[:10000]):
    with h5py.File(os.path.join(DATA_DIR_SMALL, 'nest_output', i, 'LFP_firing_rate.h5')) as f:
        lfp = f['data'][:,150:]
    lfps[j] = lfp

lfps = np.array(lfps)
labels_small = np.array(labels_small)

# Get PSDs
fs, psds_small = calc_psds(lfps)
del lfps

## Remove cases of simulations where no neurons spiked
psds_small, labels_small = remove_zero_lfps(psds_small, labels_small)

psds_small = np.swapaxes(psds_small, 1,2)

## Rescale labels
labels_large_rescaled = labels_large - np.array([[0.8, 3.5, 0.05]])
labels_large_rescaled /= np.array([[3.2, 4.5, 0.35]])

labels_small_rescaled = labels_small - np.array([[0.8, 3.5, 0.05]])
labels_small_rescaled /= np.array([[3.2, 4.5, 0.35]])

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
full_model.load_weights(LARGE_MODEL_PATH)
preds = full_model.predict(psds_large)
full_errors = preds - labels_large_rescaled

preds = full_model.predict(psds_small)
fs_errors = preds - labels_small_rescaled

small_model = set_up_model(psds_small, 0)
small_model.load_weights(SMALL_MODEL_PATH)
small_preds = small_model.predict(psds_small)
small_errors = small_preds - labels_small_rescaled


#### Create figure
fig = plt.figure()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE, 3.5])
gs = gridspec.GridSpec(2, 3, wspace=0.3, hspace=0.5)
top_axes = [plt.subplot(gs[0,i]) for i in range(3)]

## Error histograms
bins=np.linspace(-0.7, 0.7, 300)
x = bins[1:] - (bins[-1]-bins[-2])/2
full_hist_eta, _ = np.histogram(full_errors[:,0], bins=bins)
small_hist_eta, _ = np.histogram(small_errors[:,0], bins=bins)
fs_hist_eta, _ = np.histogram(fs_errors[:,0], bins=bins)

full_normed_eta = full_hist_eta/full_hist_eta.sum()
small_normed_eta = small_hist_eta/small_hist_eta.sum()
fs_normed_eta = fs_hist_eta/fs_hist_eta.sum()

top_axes[0].step(x, full_normed_eta, color='orange', lw=2.0)
top_axes[0].step(x, small_normed_eta, color='royalblue', lw=2.0)
top_axes[0].step(x, fs_normed_eta, color='darkorchid', lw=2.0)
top_axes[0].plot([small_errors[:,0].mean()]*2, [0,1], 'royalblue', lw=1.0)
top_axes[0].plot([full_errors[:,0].mean()]*2, [0,1], 'orange', lw=1.0)
top_axes[0].plot([fs_errors[:,0].mean()]*2, [0,1], 'darkorchid', lw=1.0)


full_hist_g, _ = np.histogram(full_errors[:,1], bins=bins)
small_hist_g, _ = np.histogram(small_errors[:,1], bins=bins)
fs_hist_g, _ = np.histogram(fs_errors[:,1], bins=bins)

full_normed_g = full_hist_g/full_hist_g.sum()
small_normed_g = small_hist_g/small_hist_g.sum()
fs_normed_g = fs_hist_g/fs_hist_g.sum()
top_axes[1].step(x, full_normed_g, color='orange', lw=2.0)
top_axes[1].step(x, small_normed_g, color='royalblue', lw=2.0)
top_axes[1].step(x, fs_normed_g, color='darkorchid', lw=2.0)
top_axes[1].plot([small_errors[:,1].mean()]*2, [0,1], 'royalblue', lw=1.0)
top_axes[1].plot([full_errors[:,1].mean()]*2, [0,1], 'orange', lw=1.0)
top_axes[1].plot([fs_errors[:,1].mean()]*2, [0,1], 'darkorchid', lw=1.0)

full_hist_j, _ = np.histogram(full_errors[:,2], bins=bins)
small_hist_j, _ = np.histogram(small_errors[:,2], bins=bins)
fs_hist_j, _ = np.histogram(fs_errors[:,2], bins=bins)
full_normed_j = full_hist_j/full_hist_j.sum()
small_normed_j = small_hist_j/small_hist_j.sum()
fs_normed_j = fs_hist_j/fs_hist_j.sum()
top_axes[2].step(x, full_normed_j, color='orange', lw=2.0)
top_axes[2].step(x, small_normed_j, color='royalblue', lw=2.0)
top_axes[2].step(x, fs_normed_j, color='darkorchid', lw=2.0)
top_axes[2].plot([small_errors[:,2].mean()]*2, [0,1], 'royalblue', lw=1.0)
top_axes[2].plot([full_errors[:,2].mean()]*2, [0,1], 'orange', lw=1.0)
top_axes[2].plot([fs_errors[:,2].mean()]*2, [0,1], 'darkorchid', lw=1.0)

top_axes[0].set_ylim(0,0.3)
top_axes[1].set_ylim(0,0.3)
top_axes[2].set_ylim(0,0.3)
top_axes[0].set_xlim(-0.1, 0.1)
top_axes[1].set_xlim(-0.1, 0.1)
top_axes[2].set_xlim(-0.1, 0.1)

top_axes[0].set_title('$\eta$', pad=13, fontdict={'fontsize': TITLE_FONT_SIZE})
top_axes[1].set_title('$g$', pad=13, fontdict={'fontsize': TITLE_FONT_SIZE})
top_axes[2].set_title('$J$', pad=13, fontdict={'fontsize': TITLE_FONT_SIZE})

## Real scale on top axis
multipliers = np.array([3.2, 4.5, 0.35])
twaxes = []
for i, a in enumerate(top_axes):
    twax = a.twiny()
    xmin, xmax = a.get_xlim()
    twax.set_xlim(xmin*multipliers[i], xmax*multipliers[i])
    twax.xaxis.set_minor_locator(AutoMinorLocator(2))
    twax.xaxis.set_tick_params(pad=-2)
    twaxes.append(twax)
for ax in top_axes:
    ax.set_xlabel('error', labelpad=1.0, fontdict={'fontsize': LABEL_FONT_SIZE})

### Cumulative plots

def get_cumulative_dist(errors):
    abs_hist, _ = np.histogram(np.abs(errors), bins=bins)
    cumsum = abs_hist.cumsum(dtype=np.float64)
    return cumsum / cumsum[-1]


bins = np.linspace(0, 0.7, 300)
full_cumdist_eta = get_cumulative_dist(full_errors[:,0])
full_cumdist_g = get_cumulative_dist(full_errors[:,1])
full_cumdist_j = get_cumulative_dist(full_errors[:,2])

small_cumdist_eta = get_cumulative_dist(small_errors[:,0])
small_cumdist_g = get_cumulative_dist(small_errors[:,1])
small_cumdist_j = get_cumulative_dist(small_errors[:,2])

fs_cumdist_eta = get_cumulative_dist(fs_errors[:,0])
fs_cumdist_g = get_cumulative_dist(fs_errors[:,1])
fs_cumdist_j = get_cumulative_dist(fs_errors[:,2])


lw=2.0
bottom_axes = [plt.subplot(gs[1,i]) for i in range(3)]

bottom_axes[0].step(bins, [0] + list(full_cumdist_eta), color='orange', lw=lw)
bottom_axes[0].step(bins, [0] + list(small_cumdist_eta), color='royalblue', lw=lw)
bottom_axes[0].step(bins, [0] + list(fs_cumdist_eta), color='darkorchid', lw=lw)

full_acc_90 = bins[(full_cumdist_eta >= 0.9).argmax()]
small_acc_90 = bins[(small_cumdist_eta >= 0.9).argmax()]
fs_acc_90 = bins[(fs_cumdist_eta >= 0.9).argmax()]
bottom_axes[0].plot([full_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[0].plot([small_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[0].plot([fs_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[0].plot([0, full_acc_90], [0.9, 0.9], color='black', lw=1.0, linestyle='dashed')

bottom_axes[1].step(bins, [0] + list(full_cumdist_g), color='orange', lw=lw, label='Full parameter')
bottom_axes[1].step(bins, [0] + list(small_cumdist_g), color='royalblue', lw=lw, label='AI state only')
bottom_axes[1].step(bins, [0] + list(fs_cumdist_g), color='darkorchid', lw=lw, label='Full parameter\nevaluated on\nAI state only')
bottom_axes[1].legend(framealpha=0, bbox_to_anchor=(0.24, 0.25, 0.8, 0.4), fontsize=LABEL_FONT_SIZE)

full_acc_90 = bins[(full_cumdist_g >= 0.9).argmax()]
small_acc_90 = bins[(small_cumdist_g >= 0.9).argmax()]
fs_acc_90 = bins[(fs_cumdist_g >= 0.9).argmax()]
bottom_axes[1].plot([full_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[1].plot([small_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[1].plot([fs_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[1].plot([0, full_acc_90], [0.9, 0.9], color='black', lw=1.0, linestyle='dashed')

bottom_axes[2].step(bins, [0] + list(full_cumdist_j), color='orange', lw=lw)
bottom_axes[2].step(bins, [0] + list(small_cumdist_j), color='royalblue', lw=lw)
bottom_axes[2].step(bins, [0] + list(fs_cumdist_j), color='darkorchid', lw=lw)

full_acc_90 = bins[(full_cumdist_j >= 0.9).argmax()]
small_acc_90 = bins[(small_cumdist_j >= 0.9).argmax()]
fs_acc_90 = bins[(fs_cumdist_j >= 0.9).argmax()]
bottom_axes[2].plot([full_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[2].plot([small_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[2].plot([fs_acc_90]*2, [0, 0.9], color='black', lw=1.0, linestyle='dashed')
bottom_axes[2].plot([0, full_acc_90], [0.9, 0.9], color='black', lw=1.0, linestyle='dashed')

top_axes[0].set_ylabel('error dist.', fontdict={'fontsize': LABEL_FONT_SIZE})
bottom_axes[0].set_ylabel('cumulative dist.', fontdict={'fontsize': LABEL_FONT_SIZE})

for ax in bottom_axes:
    ax.set_xlabel('abs. error', labelpad=1, fontdict={'fontsize': LABEL_FONT_SIZE})
    ax.set_xlim(0, 0.08)
    ax.set_ylim(0, 1.05)

tax = fig.add_axes([0.03, 0.84, 0.05, 0.05])
tax.axis('off')
tax.text(0, 0, 'A', fontdict={'fontsize': 12})

tax = fig.add_axes([0.03, 0.38, 0.05, 0.05])
tax.axis('off')
tax.text(0, 0, 'B', fontdict={'fontsize': 12})

for ax in top_axes + bottom_axes + twaxes:
    ax.tick_params('both', labelsize=TICK_FONT_SIZE)

for ax in top_axes[1:] + bottom_axes[1:]:
    ax.set_yticklabels('')


# fig.savefig(os.path.join(fig_dir, 'error_histograms.pdf'))
# fig.savefig(os.path.join(fig_dir, 'Fig7.pdf'))
# fig.savefig(os.path.join(fig_dir, 'Fig7.eps'))
plt.show()
