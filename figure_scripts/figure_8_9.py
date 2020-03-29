import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import os, keras, h5py
from helper_functions import read_ids_parameters, calc_psds, remove_zero_lfps

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2

DATA_DIR_LARGE = os.path.join('../simulation_code/lfp_simulations_large/')
DATA_DIR_SMALL = os.path.join('../simulation_code/lfp_simulations_small/')
LARGE_MODEL_PATH = os.path.join('../training_scripts/save/model_full_parameterspace.h5')
SMALL_MODEL_PATH = os.path.join('../training_scripts/save/model_small_parameterspace.h5')
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

## Get PSDs
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

small_model = set_up_model(psds_small, 0)
small_model.load_weights(SMALL_MODEL_PATH)
small_preds = small_model.predict(psds_small)
small_errors = small_preds - labels_small_rescaled

full_abs_errors = np.abs(full_errors)
small_abs_errors = np.abs(small_errors)

full_labels = labels_large
small_labels = labels_small

## Parameter intervals to average over
j_boxes_full = np.array([0.049, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.401])
j_boxes_small = np.array([0.099, 0.15, 0.2, 0.251])

eta_boxes_full = np.array([0.79, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.01])
eta_boxes_small = np.array([1.49, 1.75, 2.0, 2.25, 2.50, 2.75, 3.01])

g_boxes_full = np.array([3.49, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.01])
g_boxes_small = np.array([4.49, 4.75,  5.0, 5.25, 5.50, 5.75, 6.01])


## Create boolean masks to get errors in all intervals
full_j_masks = [(full_labels[:,2] >= j_boxes_full[i]) & (full_labels[:,2] < j_boxes_full[i+1])
                                                                    for i in range(len(j_boxes_full)-1)]
full_eta_masks = [(full_labels[:,0] >= eta_boxes_full[i]) & (full_labels[:,0] < eta_boxes_full[i+1])
                                                                    for i in range(len(eta_boxes_full)-1)]
full_g_masks = [(full_labels[:,1] >= g_boxes_full[i]) & (full_labels[:,1] < g_boxes_full[i+1])
                                                                    for i in range(len(g_boxes_full)-1)]
small_j_masks = [(small_labels[:,2] >= j_boxes_small[i]) & (small_labels[:,2] < j_boxes_small[i+1])
                                                                    for i in range(len(j_boxes_small)-1)]
small_g_masks = [(small_labels[:,1] >= g_boxes_small[i]) & (small_labels[:,1] < g_boxes_small[i+1])
                                                                    for i in range(len(g_boxes_small)-1)]
small_eta_masks = [(small_labels[:,0] >= eta_boxes_small[i]) & (small_labels[:,0] < eta_boxes_small[i+1])
                                                                    for i in range(len(eta_boxes_small)-1)]

## Check number of examples in each bin
N = []
[N.append((eta & g & j).sum()) for eta in full_eta_masks for j in full_j_masks for g in full_g_masks]
print(min(N))
print(max(N))
print(np.mean(N))

N_small = []
[N_small.append((eta & g & j).sum()) for eta in small_eta_masks for j in small_j_masks for g in small_g_masks]
print(min(N_small))
print(max(N_small))
print(np.mean(N_small))

## Reshape errors into an array of shape (J, eta, g)
full_eta_error_cube = np.array([[[full_abs_errors[(g & eta & j),0].mean()
                                                    for g in full_g_masks]
                                                    for eta in full_eta_masks]
                                                    for j in full_j_masks])

full_g_error_cube = np.array([[[full_abs_errors[(g & eta & j),1].mean()
                                                    for g in full_g_masks]
                                                    for eta in full_eta_masks]
                                                    for j in full_j_masks])

full_j_error_cube = np.array([[[full_abs_errors[(g & eta & j),2].mean()
                                                    for g in full_g_masks]
                                                    for eta in full_eta_masks]
                                                    for j in full_j_masks])

small_eta_error_cube = np.array([[[small_abs_errors[(g & eta & j),0].mean()
                                                    for g in small_g_masks]
                                                    for eta in small_eta_masks]
                                                    for j in small_j_masks])

small_g_error_cube = np.array([[[small_abs_errors[(g & eta & j),1].mean()
                                                    for g in small_g_masks]
                                                    for eta in small_eta_masks]
                                                    for j in small_j_masks])

small_j_error_cube = np.array([[[small_abs_errors[(g & eta & j),2].mean()
                                                    for g in small_g_masks]
                                                    for eta in small_eta_masks]
                                                    for j in small_j_masks])


## Create figure for large parameter space
j_labels = ['$J \in [%.2f, %.2f)$'%(j_boxes_full[i], j_boxes_full[i+1])
                                    for i in range(len(j_boxes_full)-1)]
j_labels[-1] = j_labels[-1].replace(')', ']')

fig = plt.figure()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE, 5.7])

eta_ticks = [0, 2, 4, 6, 8]
g_ticks = [1, 3, 5, 7, 9]
eta_tick_labels = ['0.8', '1.6', '2.4', '3.2', '4.0']
g_tick_labels = ['4', '5', '6', '7', '8']

outer_gs = gridspec.GridSpec(14, 3, hspace=0.1, wspace=0.3, right=0.9)
large_eta_gs = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_gs[:8,0], hspace=0.3)
large_g_gs = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_gs[:8,1], hspace=0.3)
large_j_gs = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_gs[:8,2], hspace=0.3)

eta_axes = []
for i in range(4):
    for j in range(2):
        eta_axes.append(plt.subplot(large_eta_gs[i,j]))
eta_axes[-1].axis('off')
del eta_axes[-1]
for ax in eta_axes:
    ax.set_yticks([])
    ax.set_xticks([])

for ax in eta_axes[1:4]:
    ax.add_patch(
        patches.Rectangle(
          (2.0, 1.875),
          3.0,
          3.75,
          color='red',
          fill=False))


g_axes = []
for i in range(4):
    for j in range(2):
        g_axes.append(plt.subplot(large_g_gs[i,j]))
g_axes[-1].axis('off')
del g_axes[-1]
g_axes[-1].axis('off')
for ax in g_axes:
    ax.set_yticks([])
    ax.set_xticks([])

for ax in g_axes[1:4]:
    ax.add_patch(
        patches.Rectangle(
          (2.0, 1.875),
          3.0,
          3.75,
          color='red',
          fill=False))

j_axes = []
for i in range(4):
    for j in range(2):
        j_axes.append(plt.subplot(large_j_gs[i,j]))
j_axes[-1].axis('off')
del j_axes[-1]
for ax in j_axes:
    ax.set_yticks([])
    ax.set_xticks([])

for ax in j_axes[1:4]:
  ax.add_patch(
  patches.Rectangle(
  (2.0, 1.875),
  3.0,
  3.75,
  color='red',
  fill=False))

def plot_accuracies(axes, error_arr):
    for i, errors in enumerate(error_arr):
        im = axes[i].pcolormesh(errors, vmin=vmin, vmax=vmax)
        axes[i].set_title(j_labels[i], pad=3, fontdict={'fontsize':TITLE_FONT_SIZE})
        axes[-1].set_yticks(eta_ticks)
        axes[-1].set_yticklabels(eta_tick_labels, fontdict={'fontsize': TICK_FONT_SIZE})
        axes[-1].set_xticks(g_ticks)
        axes[-1].set_xticklabels(g_tick_labels, fontdict={'fontsize': TICK_FONT_SIZE})
        axes[-1].set_xlabel('$g$', labelpad=-1, fontdict={'rotation':0, 'fontsize': LABEL_FONT_SIZE})
        axes[-1].set_ylabel('$\eta$', fontdict={'rotation':0, 'fontsize': LABEL_FONT_SIZE})
    return im

vmin=0.0
vmax=0.1

plot_accuracies(eta_axes, full_eta_error_cube)
plot_accuracies(g_axes, full_g_error_cube)
im = plot_accuracies(j_axes, full_j_error_cube)
cbar_ax = fig.add_axes([0.92, 0.656, 0.01, 0.225])
plt.colorbar(mappable=im, cax=cbar_ax)
cbar_ax.tick_params('both', labelsize=TICK_FONT_SIZE)
titles = ['$\eta$', '$g$', '$J$']
for i in range(3):
    tax = fig.add_axes([0.199+0.284*i, 0.92, 0.05, 0.025])
    tax.axis('off')
    tax.text(0.5, 0, titles[i], fontdict={'fontsize': TITLE_FONT_SIZE})

small_eta_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[10:,0], hspace=0.3)
small_g_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[10:,1], hspace=0.3)
small_j_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[10:,2], hspace=0.3)

j_labels = ['$J \in [%.2f, %.2f)$'%(j_boxes_small[i], j_boxes_small[i+1])
                                    for i in range(len(j_boxes_small)-1)]
j_labels[-1] = j_labels[-1].replace(')', ']')

eta_ticks = [0, 2, 4, 6]
g_ticks = [0, 2, 4, 6]
eta_tick_labels = ['1.5', '2.0', '2.5', '3.0']
g_tick_labels = ['4.5', '5.0', '5.5', '6.0']

eta_axes = []
for i in range(3):
    eta_axes.append(plt.subplot(small_eta_gs[i]))
for ax in eta_axes:
    ax.set_xticks([])
    ax.set_yticks([])

g_axes = []
for i in range(3):
    g_axes.append(plt.subplot(small_g_gs[i]))
for ax in g_axes:
    ax.set_xticks([])
    ax.set_yticks([])

j_axes = []
for i in range(3):
    j_axes.append(plt.subplot(small_j_gs[i]))
for ax in j_axes:
    ax.set_xticks([])
    ax.set_yticks([])

vmin = 0
vmax = 0.03

plot_accuracies(eta_axes, small_eta_error_cube)
plot_accuracies(g_axes, small_g_error_cube)
im = plot_accuracies(j_axes, small_j_error_cube)

cbar_ax = fig.add_axes([0.92, 0.103, 0.01, 0.225])
plt.colorbar(mappable=im, cax=cbar_ax)
cbar_ax.tick_params('both', labelsize=TICK_FONT_SIZE)

fig.text(0.1, 0.91, 'A', fontsize=12)
fig.text(0.38, 0.91, 'B', fontsize=12)
fig.text(0.66, 0.91, 'C', fontsize=12)

fig.text(0.1, 0.355, 'D', fontsize=12)
fig.text(0.38, 0.355, 'E', fontsize=12)
fig.text(0.66, 0.355, 'F', fontsize=12)

# fig.savefig('Fig8.pdf', bbox_inches='tight')
# fig.savefig('Fig8.eps', bbox_inches='tight')
plt.show()
