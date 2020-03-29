import h5py, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.signal import welch
from scipy.stats import entropy, variation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from helper_functions import read_ids_parameters

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2

data_dir = os.path.join('../simulation_code/heat_plot_simulations')
fig_dir = os.path.join('.')

## Parameter values used in heat plots
etas = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0], dtype=np.float32)
gs = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0], dtype=np.float32)
js = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], dtype=np.float32)

## Grid for pcolormesh
X, Y = np.meshgrid(np.array([3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25]),
                   np.array([0.75, 0.85, 0.95, 1.05, 1.15, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2]))

ids, labels = read_ids_parameters(os.path.join(data_dir, 'id_parameters.txt'))
labels = np.array(labels)

def normalize(x):
    return x/x.sum()

def convert_to_spiketrains(spike_arr):
    """
    Converts array of neuron IDs and spike times to list of spiketrains
    """
    id_sorted = spike_arr[spike_arr[:,0].argsort()]
    _, first_indices = np.unique(id_sorted[:,0], return_index=True)
    spiketrains = []
    for j, i in enumerate(first_indices):
        if i != first_indices[-1]:
            ii = first_indices[j+1]
        else:
            ii = len(id_sorted)
        spiketrain = np.sort(id_sorted[i:ii,1])
        spiketrains.append(spiketrain[(spiketrain > 500.0)])
    return spiketrains

def calculate_cvs(spiketrain_list):
    """
    Calculates average CV from list of spiketrains
    """
    cvs = []
    for spiketrain in spiketrain_list:
        if len(spiketrain) < 3:
            continue
        isi = np.ediff1d(spiketrain)
        cvs.append(variation(isi))
    if len(cvs) == 0:
        return np.nan
    else:
        return np.mean(cvs)

def calculate_psd_entropy(hist):
    fs, psd = welch(hist, nperseg=300, window='hann')
    return entropy(normalize(psd))


cvs = []
lfp_entropies = []
firing_rates = []
missing_spiketrains = []
hists = []
lfp_stds = []

## Loop through all simulations and calculate statistics
for j, i in enumerate(ids):
    print(j)
    with h5py.File(os.path.join(data_dir, 'nest_output', i, 'LFP_firing_rate.h5')) as f:
        ex_spikes = f['ex_spikes'][()]
        in_spikes = f['in_spikes'][()]
        hist = f['ex_hist'][500:] + f['in_hist'][500:]
        lfp = f['data'][:,500:]
    spiketrains = convert_to_spiketrains(np.concatenate((ex_spikes, in_spikes)))
    if len(spiketrains) != 1000:
        missing_spiketrains.append((j, len(spiketrains)))
    cv = calculate_cvs(spiketrains)
    hists.append(hist)
    f = hist.sum() / 30 / 12500
    if f < 0.01:
        lfp_s = np.nan
    else:
        lfp_s = calculate_psd_entropy(lfp[0])
    lfp_stds.append(np.std(lfp[0]))
    cvs.append(cv)
    lfp_entropies.append(lfp_s)
    firing_rates.append(f)

def make_parameter_cube(flat_arr, parameters):
    param_values = [np.unique(parameters[:,i]) for i in range(parameters.shape[1])]
    cube_arr = np.zeros([len(p) for p in param_values] + list(flat_arr.shape[1:]))
    for i, param in enumerate(parameters):
        pos = [(param[i] == param_values[i]).nonzero()[0][0] for i, _ in enumerate(param)]
        cube_arr[tuple(pos)] = flat_arr[i]
    return cube_arr

cvs = np.array(cvs)
lfp_entropies = np.array(lfp_entropies)
firing_rates = np.array(firing_rates)
lfp_stds = np.array(lfp_stds)

cv_cube = make_parameter_cube(cvs, labels)
lfp_entropy_cube = make_parameter_cube(lfp_entropies, labels)
firing_rate_cube = make_parameter_cube(firing_rates, labels)
lfp_stds_cube = make_parameter_cube(lfp_stds, labels)

def plot_heatmap(axes, array, norm=None):
    for i in range(4):
        im = axes[i].pcolormesh(X, Y, array[:,:,i*2], vmin=vmin, vmax=vmax, norm=norm)
        axes[i].set_title(j_labels[i], pad=3, fontdict={'fontsize':TITLE_FONT_SIZE})
        axes[-1].set_yticks(eta_ticks)
        axes[-1].set_yticklabels(eta_tick_labels, fontdict={'fontsize': TICK_FONT_SIZE})
        axes[-1].set_xticks(g_ticks)
        axes[-1].set_xticklabels(g_tick_labels, fontdict={'fontsize': TICK_FONT_SIZE})
        axes[-1].set_xlabel('$g$', labelpad=-1, fontdict={'rotation':0, 'fontsize': LABEL_FONT_SIZE})
        axes[-1].set_ylabel('$\eta$', fontdict={'rotation':0, 'fontsize': LABEL_FONT_SIZE})
    return im

## Parameter values for ticks
j_labels = ['J = 0.05', 'J = 0.15', 'J = 0.25', 'J = 0.35']
eta_ticks = [etas[0], etas[5], etas[7], etas[9], etas[11]]
g_ticks = gs[1::2]
eta_tick_labels = ['0.8', '1.6', '2.4', '3.2', '4.0']
g_tick_labels = ['4', '5', '6', '7', '8']

## Set up figure
fig = plt.figure()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE, 5.0])
outer_gs = gridspec.GridSpec(1, 4, hspace=0.1, wspace=0.8, right=0.9)
frate_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_gs[0], hspace=0.3)
cv_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_gs[1], hspace=0.3)
lfp_stds_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_gs[2], hspace=0.3)
lfp_entropy_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_gs[3], hspace=0.3)

frate_axes = []
for i in range(4):
        frate_axes.append(plt.subplot(frate_gs[i]))
for ax in frate_axes:
    ax.set_yticks([])
    ax.set_xticks([])

cv_axes = []
for i in range(4):
    cv_axes.append(plt.subplot(cv_gs[i]))
for ax in cv_axes:
    ax.set_yticks([])
    ax.set_xticks([])

lfp_stds_axes = []
for i in range(4):
        lfp_stds_axes.append(plt.subplot(lfp_stds_gs[i]))
for ax in lfp_stds_axes:
    ax.set_yticks([])
    ax.set_xticks([])

lfp_entropy_axes = []
for i in range(4):
        lfp_entropy_axes.append(plt.subplot(lfp_entropy_gs[i]))
for ax in lfp_entropy_axes:
    ax.set_yticks([])
    ax.set_xticks([])

## Plot firing rates
vmin = 1
vmax = 440
im = plot_heatmap(frate_axes, firing_rate_cube, norm=LogNorm(vmin=1, vmax=440))
cax = fig.add_axes([0.1, 0.3, 0.01, 0.4])
plt.colorbar(mappable=im, cax=cax)
cax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
cax.yaxis.set_ticks_position('left')
cax.yaxis.set_label_position('left')
cax.set_ylabel('Hz', fontdict={'fontsize': LABEL_FONT_SIZE})

## Plot CVs
vmin=0
vmax=2.5
im = plot_heatmap(cv_axes, cv_cube)
cax = fig.add_axes([0.48, 0.3, 0.01, 0.4])
cax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
plt.colorbar(mappable=im, cax=cax)

## Plot LFP STDs
vmin = 0.002
vmax = 2.0
im = plot_heatmap(lfp_stds_axes, lfp_stds_cube, norm=LogNorm())
cax = fig.add_axes([0.7, 0.3, 0.01, 0.4])
cax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
plt.colorbar(mappable=im, cax=cax)
cax.set_ylabel('mV', fontdict={'fontsize': LABEL_FONT_SIZE}, labelpad=-7)

## Plot LFP entropies
vmin = 0.85
vmax = 3.4
im = plot_heatmap(lfp_entropy_axes, lfp_entropy_cube, norm=None)
cax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
cax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
plt.colorbar(mappable=im, cax=cax)

## Set titles
xpos = [0.13, 0.4, 0.59, 0.8]
titles = ['Mean firing rate', 'CV', 'LFP STD', 'LFP Entropy']
for i in range(4):
    tax = fig.add_axes([xpos[i], 0.925, 0.1, 0.03])
    tax.axis('off')
    tax.text(0, 0, titles[i], fontdict={'fontsize': TITLE_FONT_SIZE})

## Set labels
abc = ['A', 'B', 'C', 'D']
for i in range(4):
    tax = fig.add_axes([0.1+0.217*i, 0.9, 0.02, 0.02])
    tax.axis('off')
    tax.text(0, 0, abc[i], fontdict={'fontsize': 12})

## Add points showing values of examples given in next figure
example_parameters = [np.array([2.0, 3.5, 0.25], dtype=np.float32),
                      np.array([4.0, 7.0, 0.15], dtype=np.float32),
                      np.array([0.9, 7.0, 0.25], dtype=np.float32),
                      np.array([2.8, 5.5, 0.05], dtype=np.float32),
                      np.array([2.0, 5.0, 0.35], dtype=np.float32)]
labels = ['A', 'B', 'C', 'D', 'E']
for i, p in enumerate(example_parameters):
    ax_n = (p[-1] == np.array([0.05, 0.15, 0.25, 0.35], dtype=np.float32)).nonzero()[0][0]
    circ = Circle((p[1], p[0]), 0.08, color='red')
    frate_axes[ax_n].add_artist(circ)
    tpos = (p[1]+0.2, p[0] - 0.2)
    frate_axes[ax_n].text(*tpos, labels[i], fontsize=8.0, color='white')


# fig.savefig(os.path.join(fig_dir, 'frate_cv_amplitude_with_entropy.pdf'))
# fig.savefig(os.path.join(fig_dir, 'Fig6.pdf'))
# fig.savefig(os.path.join(fig_dir, 'Fig6.eps'))
plt.show()
