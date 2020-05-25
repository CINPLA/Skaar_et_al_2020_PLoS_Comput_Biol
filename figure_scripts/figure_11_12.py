import h5py, os
import numpy as np
from scipy.signal import welch
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import matplotlib.gridspec as gridspec
from helper_functions import read_ids_parameters, calc_psds

fig_dir = os.path.join('.')
varying_delay_path = os.path.join('../simulation_code/lfp_simulations_varying_delays')
varying_taumem_path = os.path.join('../simulation_code/lfp_simulations_varying_taumem')
varying_theta_path = os.path.join('../simulation_code/lfp_simulations_varying_theta')
varying_t_ref_path = os.path.join('../simulation_code/lfp_simulations_varying_t_ref')

gaussian_delay_path = os.path.join('../simulation_code/lfp_simulations_gaussian_delay')
gaussian_taumem_path = os.path.join('../simulation_code/lfp_simulations_gaussian_taumem')
gaussian_theta_path = os.path.join('../simulation_code/lfp_simulations_gaussian_theta')
gaussian_t_ref_path = os.path.join('../simulation_code/lfp_simulations_gaussian_t_ref')

def fetch_lfps(path, nest_output='nest_output'):
    ids, labels = read_ids_parameters(os.path.join(path, 'id_parameters.txt'))
    lfps = []
    for i in ids:
        fpath = os.path.join(path, nest_output, i, 'LFP_firing_rate.h5')
        with h5py.File(fpath) as f:
            lfps.append(f['data'][:,150:])
    return np.array(labels), np.array(lfps)

## Get LFPs and PSDs
gt_labels, gt_lfps = fetch_lfps(gaussian_taumem_path)
fs, gt_psds = calc_psds(gt_lfps)
del gt_lfps

gd_labels, gd_lfps = fetch_lfps(gaussian_delay_path)
fs, gd_psds = calc_psds(gd_lfps)
del gd_lfps

gth_labels, gth_lfps = fetch_lfps(gaussian_theta_path)
fs, gth_psds = calc_psds(gth_lfps)
del gth_lfps

gtr_labels, gtr_lfps = fetch_lfps(gaussian_t_ref_path)
fs, gtr_psds = calc_psds(gtr_lfps)
del gtr_lfps

vd_labels, vd_lfps = fetch_lfps(varying_delay_path)
fs, vd_psds = calc_psds(vd_lfps)
del vd_lfps

vt_labels, vt_lfps = fetch_lfps(varying_taumem_path)
fs, vt_psds = calc_psds(vt_lfps)
del vt_lfps

vtr_labels, vtr_lfps = fetch_lfps(varying_t_ref_path)
fs, vtr_psds = calc_psds(vtr_lfps)
del vtr_lfps

vth_labels, vth_lfps = fetch_lfps(varying_theta_path, nest_output='nest_output_correct')
fs, vth_psds = calc_psds(vth_lfps)
nans = np.isnan(vth_psds[:,0,0])
vth_labels = vth_labels[~nans]
vth_psds = vth_psds[~nans]
del vth_lfps

## Transpose
gt_psds = np.swapaxes(gt_psds, 1,2)
gd_psds = np.swapaxes(gd_psds, 1,2)
gtr_psds = np.swapaxes(gtr_psds, 1,2)
gth_psds = np.swapaxes(gth_psds, 1,2)

vd_psds = np.swapaxes(vd_psds, 1,2)
vt_psds = np.swapaxes(vt_psds, 1,2)
vtr_psds = np.swapaxes(vtr_psds, 1,2)
vth_psds = np.swapaxes(vth_psds, 1,2)

def set_up_model(lr, n_dense=128, output=3):
    keras.backend.clear_session()
    input_shape = (151, 6)
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(20, kernel_size=(12), strides=(1), activation='relu', use_bias=False)(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv1D(20, kernel_size=(4), strides=(1), activation='relu', use_bias=False)(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Conv1D(20, kernel_size=(4), strides=(1), activation='relu', use_bias=False)(x)
    x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(n_dense, activation='relu')(x)
    x = keras.layers.Dense(n_dense, activation='relu')(x)
    predictions = keras.layers.Dense(output, activation=None, use_bias=False)(x)
    model = keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['mse', 'mae'])
    print(model.summary())
    return model

def rescale_labels(labels):
    labels_rescaled = labels[:,:3] - np.array([[0.8, 3.5, 0.05]])
    labels_rescaled /= np.array([[3.2, 4.5, 0.35]])
    return labels_rescaled

vd_labels_rs = rescale_labels(vd_labels)
vt_labels_rs = rescale_labels(vt_labels)
vth_labels_rs = rescale_labels(vth_labels)
vtr_labels_rs = rescale_labels(vtr_labels)

gt_labels_rs = rescale_labels(gt_labels)
gth_labels_rs = rescale_labels(gth_labels)
gd_labels_rs = rescale_labels(gd_labels)
gtr_labels_rs = rescale_labels(gtr_labels)

model_path = os.path.join('../training_scripts/save/model_full_parameterspace.h5')
model = set_up_model(0)
model.load_weights(model_path)

vd_preds = model.predict(vd_psds)
vd_errors = vd_preds - vd_labels_rs

vt_preds = model.predict(vt_psds)
vt_errors = vt_preds - vt_labels_rs

vth_preds = model.predict(vth_psds)
vth_errors = vth_preds - vth_labels_rs

vtr_preds = model.predict(vtr_psds)
vtr_errors = vtr_preds - vtr_labels_rs

gt_preds = model.predict(gt_psds)
gt_errors = gt_preds - gt_labels_rs

gd_preds = model.predict(gd_psds)
gd_errors = gd_preds - gd_labels_rs

gth_preds = model.predict(gth_psds)
gth_errors = gth_preds - gth_labels_rs

gtr_preds = model.predict(gtr_psds)
gtr_errors = gtr_preds - gtr_labels_rs

def get_small_ps(errors, labels):
    args = ((labels[:,0] >= 1.5) & (labels[:,0] <= 3.0)) \
         & ((labels[:,1] >= 4.5) & (labels[:,1] <= 6.0)) \
         & ((labels[:,2] >= 0.1) & (labels[:,2] <= 0.25))
    return errors[args], labels[args]

def get_mean_errors_restricted(errors, labels, varying):
    full_args = [(labels[:,3] == p).nonzero()[0] for p in varying]
    errors_by_varying = [errors[arg] for arg in full_args]
    labels_by_varying = [labels[arg] for arg in full_args]
    means = [np.abs(error).mean(axis=0) for error in errors_by_varying]

    ai_errors = [get_small_ps(errors_by_varying[i], labels_by_varying[i])[0] for i, _ in enumerate(errors_by_varying)]
    ai_means = [np.abs(error).mean(axis=0) for error in ai_errors]
    return means, ai_means

delays = np.unique(vd_labels[:,3])
taumems = np.unique(vt_labels[:,3])
thetas = np.unique(vth_labels[:,3])
t_refs = np.unique(vtr_labels[:,3])

gd_sigmas = np.unique(gd_labels[:,3])
gt_sigmas = np.unique(gt_labels[:,3])
gth_sigmas = np.unique(gth_labels[:,3])
gtr_sigmas = np.unique(gtr_labels[:,3])

vd_means = np.array(get_mean_errors_restricted(vd_errors, vd_labels, delays))
vt_means = np.array(get_mean_errors_restricted(vt_errors, vt_labels, taumems))
vth_means = np.array(get_mean_errors_restricted(vth_errors, vth_labels, thetas))
vtr_means = np.array(get_mean_errors_restricted(vtr_errors, vtr_labels, t_refs))

gd_means = np.array(get_mean_errors_restricted(gd_errors, gd_labels, gd_sigmas))
gt_means = np.array(get_mean_errors_restricted(gt_errors, gt_labels, gt_sigmas))
gth_means = np.array(get_mean_errors_restricted(gth_errors, gth_labels, gth_sigmas))
gtr_means = np.array(get_mean_errors_restricted(gtr_errors, gtr_labels, gtr_sigmas))


def plot_errors(ax, varying_param, means):
    ax.plot(varying_param, means[0], label='full parameter space')
    ax.plot(varying_param, means[1], label='small parameter space')

def plot_errors_logx(ax, varying_param, means):
    ax.semilogx(varying_param, means[0], label='full parameter space')
    ax.semilogx(varying_param, means[1], label='small parameter space')

##### FIGURE 12

TICK_FONT_SIZE = 7
TITLE_FONT_SIZE = 7
LABEL_FONT_SIZE = 7
FULL_WIDTH_FIG_SIZE = 7
HALF_WIDTH_FIG_SIZE = FULL_WIDTH_FIG_SIZE / 2

fig = plt.gcf()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE, 7])
gs = gridspec.GridSpec(5, 3, hspace=0.45, wspace=0.45, top=0.83)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])

ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])

ax7 = plt.subplot(gs[2,0])
ax8 = plt.subplot(gs[2,1])
ax9 = plt.subplot(gs[2,2])

ax10 = plt.subplot(gs[3,0])
ax11 = plt.subplot(gs[3,1])
ax12 = plt.subplot(gs[3,2])


ax13 = plt.subplot(gs[4,1])

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
for ax in axes:
    ax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
    ax.set_xlim(0.01, 1.5)
    ax.set_ylim(0, 0.52)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel('$\sigma$', fontsize=LABEL_FONT_SIZE, labelpad=-1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 0.35)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])


for ax in [ax13]:
    ax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)

for ax in [ax1, ax4, ax7, ax10]:
    ax.set_ylabel('error', fontsize=LABEL_FONT_SIZE)

ax1.set_title('$\eta$ predictions', fontsize=LABEL_FONT_SIZE)

ax2.set_title('$g$ predictions', fontsize=LABEL_FONT_SIZE)

ax3.set_title('$J$ predictions', fontsize=LABEL_FONT_SIZE)

lower, upper = 0.05, 1.95
mu = 1.0
colors = ['C4', 'C3', 'C2']
for i, sigma in enumerate([0.05, 0.1, 1.0]):
    tn = truncnorm((lower - mu)/(sigma*mu), (upper - mu)/(sigma*mu), loc=mu, scale=sigma*mu)
    x = np.linspace(0, 2,  100)
    ax13.plot(x, tn.pdf(x), label='$\sigma$ = %.2f'%sigma, color=colors[i])
ax13.set_ylim(0, 8)
ax13.set_ylabel('pdf', fontsize=LABEL_FONT_SIZE, labelpad=-0.8)
ax13.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(1.1, 1.1), framealpha=0)
ax13.spines['right'].set_visible(False)
ax13.spines['top'].set_visible(False)
ax13.set_xticks([0, 1, 2])
ax13.set_xticklabels(['0', '$\mu$', '$2\mu$'])

plot_errors_logx(ax1, gd_sigmas, gd_means[:,:,0])
plot_errors_logx(ax2, gd_sigmas, gd_means[:,:,1])
plot_errors_logx(ax3, gd_sigmas, gd_means[:,:,2])

plot_errors_logx(ax4, gt_sigmas, gt_means[:,:,0])
plot_errors_logx(ax5, gt_sigmas, gt_means[:,:,1])
plot_errors_logx(ax6, gt_sigmas, gt_means[:,:,2])

plot_errors_logx(ax7, gth_sigmas, gth_means[:,:,0])
plot_errors_logx(ax8, gth_sigmas, gth_means[:,:,1])
plot_errors_logx(ax9, gth_sigmas, gth_means[:,:,2])

plot_errors_logx(ax10, gtr_sigmas, gtr_means[:,:,0])
plot_errors_logx(ax11, gtr_sigmas, gtr_means[:,:,1])
plot_errors_logx(ax12, gtr_sigmas, gtr_means[:,:,2])

ax2.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(1.1,1), framealpha=0)

fig.text(0.03, 0.73, 'gaussian $t_d$', fontsize=8, rotation=90)
fig.text(0.03, 0.58, 'gaussian $\\tau_m$', fontsize=8, rotation=90)
fig.text(0.03, 0.43, 'gaussian $\\theta$', fontsize=8, rotation=90)
fig.text(0.03, 0.27, 'gaussian $t_{ref}$', fontsize=8, rotation=90)


h = [0.07, 0.36, 0.65]
v = 0.84
fig.text(h[0], v, 'A', fontsize=12)
fig.text(h[1], v, 'B', fontsize=12)
fig.text(h[2], v, 'C', fontsize=12)

v = 0.67
fig.text(h[0], v, 'D', fontsize=12)
fig.text(h[1], v, 'E', fontsize=12)
fig.text(h[2], v, 'F', fontsize=12)

v = 0.525
fig.text(h[0], v, 'G', fontsize=12)
fig.text(h[1], v, 'H', fontsize=12)
fig.text(h[2], v, 'I', fontsize=12)

v = 0.37
fig.text(h[0], v, 'J', fontsize=12)
fig.text(h[1], v, 'K', fontsize=12)
fig.text(h[2], v, 'L', fontsize=12)

fig.text(0.36, 0.21, 'M', fontsize=12)
#
# fig.savefig(os.path.join(fig_dir, 'Fig12.pdf'), bbox_inches='tight')
# fig.savefig(os.path.join(fig_dir, 'Fig12.eps'), bbox_inches='tight')
plt.show()

### FIGURE 13
plt.close()
fig = plt.gcf()
fig.set_size_inches([FULL_WIDTH_FIG_SIZE, 7])
gs = gridspec.GridSpec(5, 3, hspace=0.45, wspace=0.45, top=0.83)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])

ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])

ax7 = plt.subplot(gs[2,0])
ax8 = plt.subplot(gs[2,1])
ax9 = plt.subplot(gs[2,2])

ax10 = plt.subplot(gs[3,0])
ax11 = plt.subplot(gs[3,1])
ax12 = plt.subplot(gs[3,2])


axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
for ax in axes:
    ax.tick_params(axis='both', labelsize=LABEL_FONT_SIZE)
    ax.set_ylim(0, 0.52)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


for ax in [ax1, ax2, ax3]:
    ax.set_xlim(1, 2)
    ax.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])

for ax in [ax1, ax4, ax7, ax10]:
    ax.set_ylabel('error', fontsize=LABEL_FONT_SIZE)

for ax in [ax4, ax5, ax6]:
    ax.set_xlim(10, 30)
    ax.set_xticks(list(range(10, 31, 5)))


for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('$t_d \  (\mathrm{ms})$', fontsize=LABEL_FONT_SIZE, labelpad=0.0)

for ax in [ax4, ax5, ax6]:
    ax.set_xlabel('$\\tau_m \ (\mathrm{ms})$', fontsize=LABEL_FONT_SIZE)

for ax in [ax7, ax8, ax9]:
    ax.set_xlabel('$\\theta \ (\mathrm{mV})$', fontsize=LABEL_FONT_SIZE)
    ax.set_xlim(12, 28)

for ax in [ax10, ax11, ax12]:
    ax.set_xlim(0.2, 3.8)
    ax.set_xlabel('$t_{ref} \ (\mathrm{ms})$', fontsize=LABEL_FONT_SIZE)


ax1.set_title('$\eta$ predictions', fontsize=LABEL_FONT_SIZE)

ax2.set_title('$g$ predictions', fontsize=LABEL_FONT_SIZE)

ax3.set_title('$J$ predictions', fontsize=LABEL_FONT_SIZE)

plot_errors(ax1, delays, vd_means[:,:,0])
plot_errors(ax2, delays, vd_means[:,:,1])
plot_errors(ax3, delays, vd_means[:,:,2])

plot_errors(ax4, taumems, vt_means[:,:,0])
plot_errors(ax5, taumems, vt_means[:,:,1])
plot_errors(ax6, taumems, vt_means[:,:,2])

plot_errors(ax7, thetas, vth_means[:,:,0])
plot_errors(ax8, thetas, vth_means[:,:,1])
plot_errors(ax9, thetas, vth_means[:,:,2])

plot_errors(ax10, t_refs, vtr_means[:,:,0])
plot_errors(ax11, t_refs, vtr_means[:,:,1])
plot_errors(ax12, t_refs, vtr_means[:,:,2])

ax2.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(0.54, 0.70), loc='center', framealpha=0)

for ax in [ax1, ax2, ax3]:
    ax.plot([1.5, 1.5], [0, 0.25], color='black', lw=0.8)

for ax in [ax4, ax5, ax6]:
    ax.plot([20, 20], [0, 0.25], color='black', lw=0.8)

for ax in [ax7, ax8, ax9]:
    ax.plot([20, 20], [0, 0.25], color='black', lw=0.8)

for ax in [ax10, ax11, ax12]:
    ax.plot([2, 2], [0, 0.25], color='black', lw=0.8)


fig.text(0.03, 0.74, 'shifted $t_d$', fontsize=8, rotation=90)
fig.text(0.03, 0.585, 'shifted $\\tau_m$', fontsize=8, rotation=90)
fig.text(0.03, 0.44, 'shifted $\\theta$', fontsize=8, rotation=90)
fig.text(0.03, 0.28, 'shifted $t_{ref}$', fontsize=8, rotation=90)

h = [0.07, 0.36, 0.65]
v = 0.84
fig.text(h[0], v, 'A', fontsize=12)
fig.text(h[1], v, 'B', fontsize=12)
fig.text(h[2], v, 'C', fontsize=12)

v = 0.67
fig.text(h[0], v, 'D', fontsize=12)
fig.text(h[1], v, 'E', fontsize=12)
fig.text(h[2], v, 'F', fontsize=12)

v = 0.525
fig.text(h[0], v, 'G', fontsize=12)
fig.text(h[1], v, 'H', fontsize=12)
fig.text(h[2], v, 'I', fontsize=12)

v = 0.37
fig.text(h[0], v, 'J', fontsize=12)
fig.text(h[1], v, 'K', fontsize=12)
fig.text(h[2], v, 'L', fontsize=12)

# fig.savefig(os.path.join(fig_dir, 'Fig13.pdf'), bbox_inches='tight')
# fig.savefig(os.path.join(fig_dir, 'Fig13.eps'), bbox_inches='tight')
plt.show()
