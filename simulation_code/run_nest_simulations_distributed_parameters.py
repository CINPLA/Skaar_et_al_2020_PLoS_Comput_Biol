'''
Runs NEST simulations and predicts LFPs with parameters defined by
scripts set_up_parameters....py
'''
import os, sys, io, time
from mpi4py import MPI
from io import BytesIO
import h5py
import nest
from parameters import ParameterSet
import numpy as np
from numpy import exp
import random
from scipy.stats import truncnorm

top_dir = sys.argv[1]
param_dir = os.path.join(top_dir, 'parameters')
log_dir = os.path.join(top_dir, 'logs')


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def simulate():
    '''
    Runs a single simulation, parameters taken from upper scope
    '''

    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': PSET.dt, 'print_time': True,
                          'overwrite_files': True, 'grng_seed': PSET.nest_seed})
    np.random.seed(PSET.numpy_seed)
    random.seed(PSET.random_seed)


    print('Building network')

    ## Set parameters for neurons and poisson generator
    nest.SetDefaults('iaf_psc_delta', neuron_params)
    nest.SetDefaults('poisson_generator', {'rate': p_rate})

    ## Create all neurons and recorders
    # local populations
    nodes_ex = nest.Create('iaf_psc_delta', PSET.NE)
    nodes_in = nest.Create('iaf_psc_delta', PSET.NI)

    # set distributed neuron parameters
    if PSET.tauMem_gaussian:
        lower, upper = 2, 38
        mu, sigma = PSET.tauMem, PSET.tauMem*PSET.sigma_factor
        tn = truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)
        ex_tau_m = tn.rvs(len(nodes_ex))
        in_tau_m = tn.rvs(len(nodes_in))
        nest.SetStatus(nodes_ex, 'tau_m', ex_tau_m)
        nest.SetStatus(nodes_in, 'tau_m', in_tau_m)

    if PSET.t_ref_gaussian:
        lower, upper = 0.2, 3.8
        mu, sigma = PSET.t_ref, PSET.t_ref*PSET.sigma_factor
        tn = truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)
        ex_t_ref = np.around(tn.rvs(len(nodes_ex)), 1)
        in_t_ref = np.around(tn.rvs(len(nodes_in)), 1)
        nest.SetStatus(nodes_ex, 't_ref', ex_t_ref)
        nest.SetStatus(nodes_in, 't_ref', in_t_ref)

    if PSET.theta_gaussian:
        lower, upper = 12, 28
        mu, sigma = PSET.theta, PSET.theta*PSET.sigma_factor
        tn = truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)
        ex_theta = tn.rvs(len(nodes_ex))
        in_theta = tn.rvs(len(nodes_in))
        nest.SetStatus(nodes_ex, 'V_th', ex_theta)
        nest.SetStatus(nodes_in, 'V_th', in_theta)


    # external population
    noise = nest.Create('poisson_generator')

    # spike recorders
    espikes = nest.Create('spike_detector')
    ispikes = nest.Create('spike_detector')
    print('first exc node: {}'.format(nodes_ex[0]))
    print('first inh node: {}'.format(nodes_in[0]))

    ## Set initial membrane voltages to random values between 0 and threshold
    nest.SetStatus(nodes_ex, 'V_m',
                   np.random.rand(len(nodes_ex)) * neuron_params['V_th'])
    nest.SetStatus(nodes_in, 'V_m',
                   np.random.rand(len(nodes_in)) * neuron_params['V_th'])

    nest.CopyModel("static_synapse", "excitatory",
                   {"weight": PSET.J, 'delay': PSET.delay})

    syn_spec_ex = {'model': 'static_synapse'}
    syn_spec_in = {'model': 'static_synapse'}

    if PSET.J_gaussian:
        syn_spec_ex.update({'weight':  {'distribution': 'normal_clipped',
                                        'mu': PSET.J,
                                        'sigma': PSET.J*PSET.sigma_factor,
                                        'low': 0.0,
                                        'high': 2*PSET.J
                                        }})

        syn_spec_in.update({'weight':  {'distribution': 'normal_clipped',
                                        'mu': -PSET.J*PSET.g,
                                        'sigma': PSET.J*PSET.g*PSET.sigma_factor,
                                        'high': 0.,
                                        'low': -2*PSET.J*PSET.g
                                        }})
    else:
        syn_spec_ex.update({'weight': PSET.J})
        syn_spec_in.update({'weight': -PSET.J*PSET.g})

    if PSET.delay_gaussian:
        syn_spec_ex.update({'delay': {'distribution': 'normal_clipped',
                                      'mu': PSET.delay,
                                      'sigma': PSET.delay*PSET.sigma_factor,
                                      'low': 0.2,
                                      'high': 2.8
                                       }})

        syn_spec_in.update({'delay': {'distribution': 'normal_clipped',
                                      'mu': PSET.delay,
                                      'sigma': PSET.delay*PSET.sigma_factor,
                                      'low': 0.2,
                                      'high': 2.8
                                       }})
    else:
        syn_spec_in.update({'delay': 1.5})
        syn_spec_ex.update({'delay': 1.5})




    # Connect 'external population' poisson generator to local neurons
    nest.Connect(noise, nodes_ex, 'all_to_all', 'excitatory')
    nest.Connect(noise, nodes_in, 'all_to_all', 'excitatory')

    ## Record spikes to be saved from a subset of each population
    nest.Connect(nodes_ex, espikes, 'all_to_all', 'excitatory')
    nest.Connect(nodes_in, ispikes, 'all_to_all', 'excitatory')


    ## Connect local populations
    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': PSET.CE}
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, syn_spec_ex)
    conn_params_in = {'rule': 'fixed_indegree', 'indegree': PSET.CI}
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, syn_spec_in)

    endbuild = time.time()

    nest.Simulate(PSET.simtime)

    endsimulate = time.time()

    ## Calculate firing rate
    events_ex = nest.GetStatus(espikes, 'n_events')[0]
    events_in = nest.GetStatus(ispikes, 'n_events')[0]
    rate_ex = events_ex / PSET.simtime * 1000.0 / PSET.N_neurons
    rate_in = events_in / PSET.simtime * 1000.0 / PSET.N_neurons

    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild

    print('Build time: %.1f'%build_time)
    print('Sim time: %.1f'%sim_time)
    print('Excitatory firing rate: %.1f'%rate_ex)
    print('Inhibitory firing rate: %.1f'%rate_in)

    ## Get sample spikes for saving
    ex_events = nest.GetStatus(espikes, 'events')[0]
    in_events = nest.GetStatus(ispikes, 'events')[0]
    # ex_spikes = np.stack((ex_events['senders'], ex_events['times'])).T
    # in_spikes = np.stack((in_events['senders'], in_events['times'])).T

    ## Get population firing histograms for calculating LFP
    ex_hist_times = ex_events['times']
    in_hist_times = in_events['times']
    hist_bins_1 = np.arange(PSET.simtime + 2, dtype=np.float32) - 0.5
    ex_hist, _ = np.histogram(ex_hist_times, bins=hist_bins_1)
    in_hist, _ = np.histogram(in_hist_times, bins=hist_bins_1)

    with h5py.File(output_file, 'w') as f:
        if PSET.predict_LFP:
            d_diff = int(PSET['delay'] - 1.5) # to shift kernel with delay
            LFP = np.empty([6, len(ex_hist)])
            for i in range(6):
                LFP[i] = np.convolve(ex_hist, ex_kernel[i]*PSET.J, mode='same') \
                       + np.convolve(in_hist, in_kernel[i]*PSET.g*PSET.J, mode='same')
        dset = f.create_dataset('data', data=LFP)
        # f.create_dataset('ex_spikes', data=ex_spikes)
        # f.create_dataset('in_spikes', data=in_spikes)
        f.create_dataset('ex_hist', data=ex_hist)
        f.create_dataset('in_hist', data=in_hist)
    print('Complete')





## divvy up simulations
sim_list = os.listdir(os.path.join(top_dir, 'nest_output'))
sim_list.sort()
uncompleted_sim_list = []
for sim in sim_list:
    if not os.path.isfile(os.path.join(top_dir, 'nest_output', sim, 'LFP_firing_rate.h5')):
        uncompleted_sim_list.append(sim)
sim_list = uncompleted_sim_list
if rank == 0:
    print('%d simulations in total'%len(sim_list))
sims_per_rank = len(sim_list) // size
remainder = len(sim_list) % size
local_sim_indices = [sims_per_rank*rank + i for i in range(sims_per_rank)]
local_ids = [sim_list[i] for i in local_sim_indices]
if rank < remainder:
    local_sim_indices.append(sims_per_rank*size+rank)
    local_ids.append(sim_list[-(rank+1)])

ex_kernel = None
in_kernel = None

## Start iterating through simulations
for j, i in enumerate(local_ids):
    print('Starting sim ', local_sim_indices[j])
    ## Create separate log for each simulation
    logstream = io.StringIO()
    logstream.seek(0)
    sys.stdout = logstream

    ## Load parameters from file
    PSET = ParameterSet(os.path.join(param_dir, i + '.pset'))
    if ex_kernel is None and PSET.predict_LFP is True:
        with h5py.File(PSET.LFP_kernel_path, 'r') as f:
            ex_kernel = f['ex'][:]
            in_kernel = f['in'][:]

    output_file = os.path.join(PSET.savefolder, 'nest_output', PSET.ps_id, 'LFP_firing_rate.h5')

    if os.path.isfile(output_file):
        sys.stdout = sys.__stdout__
        continue
    startbuild = time.time()

    ## Calculate rate for external population
    nu_th  = PSET.theta/(PSET.J*PSET.CE*PSET.tauMem)
    nu_ex  = PSET.eta*nu_th
    p_rate = 1000.0*nu_ex*PSET.CE


    ## Parameters for 'iaf_psc_delta' cell model
    neuron_params = {'C_m': PSET.CMem,
                     'tau_m': PSET.tauMem,
                     't_ref': PSET.t_ref,
                     'E_L': PSET.E_L,
                     'V_reset': PSET.V_reset,
                     'V_m': PSET.V_m,
                     'V_th': PSET.theta,
                     'I_e': 0., }

    label = 'brunel-py'

    ## Run simulation
    simulate()

    ## Write log file
    logstream.seek(0)
    with open(os.path.join(log_dir, PSET.ps_id + '.txt'), 'w') as f:
        f.write(logstream.getvalue())
    sys.stdout = sys.__stdout__
    print('Sim %d complete'%local_sim_indices[j])
    break
