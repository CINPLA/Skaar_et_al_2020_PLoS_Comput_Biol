import parameters as ps
import time, operator, pickle, hashlib

## Define parameters
NEST_PSET = dict(
    LFP_kernel_path = './kernel_simulation/L4E_53rpy1_cut_L4I_oi26rbc1_kernel.h5',
    predict_LFP = True,
    label ='brunel',
    save_spikes=False,
    dt=0.1,         # Simulation time resolution in ms
    simtime=400.,  # Simulation time in ms
    nest_seed=int(time.time()), # base for seeds, will be updated for each
                                # individual parameterset
    numpy_seed=int(time.time())//2,
    random_seed=int(time.time())//3,
    g=1.0,          # ratio inhibitory weight/excitatory weight
    eta=1.0,        # external rate relative to threshold rate
    epsilon=0.1,    # connection probability (before: 0.1)
    CMem=250.0,       # capacitance of membrane in in pF
    theta=20.0,     # membrane threshold potential in mV
    V_reset=10.0,    # reset potential of membrane in mV
    E_L=0.0,        # resting membrane potential in mV
    V_m=0.0,        # membrane potential in mV

    tauMem=20.0,    # time constant of membrane potential in ms
    delay=1.5,      # synaptic delay
    t_ref=2.0,      # refractory period

    order=2500,     # network scaling factor
)

NEST_PSET.update(dict(
    NE = 4 * NEST_PSET['order'], # number of excitatory neurons
    NI = 1 * NEST_PSET['order']  # number of inhibitory neurons
))

NEST_PSET.update(dict(
    N_neurons = NEST_PSET['NE'] + NEST_PSET['NI'], # total number of neurons
    CE = int(NEST_PSET['NE'] * NEST_PSET['epsilon']), # number of excitatory synapses per neuron
    CI = int(NEST_PSET['NI'] * NEST_PSET['epsilon'])  # number of inhibitory synapses per neuron
))

NEST_PSET.update(dict(
    C_tot = NEST_PSET['CE'] + NEST_PSET['CI']  # total number of synapses per neuron
))


def sort_deep_dict(d):
    '''
    sort arbitrarily deep dictionaries into tuples

    Arguments
    ---------
    d : dict

    Returns:
    x : list of tuples of tuples of tuples ...
    '''
    x = sorted(iter(d.items()), key=operator.itemgetter(0))
    for i, (key, value) in enumerate(x):
        if type(value) == dict or type(value) == ps.ParameterSet:
            y = sorted(iter(value.items()), key=operator.itemgetter(0))
            x[i] = (key, y)
            for j, (k, v) in enumerate(y):
                if type(v) == dict or type(v) == ps.ParameterSet:
                    y[j] = (k, sort_deep_dict(v))
    return x


def get_unique_id(paramset):
    '''
    create a unique hash key for input dictionary

    Arguments
    ---------
    paramset : dict
        parameter dictionary

    Returns
    -------
    key : str
        hash key

    '''
    sorted_params = sort_deep_dict(paramset)
    string = pickle.dumps(sorted_params)
    key = hashlib.md5(string).hexdigest()
    return key
