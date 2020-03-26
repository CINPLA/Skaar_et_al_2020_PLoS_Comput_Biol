import numpy as np
from nest_parameters import NEST_PSET

HYBRID_PSET = NEST_PSET.copy()
HYBRID_PSET.update(dict(
    #no cell type specificity within each E-I population
    #hence X == x and Y == X
    X = ["EX", "IN"],

    #population-specific LFPy.Cell parameters
    cellParams = dict(
        #excitory cells
        EX = dict(
            morphology = './morphologies/L4E_53rpy1_cut.hoc',
            v_init = HYBRID_PSET['E_L'],
            passive_parameters = dict(g_pas=1/(HYBRID_PSET['tauMem'] * 1E3),
                                      e_pas = HYBRID_PSET['E_L']),
            cm = 1.0,
            Ra = 150,
            nsegs_method = 'lambda_f',
            lambda_f = 100,
            tstart = 0,
            tstop = NEST_PSET['simtime'],
            verbose = False,
            dt = 0.1,
            passive=True
        ),
        #inhibitory cells
        IN = dict(
            morphology = './morphologies/L4I_oi26rbc1.hoc',
            v_init = HYBRID_PSET['E_L'],
            passive_parameters = dict(g_pas=1/(HYBRID_PSET['tauMem'] * 1E3),
                                      e_pas = HYBRID_PSET['E_L']),
            cm = 1.0,
            Ra = 150,
            nsegs_method = 'lambda_f',
            lambda_f = 100,
            tstart = 0,
            tstop = NEST_PSET['simtime'],
            verbose = False,
            dt = 0.1,
            passive=True
    )),

    #assuming excitatory cells are pyramidal
    rand_rot_axis = dict(
        EX = ['z'],
        IN = ['x', 'y', 'z'],
    ),

    #kwargs passed to LFPy.Cell.simulate()
    simulationParams = dict(),

    #set up parameters corresponding to cylindrical model populations
    populationParams = dict(
        EX = dict(
            number = HYBRID_PSET['NE'],
            radius = np.sqrt(1000**2 / np.pi),
            z_min = -450,
            z_max = -350,
            min_cell_interdist = 1.,
            ),
        IN = dict(
            number = HYBRID_PSET['NI'],
            radius = np.sqrt(1000**2 / np.pi),
            z_min = -450,
            z_max = -350,
            min_cell_interdist = 1.,
            ),
    ),

    #set the boundaries between the "upper" and "lower" layer
    layerBoundaries = [[0., -300],
                       [-300, -500]],

    #set the geometry of the virtual recording device
    electrodeParams = dict(
            #contact locations:
            x = [0]*6,
            y = [0]*6,
            z = [x*-100. for x in range(6)],
            #extracellular conductivity:
            sigma = 0.3,
            #contact surface normals, radius, n-point averaging
            N = [[1, 0, 0]]*6,
            r = 5,
            n = 20,
            seedvalue = None,
            #dendrite line sources, soma as sphere source (Linden2014)
            method = 'soma_as_point',
            #no somas within the constraints of the "electrode shank":
            r_z = [[-1E199, -600, -550, 1E99],[0, 0, 10, 10]],
    ),

    #runtime, cell-specific attributes and output that will be stored
    savelist = [
        'somapos',
        'x',
        'y',
        'z',
        'LFP'
    ],
    pp_savelist = ['LFP'],
    plots=False,
    #flag for switching on calculation of CSD
    calculateCSD = False,

    #time resolution of saved signals
    dt_output = 1.
))


#for each population, define layer- and population-specific connectivity
#parameters
HYBRID_PSET.update(dict(
    #number of connections from each presynaptic population onto each
    #layer per postsynaptic population, preserving overall indegree
    k_yXL = dict(
        EX = [[int(HYBRID_PSET['CE']*0.5), 0],
              [int(HYBRID_PSET['CE']*0.5), HYBRID_PSET['CI']]],
        IN = [[0, 0],
              [HYBRID_PSET['CE'], HYBRID_PSET['CI']]],
    ),

    #set up synapse parameters as derived from the network
    synParams = dict(
        EX = dict(
            section = ['apic', 'dend'],
            syntype = 'AlphaISyn'
        ),
        IN = dict(
            section = ['dend', 'soma'],
            syntype = 'AlphaISyn'
        ),
    ),

    #set up table of synapse time constants from each presynaptic populations
    tau_yX = dict(
        EX = [5.0, 5.0],
        IN = [5.0, 5.0]
    ),
    #set up delays, here using fixed delays of network
    synDelayLoc = dict(
        EX = [HYBRID_PSET['delay'], HYBRID_PSET['delay']],
        IN = [HYBRID_PSET['delay'], HYBRID_PSET['delay']],
    ),
    #no distribution of delays
    synDelayScale = dict(
        EX = [None, None],
        IN = [None, None],
    ),
))

#putative mappting between population type and cell type specificity,
#but here all presynaptic senders are also postsynaptic targets
HYBRID_PSET.update(dict(
    mapping_Yy = list(zip(HYBRID_PSET['X'], HYBRID_PSET['X']))
))

J_ex = 7.357e-12 * 1e9     ## set J = g = 1, C = 7.357e-12, convert to nA
J_in = -7.357e-12 * 1e9    ##

HYBRID_PSET.update(dict(J_yX = dict(
               EX = [J_ex, J_in],
               IN = [J_ex, J_in],
               ))),
