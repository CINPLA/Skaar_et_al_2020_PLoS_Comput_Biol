'''Sets up simulation directories and parameters for NEST simulations
 including LFP approximations'''
import os
import parameters as ps
import numpy as np
from nest_parameters import get_unique_id, NEST_PSET

if __name__ == '__main__':
    ## Add the random varying parameters
    PSET = ps.ParameterSpace(NEST_PSET)
    PSET['eta'] = ps.ParameterRange([0.8, 0.9, 1.0, 1.1, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0])
    PSET['g'] = ps.ParameterRange([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    PSET['J'] = ps.ParameterRange([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    PSET['simtime'] = 30500.


    # set up directory structure
    savefolder = os.path.join('./heat_plot_simulations/')
    parameterset_dest = os.path.join(savefolder, 'parameters')
    log_dir = os.path.join(savefolder, 'logs')
    nest_jobscript_dest = os.path.join(savefolder, 'nest_jobs')
    nest_output = os.path.join(savefolder, 'nest_output')


    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)
    if not os.path.isdir(parameterset_dest):
        os.mkdir(parameterset_dest)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.isdir(nest_output):
        os.mkdir(nest_output)

    print('Start parameter iteration')
    for i, paramset in enumerate(PSET.iter_inner()):
        # unique id for each parameter set, constructed from the parameset dict
        # converted to a sorted list of tuples
        paramset = paramset.as_dict()
        paramset.update({'nest_seed': paramset['nest_seed'] + i})
        paramset.update({'numpy_seed': paramset['numpy_seed'] + i})
        paramset.update({'random_seed': paramset['random_seed'] + i})

        ps_id = get_unique_id(paramset)
        print(ps_id)

        ## Add parameters to string listing all process IDs by parameters
        with open(os.path.join(savefolder, 'id_parameters.txt'), 'a') as f:
            f.write(ps_id + '\n')
            f.write('%.3f, %.3f, %.3f'%(paramset['eta'], paramset['g'], paramset['J']) + '\n')

        # put output_path into dictionary, as we now have a unique ID of
        # though this will not affect the parameter space object PS
        spike_output_path = os.path.join(nest_output, ps_id)
        if not os.path.isdir(spike_output_path):
            os.mkdir(spike_output_path)

        paramset.update({
            'ps_id': ps_id,
            'spike_output_path': spike_output_path,
            'savefolder': savefolder
                        })

        # write using ps.ParemeterSet native format
        parameterset_file = os.path.join(parameterset_dest, '{}.pset'.format(ps_id))
        ps.ParameterSet(paramset).save(url=parameterset_file)
        # specify where to save output and errors
        nest_output_file = os.path.join(log_dir, ps_id + '.txt')
