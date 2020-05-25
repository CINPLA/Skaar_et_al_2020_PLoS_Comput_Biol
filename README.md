# Skaar_et_al_2020_PLoS_Comput_Biol

### This repository contains all the code used for the article Estimation of neural network model parameters from local field potentials (LFPs)

### Recreating the figures will usually require a set of simulations to be run, a CNN to be trained or used to predict, before executing the figure_*.py script. Running the simulations is done in two steps: First, a simulation directory is set up, where subdirectories for simulation outputs and parameters are created. Second, the simulation scripts are run, given the path to the simulation directory. The simulations should be run on an HPC cluster. Job scripts for Slurm systems can be found in the ./simulation_code/job_scripts directory. Below, the steps to create the results are given.

#### Figure 6
Set up simulation directory by running set_up_parameters_heat_plots.py
Run simulations by submitting run_nest_heat_plots.job
Run figure_6 to create the figure.

#### Figure 7
Set up simulation directories by running set_up_parameters_large_parameterspace.py and set_up_parameters_small_parameterspace.py
Run the simulations by submitting run_nest_large_parameterspace.job and run_nest_small_parameterspace.job
Train the CNNs by running train_large_parameterspace.py and train_small_parameterspace.py
Run figure_7.py to create the figure

#### Figure 8
Uses the same simulations / networks as Figure 7
Run figure_8.py to create the figure

#### Figure 9
Uses same large parameter space simulations and network from Figure 7
Run figure_9.py to create the figure

#### Figure 10
Set up simulation directory by running set_up_parameters_grid.py
Run simulations by submitting run_nest_grid_parameterspace.py
Train network by running train_grid_vs_random.py
Run figure_10.py to create the figure.

#### Figure 11 and 12
Set up simulation directories by running set_up_parameters_varying_*.py and set_up_parameters_gaussian_*.py
Run simulations by submitting run_nest_varying_*.py and run_nest_distributed_*.py
Run figure_11_12.py to create figures.
