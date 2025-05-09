# Neuromorphic Attitude Estimation and Control
This repository contains (links to) all code that is used in:\
S. Stroobants, C. De Wagter and G. C. H. E. de Croon, "Neuromorphic Attitude Estimation and Control," in IEEE Robotics and Automation Letters, vol. 10, no. 5, pp. 4858-4865, May 2025, doi: 10.1109/LRA.2025.3553418.
```
@ARTICLE{stroobants2025,
  author={Stroobants, S. and De Wagter, C. and de Croon, G. C. H. E.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Neuromorphic Attitude Estimation and Control}, 
  year={2025},
  volume={10},
  number={5},
  pages={4858-4865},
  keywords={Neuromorphics;Quadrotors;Neurons;Attitude control;Training;Estimation;Robot sensing systems;Hardware;Artificial intelligence;Motors;Imitation learning;neurorobotics;machine learning for robot control},
  doi={10.1109/LRA.2025.3553418}}
```

### Installation on WSL
#### Clone repository
```
git clone git@github.com:tudelft/neuromorphic_att_est_and_control.git
cd neuromorphic_att_est_and_control
```

#### make sure to add submodules:
```
git submodule init
git submodule update
```

#### Create a conda environment with requirements
```
sudo apt install libblas3 libomp5 liblapack3
conda create -n cf_snn python=3.10 -y
conda activate cf_snn
conda install matplotlib pandas tqdm
pip install wandb
pip install torch-directml
```

#### Install python package locally (and submodule)
```
pip install -e .
cd spiking
pip install -e .
```

#### Get the data from the 4TU repository
First create necessary folders:
```
mkdir -p data/datasets/Train
mkdir -p data/datasets/Test
mkdir -p data/datasets/Validation
```

Data can be found at:
https://data.4tu.nl/datasets/f474ef0a-6ef1-4ea1-a958-4827c4eadf60
Download and unzip in the `data` folder. Make sure all three folders have at least 1 dataset.


#### Run example of SNN supervised BPTT
Change the wandb parameters in the yaml file (`bp_onelayer_snn_control_crazyflie_control_from_att.yaml` is default).
In the config file you can select the input / output columns and a linear scaling for each.  
If logging in yaml is set to true, wandb will be used (be sure to initialize a project and insert your username).   
If plot_results is set to true, it will plot the output vs target for one sequence in the last batch  

```
python crazyflie_snn/tests/backprop_snn.py
```

### Other repositories
The code for running the SNN on a Teensy and communicating with the Crazyflie via UART can be found in:
https://github.com/sstroobants/tinysnn

The modified firmware for the Crazyflie 2.0 to communicate with the Teensy and incorporating the control commands in the control pipeline can be found in:
https://github.com/sstroobants/crazyflie-firmware/tree/teensy_fullcontrol

The functions that were used to perform all real-world flight tests can be found in:
https://github.com/sstroobants/crazyflie-lib-python/tree/snn_test/examples/mocap

### Contact us if you have questions
