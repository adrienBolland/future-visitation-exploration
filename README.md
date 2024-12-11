## Off-Policy Maximum Entropy RL with Future State and Action Visitation Measures

This repository contains experiments to compare different Maximum Entropy RL exploration frameworks.
The experiments consist in optimizing policies with different entropy regularization and compare the expected return and state visitation entropy. 

Access the relative paper @ https://arxiv.org/abs/2412.06655

Experiments can be launched providing a configuration file as follows

> python exp_sac_minigrid.py --config_file «file path» --arguments «additional arguments completing the config file» --seed «random seed»

All experiments performed in the paper can be launched in _slurm_ as follows

> bash exp_launcher