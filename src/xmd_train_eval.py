"""
Python script to train and evaluate all the first stage and the second stage models of XMD.
"""
import argparse
import datetime
import stat
import torch
from utils import Config
from create_raw_dataset import dataset_generator_downloader, dataset_split_generator, custom_collator, get_dataloader
import os
import shutil
import numpy as np
import pickle
import collections
import json
from sklearn.model_selection import train_test_split

def main():
    ############################################## Setting up the experimental parameters ##############################################
    # Location of the base folder of xmd 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_folder_location = os.path.join(dir_path.replace("/src",""),"")

    # Get the arguments and the config file object
    cfg, args = get_args(xmd_base_folder=base_folder_location)

    # Create a runs directory for this run and push the config files for this run in the directory
    args.run_dir = os.path.join(base_folder_location, 'runs', args.timestamp)
    if args.create_runs_dir and os.path.isdir(args.run_dir) is False:
        os.mkdir(args.run_dir)
        shutil.copyfile(args.default_config_file, os.path.join(args.run_dir, args.default_config_file.split('/')[-1]))
        shutil.copyfile(args.update_config_file, os.path.join(args.run_dir, args.update_config_file.split('/')[-1]))

        # Write the updated config file in final_config.yaml
        cfg.export_config(os.path.join(args.run_dir, 'final_config.yaml'))
    ####################################################################################################################################

    # Duration of the time series that will be used
    # args.truncated_duration = 20
    # Start the analysis
    main_worker(args=args, xmd_base_folder_location= base_folder_location)

if __name__ == '__main__':
    main()