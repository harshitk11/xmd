import argparse
import datetime
import torch
from utils import Config
from create_dataset import dataset_generator_downloader, dataset_split_generator
import os
import shutil

def get_args(xmd_base_folder):
    """
    Reads the config file and returns the config parameters.
    params:
        - xmd_base_folder: Location of xmd's base folder
    Output:

    """
    parser = argparse.ArgumentParser(description="XMD : Late-stage fusion.")
    # Location of the default and the update config files.
    parser.add_argument('-config_default', type=str, dest='config_default', default=os.path.join(xmd_base_folder,'config','default_config.yaml'), help="File containing the default experimental parameters.")
    parser.add_argument('-config', type=str, dest='config', default=os.path.join(xmd_base_folder,'config','update_config.yaml'), help="File containing the experimental parameters to update.")
    opt = parser.parse_args()

    # Create a config object. Initialize the default config parameters from the default config file.
    cfg = Config(file=opt.config_default)

    # Update the default config parameters with the parameters present in the update config file.
    cfg.update(updatefile=opt.config)

    # Get the config parameters in args [args is an easydict]
    args = cfg.get_config()
    args['default_config_file'] = opt.config_default
    args['update_config_file'] = opt.config

    # Timestamp for timestamping the logs of a run
    timestamp = str(datetime.datetime.now()).replace(':', '-').replace(' ', '_')
    args.timestamp = timestamp

    return cfg, args

def get_dataset_type_and_partition_dist(dataset_type):
    """
    Returns the dataset type and the partition dist for the dataset_split_generator.
    params:
        - dataset_type : Type of dataset. Can take one of the following values: {"std-dataset","bench-dataset","cd-dataset",
                                                                                "cdyear1-dataset","cdyear2-dataset","cdyear3-dataset"}
                                                                                
    Output:
        - dsGen_dataset_type : Type of dataset. Can take one of the following values: {"std-dataset","bench-dataset","cd-dataset"}
        - dsGem_partition_dist : Split of the partition [num_train_%, num_trainSG_%, num_test_%]
    """
    if dataset_type == "std-dataset":
        # This dataset is used for training the base-classifiers and the second stage model
        dsGen_dataset_type = "std-dataset"
        dsGem_partition_dist = [0.70,0.30,0]

    elif dataset_type == "bench-dataset":
        # This dataset is used for training and testing the base-classifiers. The goal of the dataset is to establish the non-determinism in the GLOBL channels.
        dsGen_dataset_type = "bench-dataset"
        dsGem_partition_dist = [0.70,0,0.30]

    elif (dataset_type == "cd-dataset") or (dataset_type == "cdyear1-dataset") or (dataset_type == "cdyear2-dataset") or (dataset_type == "cdyear3-dataset"):
        # This dataset is used for training and testing the base-classifiers. The goal of the dataset is to establish the non-determinism in the GLOBL channels.
        dsGen_dataset_type = "cd-dataset"
        dsGem_partition_dist = [0,0,1]
    
    else:
        raise ValueError("[main_worker] Incorrect dataset type specified for the dataset split generator.")

    return dsGen_dataset_type, dsGem_partition_dist

def main_worker(args):
    """
    Worker node that performs the complete analysis.
    params:
        - args: easydict storing the experimental parameters
    """
    # Download the dataset if it has not been downloaded
    if args.dataset_download_flag:
        dataset_generator_instance = dataset_generator_downloader(filter_values= [args.runtime_per_file, args.num_logcat_lines_per_file, args.freq_logcat_event_per_file], 
                                                                    dataset_type=args.dataset_type)
        dataset_generator_instance.generate_dataset(download_file_flag=args.dataset_download_flag)
    
    # Get the dataset type and the partition dist for the dataset split generator
    dsGen_dataset_type, dsGem_partition_dist = get_dataset_type_and_partition_dist(dataset_type = args.dataset_type)
    # Get the dataset base location
    dataset_base_location = os.path.join(args.dataset_base_location, args.dataset_type)

    # Generate the dataset splits
    dsGen = dataset_split_generator(seed=args.seed, 
                                    partition_dist=dsGem_partition_dist, 
                                    datasplit_dataset_type=dsGen_dataset_type)
    x = dsGen.create_all_datasets(base_location=dataset_base_location)

    # Generate the dataloaders for all the classification tasks
    
    
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

    # Start the analysis
    main_worker(args=args)

if __name__ == '__main__':
    main()