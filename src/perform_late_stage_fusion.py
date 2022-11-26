import argparse
import datetime
import stat
import torch
from utils import Config
from create_dataset import dataset_generator_downloader, dataset_split_generator, custom_collator, get_dataloader
import os
import shutil
import numpy as np
import pickle
import collections


def makehash():
    return collections.defaultdict(makehash)

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

class dataloader_generator:
    """
    Contains all the helper methods for generating the dataloader for all the classification tasks.
    """
    # Used to identify the partitions required for the different classification tasks for the different datasets : std-dataset, cd-dataset, bench-dataset
    partition_activation_flags = {
                                    "std-dataset":{'DVFS_partition_for_HPC_DVFS_fusion':{"train":False, "trainSG":True, "test":False},
                                                'HPC_partition_for_HPC_DVFS_fusion':{"train":False, "trainSG":True, "test":False},
                                                'HPC_individual':{"train":True, "trainSG":False, "test":False},
                                                'DVFS_individual':{"train":True, "trainSG":False, "test":False},
                                                'DVFS_fusion':{"train":False, "trainSG":False, "test":False}},
                                    "cd-dataset":{'DVFS_partition_for_HPC_DVFS_fusion':{"train":False, "trainSG":False, "test":True},
                                                'HPC_partition_for_HPC_DVFS_fusion':{"train":False, "trainSG":False, "test":True},
                                                'HPC_individual':{"train":False, "trainSG":False, "test":True},
                                                'DVFS_individual':{"train":False, "trainSG":False, "test":True},
                                                'DVFS_fusion':{"train":False, "trainSG":False, "test":True}},
                                    "bench-dataset":{'DVFS_partition_for_HPC_DVFS_fusion':{"train":False, "trainSG":False, "test":False},
                                                    'HPC_partition_for_HPC_DVFS_fusion':{"train":False, "trainSG":False, "test":False},
                                                    'HPC_individual':{"train":True, "trainSG":False, "test":True},
                                                    'DVFS_individual':{"train":True, "trainSG":False, "test":True},
                                                    'DVFS_fusion':{"train":False, "trainSG":False, "test":True}}
                                    }

    @staticmethod
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

    @staticmethod
    def prepare_dataloader(partition_dict, labels, file_type, dataset_type, clf_toi, args):
        """ 
        Configure the dataloader. Based on the dataset type and the classification task of interest,
        this will return dataloaders for the tasks: "train","trainSG","test".

        params: 
            - partition_dict : {'train' : [file_path1, file_path2, ..],
                            'trainSG' : [file_path1, file_path2, ..],
                            'test' : [file_path1, file_path2]}

            - labels : {file_path1 : 1, file_path2: 0, ...}
            - file_type : 'dvfs' or 'simpleperf'
            - dataset_type : Type of dataset. Can take one of the following values: {"std-dataset","bench-dataset","cd-dataset"}
            - clf_toi : Classification task of interest. Can take one of the following values:
                    ["DVFS_partition_for_HPC_DVFS_fusion", "HPC_partition_for_HPC_DVFS_fusion", "HPC_individual", "DVFS_individual", "DVFS_fusion"]
            - args :  arguments from the config file

        Output: 
            - train, trainSG, and test dataloader objects depending on the dataset_type
        """

        print(f'[Info] Fetching dataloader objects for dataset:{dataset_type}, classification toi: {clf_toi}, channel: {file_type}\n{"-"*140}')
    
        # Find the partitions that will be required for this dataset. required_partitions = {"train":T or F, "trainSG":T or F, "test":T or F}
        required_partitions = dataloader_generator.partition_activation_flags[dataset_type][clf_toi]
        
        # Intitialize the custom collator
        custom_collate_fn = custom_collator(args=args,
                                            file_type=file_type)
 
        # Normalize flag set to True for file_type = 'dvfs' and False for file_type = 'simpleperf'
        if file_type=='dvfs':
            normalize_flag = True
        elif file_type=='simpleperf':
            normalize_flag = False
        else:
            raise ValueError("Incorrect file_type passed to wrapper for dataloader")

        # Get the dataloader object : # get_dataloader() returns an object that is returned by torch.utils.data.DataLoader
        trainloader, trainSGloader, testloader = get_dataloader(args, 
                                                            partition = partition_dict, 
                                                            labels = labels, 
                                                            custom_collate_fn =custom_collate_fn,
                                                            required_partitions=required_partitions, 
                                                            normalize_flag=normalize_flag, 
                                                            file_type= file_type, 
                                                            N = None)

        return {'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader}
    
    @staticmethod
    def get_dataloader_for_all_classification_tasks(all_dataset_partitionDict_label, dataset_type, args):
        """
        Generates the dataloader for all the classification tasks.
        NOTE: args.truncated_duration is used to truncate the timeseries. If you want truncated time series, then args.truncated_duration
              needs to be changed before calling this method.

        params:
            - all_dataset_partitionDict_label: 
                                                [
                                                (DVFS_partition_for_HPC_DVFS_fusion, DVFS_partition_labels_for_HPC_DVFS_fusion),
                                                (HPC_partition_for_HPC_DVFS_fusion, HPC_partition_labels_for_HPC_DVFS_fusion),
                                                (HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual),
                                                (DVFS_partition_for_DVFS_individual,DVFS_partition_labels_for_DVFS_individual),
                                                (DVFS_partition_for_DVFS_fusion,DVFS_partition_labels_for_DVFS_fusion)
                                                ]
                                        
                                                partition -> {'train' : [file_path1, file_path2, ..],
                                                                'trainSG' : [file_path1, file_path2, ..],
                                                                'test' : [file_path1, file_path2]}

                                                labels -> {file_path1 : 0, file_path2: 1, ...}        [Benigns have label 0 and Malware have label 1]
            
            - dataset_type: Can take one of the following values {'std-dataset','bench-dataset','cd-dataset'}. 
                            Based on the dataset type we will activate different partitions ("train", "trainSG", "test") for the different classification tasks.

            - args: easydict storing the arguments for the experiment

        Output:
            - dataloaderAllClfToi =
                        {
                            'DVFS_partition_for_HPC_DVFS_fusion':{
                                                                    'rn1':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn2':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn3':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn4':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader}
                                                                    },
                            'HPC_partition_for_HPC_DVFS_fusion':{
                                                                    'rn1':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn2':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn3':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn4':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader}
                                                                    },
                            'HPC_individual':                   {
                                                                    'rn1':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn2':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn3':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader},
                                                                    'rn4':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader}
                                                                    },
                            'DVFS_individual' :                 {
                                                                    'all':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader}
                                                                    },
                            'DVFS_fusion' :                     {
                                                                    'all':{'trainloader': trainloader, 'trainSGloader': trainSGloader, 'testloader': testloader}
                                                                    }
                        }
                    NOTE: If the dataloader for a particular task is not required, then it is None.
        """
        # For selecting the dataset partition, labels, and file_type of the dataset_of_interest in all_datasets
        select_dataset = {
                        'DVFS_partition_for_HPC_DVFS_fusion' : {'rn1':{'partition':all_dataset_partitionDict_label[0][0][0],'label':all_dataset_partitionDict_label[0][1][0],'file_type':'dvfs'},
                                                                'rn2':{'partition':all_dataset_partitionDict_label[0][0][1],'label':all_dataset_partitionDict_label[0][1][1],'file_type':'dvfs'},
                                                                'rn3':{'partition':all_dataset_partitionDict_label[0][0][2],'label':all_dataset_partitionDict_label[0][1][2],'file_type':'dvfs'},
                                                                'rn4':{'partition':all_dataset_partitionDict_label[0][0][3],'label':all_dataset_partitionDict_label[0][1][3],'file_type':'dvfs'}},

                        'HPC_partition_for_HPC_DVFS_fusion' :  {'rn1':{'partition':all_dataset_partitionDict_label[1][0][0],'label':all_dataset_partitionDict_label[1][1][0],'file_type':'simpleperf'},
                                                                'rn2':{'partition':all_dataset_partitionDict_label[1][0][1],'label':all_dataset_partitionDict_label[1][1][1],'file_type':'simpleperf'},
                                                                'rn3':{'partition':all_dataset_partitionDict_label[1][0][2],'label':all_dataset_partitionDict_label[1][1][2],'file_type':'simpleperf'},
                                                                'rn4':{'partition':all_dataset_partitionDict_label[1][0][3],'label':all_dataset_partitionDict_label[1][1][3],'file_type':'simpleperf'}},

                        'HPC_individual' :                     {'rn1':{'partition':all_dataset_partitionDict_label[2][0][0],'label':all_dataset_partitionDict_label[2][1][0],'file_type':'simpleperf'},
                                                                'rn2':{'partition':all_dataset_partitionDict_label[2][0][1],'label':all_dataset_partitionDict_label[2][1][1],'file_type':'simpleperf'},
                                                                'rn3':{'partition':all_dataset_partitionDict_label[2][0][2],'label':all_dataset_partitionDict_label[2][1][2],'file_type':'simpleperf'},
                                                                'rn4':{'partition':all_dataset_partitionDict_label[2][0][3],'label':all_dataset_partitionDict_label[2][1][3],'file_type':'simpleperf'}},

                        'DVFS_individual' :                    {'all':{'partition':all_dataset_partitionDict_label[3][0],'label':all_dataset_partitionDict_label[3][1],'file_type':'dvfs'}},

                        'DVFS_fusion' :                        {'all':{'partition':all_dataset_partitionDict_label[4][0],'label':all_dataset_partitionDict_label[4][1],'file_type':'dvfs'}}
                        }
        
        # Get the dataloader for all the classification toi
        dataloaderAllClfToi = {}
        for clf_toi, partition_bin_details in select_dataset.items():
            dataloaderAllClfToi[clf_toi] = {}
            for rnBin, partitionDetails in partition_bin_details.items(): 
                dataloaderAllClfToi[clf_toi][rnBin] = dataloader_generator.prepare_dataloader(partition_dict = partitionDetails['partition'], 
                                                                                            labels = partitionDetails['label'], 
                                                                                            file_type = partitionDetails['file_type'], 
                                                                                            dataset_type = dataset_type, 
                                                                                            clf_toi = clf_toi, 
                                                                                            args = args)
        
        # ################################################################ Testing the supervised learning dataloader ################################################################
        # iter_loader = iter(dataloaderAllClfToi['DVFS_fusion']['all']['testloader'])
        # batch_spec_tensor, labels, f_paths = next(iter_loader)
        # f_paths = "\n - ".join(f_paths)
        # print(f"- Shape of batch tensor (B,N_ch,reduced_feature_size) : {batch_spec_tensor.shape}")
        # print(f"- Batch labels : {labels}")
        # print(f"- File Paths : {f_paths}")
        # exit()

        # iter_loader = iter(dataloaderAllClfToi['HPC_individual']['rn1']['trainloader'])
        # batch_spec_tensor, labels, f_paths = next(iter_loader)
        # f_paths = "\n - ".join(f_paths)
        # print(f"- Shape of batch tensor (B,N_ch,reduced_feature_size) : {batch_spec_tensor.shape}")
        # print(f"- Batch labels : {labels}")
        # print(f"- File Paths : {f_paths}")
        # exit()

        # # Testing the alignment of DVFS and HPC for HPC-DVFS fusion
        # # HPC
        # iter_testloader_hpc = iter(dataloaderAllClfToi['HPC_partition_for_HPC_DVFS_fusion']['rn2']['trainSGloader'])
        # batch_spec_tensor_hpc, labels_hpc, f_paths_hpc = next(iter_testloader_hpc)
        # # DVFS
        # iter_testloader_dvfs = iter(dataloaderAllClfToi['DVFS_partition_for_HPC_DVFS_fusion']['rn2']['trainSGloader'])
        # batch_spec_tensor_dvfs, labels_dvfs, f_paths_dvfs = next(iter_testloader_dvfs)
        # for i,j in zip(f_paths_dvfs,f_paths_hpc):
        #     print(f"-- {i,j}")
        # exit()
        ###############################################################################################################################################################################
        return dataloaderAllClfToi
        
class feature_engineered_dataset:
    """
    Class containing all the methods for creating the feature engineered datasets for HPC and GLOBL channels.
    """
    def __init__(self, args, all_dataset_partitionDict_label, dataset_type, results_path, fDataset) -> None:
        """
        params:
            - all_dataset_partitionDict_label: 
                                                    [
                                                    (DVFS_partition_for_HPC_DVFS_fusion, DVFS_partition_labels_for_HPC_DVFS_fusion),
                                                    (HPC_partition_for_HPC_DVFS_fusion, HPC_partition_labels_for_HPC_DVFS_fusion),
                                                    (HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual),
                                                    (DVFS_partition_for_DVFS_individual,DVFS_partition_labels_for_DVFS_individual),
                                                    (DVFS_partition_for_DVFS_fusion,DVFS_partition_labels_for_DVFS_fusion)
                                                    ]
                                            
                                                    partition -> {'train' : [file_path1, file_path2, ..],
                                                                    'trainSG' : [file_path1, file_path2, ..],
                                                                    'test' : [file_path1, file_path2]}

                                                    labels -> {file_path1 : 0, file_path2: 1, ...}        [Benigns have label 0 and Malware have label 1]
                
            - dataset_type: Can take one of the following values {'std-dataset','bench-dataset','cd-dataset'}. 
                            Based on the dataset type we will activate different partitions ("train", "trainSG", "test") for the different classification tasks.
            - results_path: Location of the base folder where all the feature engineered datasets are stored.
            - fDataset: Dict storing the paths of all the datasets for all the classification tasks
                        Format: fDataset = {"filter_val":{"runtime":{"clf_toi":{"rn_bin":{"split" :[dataset_path, labels_path, file_path], ... }}}}}
        
        Output:
            - fDataset: Updated fDataset
        """
        self.args = args
        self.all_dataset_partitionDict_label = all_dataset_partitionDict_label
        self.dataset_type = dataset_type
        self.results_path = results_path
        
        # List of candidate truncated durations
        self.rtime_list = [i for i in range(args.step_size_truncated_duration, args.collected_duration+args.step_size_truncated_duration, args.step_size_truncated_duration)]

        # logcat_filter_rtime: Filter runtime used when filtering and downloading the dataset (in s) [Used for naming the output file]
        self.logcat_filter_rtime_threshold = args.runtime_per_file

        # Classification tasks of interest
        self.clf_toi_list = ["DVFS_partition_for_HPC_DVFS_fusion", "HPC_partition_for_HPC_DVFS_fusion", "HPC_individual", "DVFS_individual", "DVFS_fusion"]

        self.fDataset = fDataset
        # Dict storing the paths of all the datasets for all the classification tasks
        if self.logcat_filter_rtime_threshold not in self.fDataset:
            # Add an entry for logcat_filter_rtime_threshold
            self.initialize_fDataset()
        

    def initialize_fDataset(self):
        """
        Creates a new entry for logcat_filter_rtime_threshold.
        fDataset[logcat_filter_rtime_threshold][rtime][clf_toi][rnBin][split] = None
        """    
        self.fDataset[self.logcat_filter_rtime_threshold] = {}
        for rtime in self.rtime_list:
            self.fDataset[self.logcat_filter_rtime_threshold][rtime] = {}
            for clf_toi in self.clf_toi_list:
                self.fDataset[self.logcat_filter_rtime_threshold][rtime][clf_toi] = {}
                for rnBin in ["rn1","rn2","rn3","rn4","all"]:
                    self.fDataset[self.logcat_filter_rtime_threshold][rtime][clf_toi][rnBin] = {}
                    for split in ["train","trainSG","test"]:
                        self.fDataset[self.logcat_filter_rtime_threshold][rtime][clf_toi][rnBin][split] = None

    def generate_feature_engineered_dataset(self):
        """
        Method to generate the feature engineered dataset for all the classification tasks for a given logcat filter-value.
        
        Output:
            Updates fDataset which is a parameter of the class.
        """
        for rtime in self.rtime_list:
            print(f"----------- Generating feature engineered dataset for truncated duration : {rtime} -----------")
            self.generate_feature_engineered_dataset_per_rtime(truncated_duration=rtime)

    def generate_feature_engineered_dataset_per_rtime(self, truncated_duration):
        """
        Method to generate the feature engineered dataset for all the classification tasks for a given filter-value and truncated-duration.
        
        params:
            - truncated_duration: time to which you want to trim the time series (in s)
            
        Output:
            Updates fDataset which is a parameter of the class.
        """
        # Set the truncated duration in the args
        self.args.truncated_duration = truncated_duration
        # Generate the dataloaders for all the classification tasks
        dataloaderAllClfToi = dataloader_generator.get_dataloader_for_all_classification_tasks(all_dataset_partitionDict_label = self.all_dataset_partitionDict_label, 
                                                                                        dataset_type=self.dataset_type, 
                                                                                        args=self.args)

        # For all clf_toi, generate the feature engineered vectors.
        for clf_toi, rn_bin_details in dataloaderAllClfToi.items():
            for rnBin, dataloaderDict in rn_bin_details.items():
                for dataloaderName, dataloaderX in dataloaderDict.items():
                    if dataloaderX is not None:
                        # Generate the feature engineered dataset
                        splitName = dataloaderName.replace("loader","")
                        fpathList = self.create_channel_bins(dataloaderX=dataloaderX,
                                                truncated_duration= self.args.truncated_duration,
                                                clf_toi=clf_toi,
                                                rnBin=rnBin,
                                                split_type=splitName)
                        
                        ## Save the paths of the created dataset to fDataset
                        print(fpathList)
                        self.fDataset[self.logcat_filter_rtime_threshold][self.args.truncated_duration][clf_toi][rnBin][splitName] = fpathList


    def create_channel_bins(self, dataloaderX, truncated_duration, clf_toi, rnBin, split_type):
        """
        Creates the dataset (post feature engineering) for the different classification tasks. 
        Writes the dataset, corresponding labels, and file paths to npy files.
        
        params:
            - dataloaderX : dataloader for the training dataset that returns (batch_tensor, batch_labels, batch_paths) 
                            -batch_tensor (batch of iterations) : Shape - B, N_channels, feature_size
                            -batch_labels (label for each iteration in the batch) : Shape - B
                            -batch_paths (path of the file for each iteration in the batch) : Shape - B
            
            ----------------------------- Used for labelling the output files ----------------------------- 
            - self.logcat_filter_rtime_threshold
            - truncated_duration: time to which you want to trim the time series (in s)           
            - clf_toi : classification task of interest - "DVFS_partition_for_HPC_DVFS_fusion", "HPC_partition_for_HPC_DVFS_fusion", "HPC_individual", "DVFS_individual", or "DVFS_fusion"
            - rnBin : "rn1","rn2","rn3","rn4", or "all"
            - split_type : "train","trainSG", or "test" 
            -----------------------------------------------------------------------------------------------
            
        Output:
            Iterate over all the batches in the dataset and separate each of the channels into their respective bins
            - channel_bins : channel_bins[i][j] : stores the reduced feature of the jth iteration of the ith channel [Shape: N_ch, N_samples, feature_size]
            - labels       : labels[j] : stores the labels of the jth iteration
            - f_paths      : f_paths[j] : stores the file path (contains the file name) of the jth iteration

        Returns the paths of the files where channel_bins, labels, f_paths are stored, i.e., 
        {"path_channel_bins": ... , "path_labels": ... , "path_f_paths": ... }
        """

        # Dettermine the number of channels [=15 for GLOBL, and =1 for HPC]
        test_channel_bins,_,_ = next(iter(dataloaderX))
        _,num_channel_bins,_ = test_channel_bins.shape
        
        channel_bins = [[] for _ in range(num_channel_bins)] # Stores the reduced_feature of each channel
        labels = [] # Stores the labels of the corresponding index in the channel_bins
        f_paths = [] # Stores the file paths of the corresponding index in the channel_bins

        for batch_idx,(batch_tensor, batch_labels, batch_paths) in enumerate(dataloaderX):
            # Get the dimensions of the batch tensor
            B,N_ch,ft_size = batch_tensor.shape

            # print(f"Shape of batch (B,N_ch,reduced_feature_size) : {batch_tensor.shape}")
            
            # Split the batch into individual iterations
            for iter_idx,iterx in enumerate(torch.unbind(batch_tensor,dim=0)):
                # print(f"Shape of iteration tensor (Nch, reduced_feature_size): {iter_idx,iterx.shape}")

                # Add the label for this iteration into the labels array
                labels.append(batch_labels[iter_idx].item())

                # Add the file path for this iteration into the f_paths array
                f_paths.append(batch_paths[iter_idx])

                # Split the iteration into channels and store the channels into their respective bins
                for chn_idx,chn in enumerate(torch.unbind(iterx,dim=0)):
                    # print(f"Shape of channel tensor (reduced_feature_size): {chn_idx,chn.shape}")
                    
                    # create a new tensor that doesn't share the same underlying address
                    chn = chn.clone()
                    
                    # Add it to the channel bin
                    channel_bins[chn_idx].append(chn.numpy())
                    # print(f"****{channel_bins}")

                # print("Sanity checks")
                # print(iterx[0], channel_bins[0])
                # print(iterx[14], channel_bins[14])

                # if iter_idx == 1:
                #     sys.exit()

            # ### Sanity checks (for one batch) ###
            # print(f"Check if labels and batch_lables are same : {labels,batch_labels}")
            # # Check if binning of channels is correct
            # print(f"channel_bins : length (num_channels) - {len(channel_bins)} | length of each element channel_bins[i] (batch_size) - {len(channel_bins[0])} | length of channel[i][j] (feature_size) : {len(channel_bins[0][0])}")
            
            # if batch_idx==2:
            #     sys.exit()

        # Convert to numpy array
        channel_bins = np.array(channel_bins)
        labels = np.array(labels)

        # print(f"*************** Results path : {split_type} *************** ")
        # print(f"*************** Shape of channel_bins (N_ch, N_samples, feature_size) and corresponding labels (N_samples) : {channel_bins.shape, labels.shape} *************** ")
        # sys.exit()

        ############################################# Saving the files #############################################
        # Generate the paths for saving the files
        saveDir = os.path.join(self.results_path, str(self.logcat_filter_rtime_threshold), str(truncated_duration), clf_toi, rnBin) 
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)

        # Paths where files needs to be saved
        channel_bins_path = os.path.join(saveDir, f'channel_bins_{split_type}.npy')
        labels_path = os.path.join(saveDir, f'labels_{split_type}.npy')
        filePath_path = os.path.join(saveDir, f'file_paths_{split_type}.npy')

        # Save the numpy array
        with open(channel_bins_path, 'wb') as f:
            np.save(f, channel_bins, allow_pickle=True)

        with open(labels_path, 'wb') as f:
            np.save(f, labels, allow_pickle=True)
        
        # Save the file paths
        with open(filePath_path, 'wb') as fp:
            pickle.dump(f_paths, fp)
        ############################################################################################################

        return [channel_bins_path, labels_path, filePath_path]

def main_worker(args, xmd_base_folder_location):
    """
    Worker node that performs the complete analysis.
    params:
        - args: easydict storing the experimental parameters
        - xmd_base_folder_location: Location of base folder of xmd
    """
    # Download the dataset if it has not been downloaded
    if args.dataset_download_flag:
        dataset_generator_instance = dataset_generator_downloader(filter_values= [args.runtime_per_file, args.num_logcat_lines_per_file, args.freq_logcat_event_per_file], 
                                                                    dataset_type=args.dataset_type)
        dataset_generator_instance.generate_dataset(download_file_flag=args.dataset_download_flag)
    
    # Get the dataset type and the partition dist for the dataset split generator
    dsGen_dataset_type, dsGem_partition_dist = dataloader_generator.get_dataset_type_and_partition_dist(dataset_type = args.dataset_type)
    # dsGen_dataset_type, dsGem_partition_dist = dataloader_generator.get_dataset_type_and_partition_dist(dataset_type = 'bench-dataset')

    # Get the dataset base location
    dataset_base_location = os.path.join(args.dataset_base_location, args.dataset_type)

    # Generate the dataset splits (partition dict and the labels dict)
    dsGen = dataset_split_generator(seed=args.seed, 
                                    partition_dist=dsGem_partition_dist, 
                                    datasplit_dataset_type=dsGen_dataset_type)
    all_datasets = dsGen.create_all_datasets(base_location=dataset_base_location)

    ################################################## Generate feature engineered dataset ##################################################
    featEngDatsetBasePath = os.path.join(xmd_base_folder_location,"data","featureEngineeredDataset")
    fDataset={}
    featEngineeringDriver = feature_engineered_dataset(args=args, 
                                                all_dataset_partitionDict_label = all_datasets, 
                                                dataset_type = dsGen_dataset_type, 
                                                results_path = featEngDatsetBasePath,
                                                fDataset=fDataset)
    featEngineeringDriver.generate_feature_engineered_dataset()
    #########################################################################################################################################

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