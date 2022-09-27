'''
This dataloader is for the late stage fusion using classical ML algorithms. 
This dataloader will have dvfs as well as performance counter parser and processor.
'''

from cProfile import label
from matplotlib.style import available
import torch
import contextlib
import time
import csv
import random
import os
from os import listdir, mkdir
from os.path import isfile, join, isdir
import math
import sys
import shutil
import json
from torch._C import Value
from torch.nn.functional import pad
import torch.utils.data as data_utils
from collections import Counter
import numpy as np
from scipy.sparse import base, csr_matrix 
import matplotlib.pyplot as plt
import dropbox
import re
import torchaudio
import logging
import numpy as np
from sklearn.decomposition import PCA
import itertools
import pandas as pd
import pickle
from glob import glob
import shutil
import traceback

BENIGN_LABEL = 0
MALWARE_LABEL = 1

@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print('Total elapsed time for %s: %.3f' % (message, t1 - t0))

def download(dbx, path, download_path):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    #  path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
    while '//' in path:
        path = path.replace('//', '/')
    with stopwatch('download'):
        try:
            dbx.files_download_to_file(download_path, path)
        except dropbox.exceptions.HttpError as err:
            print('*** HTTP error ***', err)
            return None



def create_file_dict(file_list, file_type):
    """
    Creates a dict from the file list with key = file_hash and value = [((it0,rn0), index_in_file),((it1,rn0), index_in_file),..] i.e., list of tuples containing the rn and iter values associated with the hash 

    Input : 
        - file_list : List of file paths
        - file_type : DVFS or HPC (Different parsers for different file types)    
    
    Output : 
        -hash_dict :  key = file_hash and value = [((it0,rn0), index_in_file),((it1,rn0), index_in_file),..] 
    """
    # Determine the parser on the basis of file_type
    regex_pattern = None
    if file_type == 'simpleperf':
        regex_pattern = r'.*\/(.*)__.*it(\d*)_rn(\d*).txt'
    elif file_type == 'dvfs':
        regex_pattern = r'.*\/(.*)__.*iter_(\d*)_rn(\d*).txt'
    else:
        raise ValueError("Incorrect file type provided.")

    # Stores the output of this module
    hash_dict = {}

    # Populate the hash_dict
    # Parse the file_list_ to extract the hash from the file path [includes the file name]
    for file_indx,file_name in enumerate(file_list):
        file_hash_obj = re.search(regex_pattern, file_name, re.M|re.I)
        
        if file_hash_obj: 
            file_hash_string = file_hash_obj.group(1).strip()
            iter_val = int(file_hash_obj.group(2).strip())
            rn_val = int(file_hash_obj.group(3).strip())
            
            # Add this hash to the dict if its not present, else add the (iter_val,rn_val)
            if file_hash_string not in hash_dict:
                hash_dict[file_hash_string] = [((iter_val,rn_val),file_indx)]
            else:
                hash_dict[file_hash_string].append(((iter_val,rn_val), file_indx))
    
    # Sanity check for verifying that the parser's functionality [Total occurences in dict = Number of files in the folder]
    num_files_in_folder = len(file_list)
    total_occurences_per_folder = sum([len(dlist) for dlist in hash_dict.values()])
    
    # print("----------------------------------------------- TESTING PARSER --------------------------------------------------")
    # # print(hash_dict)
    # print(f"File Type : {file_type} | Num files : {num_files_in_folder} | Total occurences : {total_occurences_per_folder} | Equal : {num_files_in_folder == total_occurences_per_folder}")

    return hash_dict

def get_hpc_dvfs_file_list(hpc_path, dvfs_path):
    """
    Function to extract the corresponding dvfs files [if it exists] for the HPC logs. (In principle every HPC log should have a corresponding DVFS log)

    Input :
        - hpc_path : Path of the folder containing the HPC logs
        - dvfs_path : Path of the folder containing the DVFS logs

    Output : 
        - (matched_hpc_files, matched_dvfs_files)
            - matched_hpc_files : List of HPC file paths whose corresponding DVFS files have been found
            - matched_dvfs_files : Corresponding DVFS file paths for the HPC file (NOTE : Order is same as HPC files)
        
    """
    # Create a list of files that are present in each of the folders
    hpc_file_list = [join(hpc_path,f) for f in listdir(hpc_path) if isfile(join(hpc_path,f))]
    dvfs_file_list = [join(dvfs_path,f) for f in listdir(dvfs_path) if isfile(join(dvfs_path,f))]

    # Create the dict from the corresponding file lists
    hpc_dict = create_file_dict(hpc_file_list, "simpleperf")
    dvfs_dict = create_file_dict(dvfs_file_list, "dvfs")


    # Iterate through the hpc_dict and check if you have a corresponding file in dvfs_dict
    # If yes, then add the corresponding file paths to matched_hpc_files and matched_dvfs_files
    # Pick the first folder_dict and see which hashes are common with all the other folder_dicts
    matched_hpc_files = []
    matched_dvfs_files = []
    
    for hash_val, occurences in hpc_dict.items():
        
        if hash_val in dvfs_dict:
            # Found hash. Now check if there are common files for the two hashes.
            # Get a list of iter_and_rn tuples for this hash in dvfs dict
            iter_and_rn_dvfs_list = [ele[0] for ele in dvfs_dict[hash_val]]
            
            ## iter_and_rn_and_index[0] = (iter, rn) | iter_and_rn_and_index[1] = index
            for iter_and_rn_and_index in occurences: # For each iter_and_rn tuple in hpc for this hash, check if there is corresponding iter and hash in dvfs
                if iter_and_rn_and_index[0] in iter_and_rn_dvfs_list:
                    # Found match for iter and rn. Append the corresponding file paths to their corresponding lists using the index 
                    matched_hpc_files.append(hpc_file_list[iter_and_rn_and_index[1]])
                    matched_dvfs_files.append(dvfs_file_list[dvfs_dict[hash_val][iter_and_rn_dvfs_list.index(iter_and_rn_and_index[0])][1]])

    # # Sanity check : Length of the matched list should be same
    # print("----------------------------------------------- TESTING MATCHED FILE MODULE --------------------------------------------------")    
    # print(f"Length : matched_dvfs_files = {len(matched_dvfs_files)} |  matched_hpc_files = {len(matched_hpc_files)} | Equal = {len(matched_hpc_files) == len(matched_dvfs_files)}")

    return (matched_hpc_files, matched_dvfs_files)            


def create_matched_lists(base_location):
    """
    Function to create matched_lists for dvfs and simpleperf rn files. For each rn folder, identify the matched files in the dvfs folder and return a list of the matched
    files for both the rn folder and the dvfs folder.

    Directory structure assumed : 
        ---base_location/
            |
            ----benign/
                |
                ----dvfs/
                ----simpleperf/
                    |
                    ----rn1/    
                    ----rn2/
                    ----rn3/
                    ----rn4/
            ----malware/
                |
                ----dvfs/
                ----simpleperf/
                    |
                    ----rn1/    
                    ----rn2/
                    ----rn3/
                    ----rn4/
    Input : 
        - base_location : Location of the base folder. See the directory structure assumed.
    Output :
        - matched_lists_benign : [(matched_hpc_rn1_files, matched_dvfs_rn1_files), (...rn2...), (...rn3...), (...rn4...)]
        - matched_lists_malware : [(matched_hpc_rn1_files, matched_dvfs_rn1_files), (...rn2...), (...rn3...), (...rn4...)]
    """
    dvfs_benign_loc = os.path.join(base_location, "benign","dvfs")
    dvfs_malware_loc = os.path.join(base_location, "malware","dvfs")
    simpleperf_benign_rn_loc = [os.path.join(base_location, "benign","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]
    simpleperf_malware_rn_loc = [os.path.join(base_location, "malware","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]

    # Create matched lists for benign
    matched_lists_benign = []
    for benign_perf_loc in simpleperf_benign_rn_loc:
        # print(f"********************************************** Generating matched list : {benign_perf_loc} : **********************************************")
        matched_lists_benign.append(get_hpc_dvfs_file_list(hpc_path = benign_perf_loc, dvfs_path = dvfs_benign_loc))

    # Create matched lists for malware
    matched_lists_malware = []    
    for malware_perf_loc in simpleperf_malware_rn_loc:
        # print(f"********************************************** Generating matched list : {malware_perf_loc} : **********************************************")
        matched_lists_malware.append(get_hpc_dvfs_file_list(hpc_path = malware_perf_loc, dvfs_path = dvfs_malware_loc))
    
    # # Testing # #
    # for i,j in matched_lists_benign:
    #     for x,y in zip(i,j):
    #         print(f" - {x} ====== {y}")

    return matched_lists_benign, matched_lists_malware
        
def create_all_datasets(base_location):
    """
    Function to create splits: Train, Val, Test for all the tasks:
                                - Individual DVFS
                                - Fused DVFS
                                - Individual HPC
                                - Fused HPC
                                - HPC_DVFS fused (DVFS part)
                                - HPC_DVFS fused (HPC part)

    Input : 
        - base_location : Location of the base folder. See the directory structure in create_matched_lists()
    Output :
        - Partition and partition labels for DVFS individual, DVFS fusion, HPC individual, HPC fusion, HPC partition of DVFS-HPC fusion, DVFS partition of DVFS-HPC fusion
        -  [(DVFS_partition_for_HPC_DVFS_fusion, DVFS_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_DVFS_fusion, HPC_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual),
            (HPC_partition_for_HPC_fusion,HPC_partition_labels_for_HPC_fusion),
            (DVFS_partition_for_DVFS_individual,DVFS_partition_labels_for_DVFS_individual),
            (DVFS_partition_for_DVFS_fusion,DVFS_partition_labels_for_DVFS_fusion)]

    """
    ####### To keep track of which files have not been selected for testing and validation [For Individual/Fused DVFS and Individual/Fused HPC] #######
    dvfs_benign_loc = os.path.join(base_location, "benign","dvfs")
    dvfs_malware_loc = os.path.join(base_location, "malware","dvfs")
    simpleperf_benign_rn_loc = [os.path.join(base_location, "benign","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]
    simpleperf_malware_rn_loc = [os.path.join(base_location, "malware","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]

    # Generate file_lists from these locations
    dvfs_benign_file_list = [join(dvfs_benign_loc,f) for f in listdir(dvfs_benign_loc) if isfile(join(dvfs_benign_loc,f))]
    dvfs_malware_file_list = [join(dvfs_malware_loc,f) for f in listdir(dvfs_malware_loc) if isfile(join(dvfs_malware_loc,f))]
    simpleperf_benign_file_list = [[join(_path,f) for f in listdir(_path) if isfile(join(_path,f))] for _path in simpleperf_benign_rn_loc]
    simpleperf_malware_file_list = [[join(_path,f) for f in listdir(_path) if isfile(join(_path,f))] for _path in simpleperf_malware_rn_loc]

    # Create a dict from these lists with key = file_path | value = Indicator (0 or 1) to identify whether this file has been selected for some spit or not
    dvfs_benign_file_dict = {path: 0 for path in dvfs_benign_file_list}
    dvfs_malware_file_dict = {path: 0 for path in dvfs_malware_file_list}
    simpleperf_benign_file_dict = [{path: 0 for path in simpleperf_rn} for simpleperf_rn in simpleperf_benign_file_list]
    simpleperf_malware_file_dict = [{path: 0 for path in simpleperf_rn} for simpleperf_rn in simpleperf_malware_file_list]
    ###################################################################################################################################################

    #########################******************************** Creating the splits for HPC-DVFS fusion ******************************##############################
    # Get the list of matched files
    #  - matched_lists_benign : [(matched_hpc_rn1_files, matched_dvfs_rn1_files), (...rn2...), (...rn3...), (...rn4...)]
    #  - matched_lists_malware : [(matched_hpc_rn1_files, matched_dvfs_rn1_files), (...rn2...), (...rn3...), (...rn4...)]
    matched_list_benign, matched_list_malware = create_matched_lists(base_location)

    # DVFS dataset partitions for HPC fusion
    # DVFS_partition_for_HPC_fusion : [dvfs_partition_for_fusion_with_rn1, dvfs_partition_for_fusion_with_rn2, ...rn3, ...rn4] 
    DVFS_partition_for_HPC_DVFS_fusion = []
    DVFS_partition_labels_for_HPC_DVFS_fusion = []
    
    # HPC_partition_for_HPC_fusion : [hpc_partition_for_fusion_with_rn1, hpc_partition_for_fusion_with_rn2, ...rn3, ...rn4]
    HPC_partition_for_HPC_DVFS_fusion = []
    HPC_partition_labels_for_HPC_DVFS_fusion = []

    for indx,(rn_hpc_dvfs_benign,rn_hpc_dvfs_malware) in enumerate(zip(matched_list_benign, matched_list_malware)):
        ############################################ Splits for HPC DVFS fusion ############################################
        ######---------------------------------------------------------------- DVFS --------------------------------------------------------------######
        # Create splits for benign and malware [dvfs]
        benign_label, malware_label = create_labels_from_filepaths(benign_filepaths= rn_hpc_dvfs_benign[1], malware_filepaths= rn_hpc_dvfs_malware[1])
        # Create the partition dict using the matched labels
        partition = create_splits(benign_label= benign_label,malware_label= malware_label, partition_dist=None)
        DVFS_partition_for_HPC_DVFS_fusion.append(partition)

         # You can use all the files (not just the matched files) for creating labels.
        all_benign_label, all_malware_label = create_labels_from_filepaths(benign_filepaths= dvfs_benign_file_list, malware_filepaths= dvfs_malware_file_list)
        all_labels = {**all_benign_label,**all_malware_label}
        DVFS_partition_labels_for_HPC_DVFS_fusion.append(all_labels) # One labels dict for each rn
        
        # Mark the files that are used in the val and test splits [The unmarked files will be used in the training for Individual and Fused DVFS]
        for file_path in partition['val']:
            if file_path in dvfs_benign_file_dict:
                dvfs_benign_file_dict[file_path] = 1
            elif file_path in dvfs_malware_file_dict:
                dvfs_malware_file_dict[file_path] = 1
        
        for file_path in partition['test']:
            if file_path in dvfs_benign_file_dict:
                dvfs_benign_file_dict[file_path] = 1
            elif file_path in dvfs_malware_file_dict:
                dvfs_malware_file_dict[file_path] = 1

        ########---------------------------------------------------------------- HPC --------------------------------------------------------------######
        # Create splits for benign and malware [hpc]
        benign_label, malware_label = create_labels_from_filepaths(benign_filepaths= rn_hpc_dvfs_benign[0], malware_filepaths= rn_hpc_dvfs_malware[0])
        
        partition = create_splits(benign_label= benign_label,malware_label= malware_label, partition_dist=None)
        HPC_partition_for_HPC_DVFS_fusion.append(partition)

        # You can use all the files for a given rn (not just the matched files) for creating labels.
        all_benign_label, all_malware_label = create_labels_from_filepaths(benign_filepaths= simpleperf_benign_file_list[indx], malware_filepaths= simpleperf_malware_file_list[indx])
        all_labels = {**all_benign_label,**all_malware_label}
        HPC_partition_labels_for_HPC_DVFS_fusion.append(all_labels)

        # Mark the files that are used in the val and test splits [The unmarked files will be used in the training for Individual and Fused HPC]
        for file_path in partition['val']:
            if file_path in simpleperf_benign_file_dict[indx]:
                simpleperf_benign_file_dict[indx][file_path] = 1
            elif file_path in simpleperf_malware_file_dict[indx]:
                simpleperf_malware_file_dict[indx][file_path] = 1
        
        for file_path in partition['test']:
            if file_path in simpleperf_benign_file_dict[indx]:
                simpleperf_benign_file_dict[indx][file_path] = 1
            elif file_path in simpleperf_malware_file_dict[indx]:
                simpleperf_malware_file_dict[indx][file_path] = 1

    # print("********** Stats for HPC partitions HPC-DVFS fusion ********** ")
    # for rn_indx, rn_partition_dict in enumerate(HPC_partition_for_HPC_DVFS_fusion):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")

    
    # print("********** Stats for HPC partitions HPC-DVFS fusion ********** ")
    # for rn_indx, rn_partition_dict in enumerate(DVFS_partition_for_HPC_DVFS_fusion):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")

    # ##### Testing for one to one correspondence between the DVFS partition and HPC partition #####
    # rn_minus_1 = 3 # For selecting the rn_val
    # for i,j in zip(DVFS_partition_for_HPC_DVFS_fusion[rn_minus_1]['val'], HPC_partition_for_HPC_DVFS_fusion[rn_minus_1]['val']):
    #     print(f"- {i} ====== {j} ====== {DVFS_partition_labels_for_HPC_DVFS_fusion[rn_minus_1][i]} ======= {HPC_partition_labels_for_HPC_DVFS_fusion[rn_minus_1][j]}")
    
    #########################***************************************************************************************************##########################
    
    #########################******************************** Creating the splits for Individual HPC and HPC Fusion ******************************##############################

    # For individual HPC, there is no val dataset, only test dataset = val+test dataset in HPC_DVFS fusion
    HPC_partition_for_HPC_individual = [{'test':partition['val']+partition['test']} for partition in HPC_partition_for_HPC_DVFS_fusion] # One partition for each rn
    HPC_partition_labels_for_HPC_individual = HPC_partition_labels_for_HPC_DVFS_fusion

    # For fusion HPC, the val and test dataset is the same as HPC_DVFS fusion
    HPC_partition_for_HPC_fusion = [{'val': partition['val'],'test':partition['test']} for partition in HPC_partition_for_HPC_DVFS_fusion]
    HPC_partition_labels_for_HPC_fusion = HPC_partition_labels_for_HPC_DVFS_fusion
    
    # Creating the training partition for Fused and Individual HPC
    # Create a list of all the samples that were not used for val and testing
    train_benign = [[path for path,taken in simpleperf_benign_file_rn_dict.items() if taken==0] for simpleperf_benign_file_rn_dict in simpleperf_benign_file_dict]
    train_malware = [[path for path,taken in simpleperf_malware_file_rn_dict.items() if taken==0] for simpleperf_malware_file_rn_dict in simpleperf_malware_file_dict]
    train = [train_benign_rn+train_malware_rn for train_benign_rn,train_malware_rn in zip(train_benign,train_malware)]
    
    # Add it to the partition for both individual and Fused HPC
    for indx,rn_dict in enumerate(HPC_partition_for_HPC_individual):
        rn_dict['train'] = train[indx]
    for indx,rn_dict in enumerate(HPC_partition_for_HPC_fusion):
        rn_dict['train'] = train[indx]

    # print("********** Stats for HPC partitions individual ********** ")
    # for rn_indx, rn_partition_dict in enumerate(HPC_partition_for_HPC_individual):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")

    
    # print("********** Stats for HPC partitions fusion ********** ")
    # for rn_indx, rn_partition_dict in enumerate(HPC_partition_for_HPC_fusion):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")
    
    #########################******************************** Creating the splits for Individual DVFS and DVFS Fusion ******************************##############################

    # For individual DVFS, there is no val dataset, only test dataset = val+test dataset in HPC_DVFS fusion
    DVFS_partition_for_DVFS_individual = {'test':list(itertools.chain.from_iterable([partition['val']+partition['test'] for partition in DVFS_partition_for_HPC_DVFS_fusion]))}
    DVFS_partition_labels_for_DVFS_individual = {k:v for d in DVFS_partition_labels_for_HPC_DVFS_fusion for k,v in d.items()}

    # For fusion DVFS, the val and test dataset is the same as HPC_DVFS fusion
    DVFS_partition_for_DVFS_fusion = {'test':list(itertools.chain.from_iterable([partition['test'] for partition in DVFS_partition_for_HPC_DVFS_fusion])),
                                       'val':list(itertools.chain.from_iterable([partition['val'] for partition in DVFS_partition_for_HPC_DVFS_fusion])) }
    DVFS_partition_labels_for_DVFS_fusion = DVFS_partition_labels_for_DVFS_individual
    
    # Creating the training partition for Fused and Individual DVFS
    # Create a list of all the samples that were not used for val and testing
    train_benign = [path for path,taken in dvfs_benign_file_dict.items() if taken==0]
    train_malware = [path for path,taken in dvfs_malware_file_dict.items() if taken==0]
    train = train_benign+train_malware
    # Shuffle the list in place
    random.shuffle(train)   
    
    # Add it to the partition for both individual and Fused DVFS
    DVFS_partition_for_DVFS_individual['train'] = train
    DVFS_partition_for_DVFS_fusion['train'] = train

    # # Testing the partition for individual and fused dvfs
    # # print(DVFS_partition_for_DVFS_individual['train'])
    # print(" ********** Stats for DVFS individual ********** ")
    # for key,value in DVFS_partition_for_DVFS_individual.items():
    #     print(f"  - {key, len(value)}")
    
    # print(" ********** Stats for DVFS fusion **********  ")
    # for key,value in DVFS_partition_for_DVFS_fusion.items():
    #     print(f"  - {key, len(value)}")
    #########################**************************************************************************************************************************##########################


    return [(DVFS_partition_for_HPC_DVFS_fusion, DVFS_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_DVFS_fusion, HPC_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual),
            (HPC_partition_for_HPC_fusion,HPC_partition_labels_for_HPC_fusion),
            (DVFS_partition_for_DVFS_individual,DVFS_partition_labels_for_DVFS_individual),
            (DVFS_partition_for_DVFS_fusion,DVFS_partition_labels_for_DVFS_fusion)]

def create_all_datasets_for_unknown_dataset(base_location):
    """
    Function to create splits: Train, Val, Test for all the tasks:
                                - Individual DVFS
                                - Fused DVFS
                                - Individual HPC
                                - Fused HPC
                                - HPC_DVFS fused (DVFS part)
                                - HPC_DVFS fused (HPC part)

    Input : 
        - base_location : Location of the base folder. See the directory structure in create_matched_lists()
    Output :
        - Partition and partition labels for DVFS individual, DVFS fusion, HPC individual, HPC fusion, HPC partition of DVFS-HPC fusion, DVFS partition of DVFS-HPC fusion
        -  [(DVFS_partition_for_HPC_DVFS_fusion, DVFS_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_DVFS_fusion, HPC_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual),
            (HPC_partition_for_HPC_fusion,HPC_partition_labels_for_HPC_fusion),
            (DVFS_partition_for_DVFS_individual,DVFS_partition_labels_for_DVFS_individual),
            (DVFS_partition_for_DVFS_fusion,DVFS_partition_labels_for_DVFS_fusion)]

    """
    ####### To keep track of which files have not been selected for testing and validation [For Individual/Fused DVFS and Individual/Fused HPC] #######
    dvfs_benign_loc = os.path.join(base_location, "benign","dvfs")
    dvfs_malware_loc = os.path.join(base_location, "malware","dvfs")
    simpleperf_benign_rn_loc = [os.path.join(base_location, "benign","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]
    simpleperf_malware_rn_loc = [os.path.join(base_location, "malware","simpleperf",rn) for rn in ['rn1','rn2','rn3','rn4']]

    # Generate file_lists from these locations
    dvfs_benign_file_list = [join(dvfs_benign_loc,f) for f in listdir(dvfs_benign_loc) if isfile(join(dvfs_benign_loc,f))]
    dvfs_malware_file_list = [join(dvfs_malware_loc,f) for f in listdir(dvfs_malware_loc) if isfile(join(dvfs_malware_loc,f))]
    simpleperf_benign_file_list = [[join(_path,f) for f in listdir(_path) if isfile(join(_path,f))] for _path in simpleperf_benign_rn_loc]
    simpleperf_malware_file_list = [[join(_path,f) for f in listdir(_path) if isfile(join(_path,f))] for _path in simpleperf_malware_rn_loc]

    # Create a dict from these lists with key = file_path | value = Indicator (0 or 1) to identify whether this file has been selected for some spit or not
    dvfs_benign_file_dict = {path: 0 for path in dvfs_benign_file_list}
    dvfs_malware_file_dict = {path: 0 for path in dvfs_malware_file_list}
    simpleperf_benign_file_dict = [{path: 0 for path in simpleperf_rn} for simpleperf_rn in simpleperf_benign_file_list]
    simpleperf_malware_file_dict = [{path: 0 for path in simpleperf_rn} for simpleperf_rn in simpleperf_malware_file_list]
    ###################################################################################################################################################

    #########################******************************** Creating the splits for HPC-DVFS fusion ******************************##############################
    # Get the list of matched files
    #  - matched_lists_benign : [(matched_hpc_rn1_files, matched_dvfs_rn1_files), (...rn2...), (...rn3...), (...rn4...)]
    #  - matched_lists_malware : [(matched_hpc_rn1_files, matched_dvfs_rn1_files), (...rn2...), (...rn3...), (...rn4...)]
    matched_list_benign, matched_list_malware = create_matched_lists(base_location)

    # DVFS dataset partitions for HPC fusion
    # DVFS_partition_for_HPC_fusion : [dvfs_partition_for_fusion_with_rn1, dvfs_partition_for_fusion_with_rn2, ...rn3, ...rn4] 
    DVFS_partition_for_HPC_DVFS_fusion = []
    DVFS_partition_labels_for_HPC_DVFS_fusion = []
    
    # HPC_partition_for_HPC_fusion : [hpc_partition_for_fusion_with_rn1, hpc_partition_for_fusion_with_rn2, ...rn3, ...rn4]
    HPC_partition_for_HPC_DVFS_fusion = []
    HPC_partition_labels_for_HPC_DVFS_fusion = []

    for indx,(rn_hpc_dvfs_benign,rn_hpc_dvfs_malware) in enumerate(zip(matched_list_benign, matched_list_malware)):
        ############################################ Splits for HPC DVFS fusion ############################################
        ######---------------------------------------------------------------- DVFS --------------------------------------------------------------######
        # Create splits for benign and malware [dvfs]
        benign_label, malware_label = create_labels_from_filepaths(benign_filepaths= rn_hpc_dvfs_benign[1], malware_filepaths= rn_hpc_dvfs_malware[1])
        # Create the partition dict using the matched labels [100% of samples are in test partition]
        partition = create_splits(benign_label= benign_label,malware_label= malware_label, partition_dist=[0,0,1])
        # print(f" - DVFS_partition_HPC_DVFS_fusion rn{indx+1} : {len(partition['train']),len(partition['val']),len(partition['test'])}")
        
        DVFS_partition_for_HPC_DVFS_fusion.append(partition)

         # You can use all the files (not just the matched files) for creating labels.
        all_benign_label, all_malware_label = create_labels_from_filepaths(benign_filepaths= dvfs_benign_file_list, malware_filepaths= dvfs_malware_file_list)
        all_labels = {**all_benign_label,**all_malware_label}
        DVFS_partition_labels_for_HPC_DVFS_fusion.append(all_labels) # One labels dict for each rn
        
       
        ########---------------------------------------------------------------- HPC --------------------------------------------------------------######
        # Create splits for benign and malware [hpc]
        benign_label, malware_label = create_labels_from_filepaths(benign_filepaths= rn_hpc_dvfs_benign[0], malware_filepaths= rn_hpc_dvfs_malware[0])
        
        partition = create_splits(benign_label= benign_label,malware_label= malware_label, partition_dist=[0,0,1])
        # print(f" - HPC_partition_HPC_DVFS_fusion rn{indx+1} : {len(partition['train']),len(partition['val']),len(partition['test'])}")

        HPC_partition_for_HPC_DVFS_fusion.append(partition)

        # You can use all the files for a given rn (not just the matched files) for creating labels.
        all_benign_label, all_malware_label = create_labels_from_filepaths(benign_filepaths= simpleperf_benign_file_list[indx], malware_filepaths= simpleperf_malware_file_list[indx])
        all_labels = {**all_benign_label,**all_malware_label}
        HPC_partition_labels_for_HPC_DVFS_fusion.append(all_labels)

        
    # print("********** Stats for HPC partitions HPC-DVFS fusion ********** ")
    # for rn_indx, rn_partition_dict in enumerate(HPC_partition_for_HPC_DVFS_fusion):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")

    
    # print("********** Stats for HPC partitions HPC-DVFS fusion ********** ")
    # for rn_indx, rn_partition_dict in enumerate(DVFS_partition_for_HPC_DVFS_fusion):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")

    # ##### Testing for one to one correspondence between the DVFS partition and HPC partition #####
    # rn_minus_1 = 3 # For selecting the rn_val
    # for i,j in zip(DVFS_partition_for_HPC_DVFS_fusion[rn_minus_1]['test'], HPC_partition_for_HPC_DVFS_fusion[rn_minus_1]['test']):
    #     print(f"- {i} ====== {j} ====== {DVFS_partition_labels_for_HPC_DVFS_fusion[rn_minus_1][i]} ======= {HPC_partition_labels_for_HPC_DVFS_fusion[rn_minus_1][j]}")
    # sys.exit()
    #########################***************************************************************************************************##########################
    
    #########################******************************** Creating the splits for Individual HPC and HPC Fusion ******************************##############################
    HPC_partition_for_HPC_individual = []
    HPC_partition_for_HPC_fusion = []
    # For each rn
    for rn_val in range(4):
        # Get the file list for malware and benign
        file_list_b = simpleperf_benign_file_list[rn_val]
        file_list_m = simpleperf_malware_file_list[rn_val]

        # Create labels
        benign_label, malware_label = create_labels_from_filepaths(benign_filepaths= file_list_b, malware_filepaths= file_list_m)

        # Create partition dict from the labels [100% of samples in the test dataset]
        partition = create_splits(benign_label= benign_label,malware_label= malware_label, partition_dist=[0,0,1])

        # Append it to HPC individual and HPC fusion
        HPC_partition_for_HPC_individual.append(partition)
        HPC_partition_for_HPC_fusion.append(partition)

        # print(f" - HPC_individual rn{indx+1} : {len(partition['train']),len(partition['val']),len(partition['test'])}")

    # Use the old labels dict
    HPC_partition_labels_for_HPC_individual = HPC_partition_labels_for_HPC_DVFS_fusion
    HPC_partition_labels_for_HPC_fusion = HPC_partition_labels_for_HPC_DVFS_fusion
    

    # print("********** Stats for HPC partitions individual ********** ")
    # for rn_indx, rn_partition_dict in enumerate(HPC_partition_for_HPC_individual):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     print(f" - Length of label dict : {len(HPC_partition_labels_for_HPC_individual[rn_indx])}")
    
    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")

    
    # print("********** Stats for HPC partitions fusion ********** ")
    # for rn_indx, rn_partition_dict in enumerate(HPC_partition_for_HPC_fusion):
    #     print(f" - Stats for rn : {rn_indx+1}")
    #     print(f" - Length of label dict : {len(HPC_partition_labels_for_HPC_fusion[rn_indx])}")

    #     for key,value in rn_partition_dict.items():
    #         print(f"  - {key,len(value)}")
    # sys.exit()
    #########################******************************** Creating the splits for Individual DVFS and DVFS Fusion ******************************##############################
    DVFS_partition_for_DVFS_individual = None
    DVFS_partition_for_DVFS_fusion = None
    
    # Get the file list for malware and benign
    file_list_b = dvfs_benign_file_list
    file_list_m = dvfs_malware_file_list

    # Create labels
    benign_label, malware_label = create_labels_from_filepaths(benign_filepaths= file_list_b, malware_filepaths= file_list_m)

    # Create partition dict from the labels [100% of samples in the test dataset]
    partition = create_splits(benign_label= benign_label,malware_label= malware_label, partition_dist=[0,0,1])

    # Add the partition to fusion and individual
    DVFS_partition_for_DVFS_fusion = partition
    DVFS_partition_for_DVFS_individual = partition

    # Generate the labels dict
    all_labels = {**benign_label,**malware_label}
        
    # print(f" - DVFS individual and fusion : {len(partition['train']),len(partition['val']),len(partition['test'])}")

    # Use the old labels dict
    DVFS_partition_labels_for_DVFS_individual = all_labels
    DVFS_partition_labels_for_DVFS_fusion = all_labels
    

    # # Testing the partition for individual and fused dvfs
    # # print(DVFS_partition_for_DVFS_individual['train'])
    # print(" ********** Stats for DVFS individual ********** ")
    # for key,value in DVFS_partition_for_DVFS_individual.items():
    #     print(f"  - {key, len(value)}")
    
    # print(" ********** Stats for DVFS fusion **********  ")
    # for key,value in DVFS_partition_for_DVFS_fusion.items():
    #     print(f"  - {key, len(value)}")
    
    # print(f" - Length of label dict : {len(all_labels)}")
    # sys.exit()
    #########################**************************************************************************************************************************##########################


    return [(DVFS_partition_for_HPC_DVFS_fusion, DVFS_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_DVFS_fusion, HPC_partition_labels_for_HPC_DVFS_fusion),
            (HPC_partition_for_HPC_individual,HPC_partition_labels_for_HPC_individual),
            (HPC_partition_for_HPC_fusion,HPC_partition_labels_for_HPC_fusion),
            (DVFS_partition_for_DVFS_individual,DVFS_partition_labels_for_DVFS_individual),
            (DVFS_partition_for_DVFS_fusion,DVFS_partition_labels_for_DVFS_fusion)]


def get_common_apps(path_list):
    """
    Function to extract the common applications that are present in the list of folders whose path is specified in path_list

    Input : 
        path_list : List of folder paths containing the data logs
    Output :
        common_app_list : List of app_hashes that are common in all the folders specified in path_list
    """

    # Create a list of files that are present in each of the folders
    # file_list_per_folder : [ [list of files in path_list[0]], [list of files in path_list[1]], [], ...]
    file_list_per_folder = [[join(_path,f) for f in listdir(_path) if isfile(join(_path,f))] for _path in path_list]
    
    # Extract hashes from each file list and create a dict: key = hash | value = # of occurences of the hash (should be <=2)
    # app_list_hash = [{hash1:num_occurences, hash2:num_occurences, ...}, {...}, {...}, ...]
    app_list_hash = [{} for _ in path_list] # Creating one dict for each folder

    for hash_list_dict, file_list_ in zip(app_list_hash, file_list_per_folder): # For each folder 

        # Parse the file_list_ to extract the hash from the file path [includes the file name]
        for file_name in file_list_:
            file_hash_obj = re.search(r'.*\/(.*)__.*', file_name, re.M|re.I)
            
            if file_hash_obj: 
                file_hash_string = file_hash_obj.group(1).strip()
                
                # Add this hash to the dict if its not present, else update the # of occurences
                if file_hash_string not in hash_list_dict:
                    hash_list_dict[file_hash_string] = 1
                else:
                    hash_list_dict[file_hash_string] += 1
            
    # Sanity check for verifying that the parser's functionality [Total occurences in dict = Number of files in the folder]
    num_files_per_folder = [len(folder) for folder in file_list_per_folder]
    total_occurences_per_folder = [sum(d.values()) for d in app_list_hash]
    total_apk_per_folder = [len(d.keys()) for d in app_list_hash]
    
    print("-------------------------------------------------------------------------------------------------")
    print(f"Num apks per folder : {total_apk_per_folder}")
    print(f"Num files : {num_files_per_folder} | Total occurences : {total_occurences_per_folder} | Equal : {num_files_per_folder == total_occurences_per_folder}")

    # Find the common apps in all the folders specified in path_list
    # common_app_hashes = [(hash1, [num_files_folder1, num_files_folder2, ...]), (...), (...), ...]
    common_app_hashes = [] # Stores the hashes and total files of the common apps that are present in all the folders
    
    num_folders = len(path_list) # Number of folders
     
    # Pick the first folder_dict and see which hashes are common with all the other folder_dicts
    for hash_val, occurences in app_list_hash[0].items():
        hash_present_in_folder = 1
        # Number of files for this apk
        num_files_for_apk = [occurences]

        # Check if this hash is present in other folders. 
        for hash_list_dict in app_list_hash[1:]:
            if hash_val in hash_list_dict:
                hash_present_in_folder+=1
                
                num_files_for_apk.append(hash_list_dict[hash_val]) # Collect the number of files for this apk
        
        if hash_present_in_folder == num_folders: 
            # Hash is common to all the folders. Add it to the common app hashes
            common_app_hashes.append((hash_val,num_files_for_apk))
            

    # Common apks across all folders, and the number of files in each folder for the common apks
    print(f" - Number of common apks : {len(common_app_hashes)}")
    print(f" - Number of files for the common apks : {[ sum([occur[i] for hashv,occur in common_app_hashes]) for i in range(len(path_list))]}")

    return common_app_hashes

def create_labels_from_filepaths(benign_filepaths = None, malware_filepaths = None):
    '''
    Function to create a dict containing file location and its corresponding label
    Input : -benign_filepaths - List of file paths of the benign logs
            -malware_filepaths - List of file paths of the malware logs
    
    Output : -benign_label = {file_path1 : 0, file_path2: 0, ...}  (Benigns have label 0)
             -malware_label = {file_path1 : 1, file_path2: 1, ...}  (Malware have label 1)   
    '''

    # Create the labels dict from the list
    if benign_filepaths is not None:
        benign_label = {path: BENIGN_LABEL for path in benign_filepaths}
    
    if malware_filepaths is not None:
        malware_label = {path: MALWARE_LABEL for path in malware_filepaths} 

    if benign_filepaths is None: # Just return the malware labels
        return malware_label
    
    elif malware_filepaths is None: # Just return the benign labels
        return benign_label

    elif (benign_filepaths is None) and (malware_filepaths is None):
        raise ValueError('Need to pass arguments to create_labels_from_filepaths()')

    return benign_label, malware_label

def create_labels(benign_path, malware_path):
    '''
    Function to create a dict containing file location and its corresponding label
    Input : -benign_path (path of directory where the benign files are stored)
            -malware_path (path of directory where the malware files are stored)
    
    Output : -benign_label = {file_path1 : 0, file_path2: 0, ...}  (Benigns have label 0)
             -malware_label = {file_path1 : 1, file_path2: 1, ...}  (Malware have label 1)   
    '''

    # Creating a list of files in the benign_path and malware_path
    benign_filelist = [join(benign_path,f) for f in listdir(benign_path) if isfile(join(benign_path,f))]
    malware_filelist = [join(malware_path,f) for f in listdir(malware_path) if isfile(join(malware_path,f))]

    # Create the labels dict from the list
    benign_label = {path: BENIGN_LABEL for path in benign_filelist}
    malware_label = {path: MALWARE_LABEL for path in malware_filelist} 

    # for key,value in benign_label.items():
    #     print(key,value)
    
    # for key,value in malware_label.items():
    #     print(key,value)
        
    # print(len(benign_label), len(malware_label))

    return benign_label, malware_label
        

def create_splits(benign_label=None, malware_label=None, partition_dist = None):
    '''
    Function for splitting the dataset into Train, Test, and Validation
    NOTE: If any of benign_label or malware_label is not passed as argument, then we ignore that, and
          create splits from whatever is passed as argument.

    Input : -benign_label = {file_path1 : 0, file_path2: 0, ...}  (Benigns have label 0)
            -malware_label = {file_path1 : 1, file_path2: 1, ...}  (Malware have label 1)
            -partition_dist = [num_train_%, num_val_%, num_test_%] -> If not passed then default split is [0.70,0.15,0.15]

    Output : -partition = {'train' : [file_path1, file_path2, ..],
                            'test' : [file_path1, file_path2, ..],
                            'val' : [file_path1, file_path2]}
    '''
    # Fix the seed value of random number generator for reproducibility
    random.seed(10) 
    
    # Create the partition dict (This is the output.)
    partition = {'train':[], 'test':[], 'val':[]}   

    # Decide the % of samples in the partition
    if partition_dist is None:
        # If partitioning percentage is not passed, then default split is [0.70,0.15,0.15]
        partition_dist = [0.70,0.15,0.15]
    else:
        partition_dist = partition_dist

    ################################## Handling the benign labels ##################################
    if benign_label is not None:
        # Shuffle the dicts of benign and malware: Convert to list. Shuffle. 
        benign_label_list = list(benign_label.items())
        random.shuffle(benign_label_list)

        # Calculate the number of training, validation, and test samples (70,15,15 split)
        num_train_benign, num_val_benign, num_test_benign = [math.ceil(x * len(benign_label)) for x in partition_dist]

        # Dividing the list of benign files into training, validation, and test buckets
        benign_train_list = benign_label_list[:num_train_benign]
        benign_val_list = benign_label_list[num_train_benign+1:num_train_benign+num_val_benign]
        benign_test_list = benign_label_list[num_train_benign+num_val_benign+1:num_train_benign+num_val_benign+num_test_benign]

        # Add items in train list to train partition
        for path,label  in benign_train_list:
            partition['train'].append(path)

        # Add items in val list to val partition
        for path,label  in benign_val_list:
            partition['val'].append(path)

        # Add items in test list to test partition
        for path,label  in benign_test_list:
            partition['test'].append(path)
    ################################################################################################
    if malware_label is not None:
        # Shuffle the dicts of benign and malware: Convert to list. Shuffle. 
        malware_label_list = list(malware_label.items())
        random.shuffle(malware_label_list)

        # Calculate the number of training, validation, and test samples (70,15,15 split)
        num_train_malware, num_val_malware, num_test_malware = [math.ceil(x * len(malware_label)) for x in partition_dist]

        # Dividing the list of malware files into training, validation, and test buckets
        malware_train_list = malware_label_list[:num_train_malware]
        malware_val_list = malware_label_list[num_train_malware+1:num_train_malware+num_val_malware]
        malware_test_list = malware_label_list[num_train_malware+num_val_malware+1:num_train_malware+num_val_malware+num_test_malware]

        # Add items in train list to train partition
        for path,label  in malware_train_list:
            partition['train'].append(path)

        # Add items in val list to val partition
        for path,label  in malware_val_list:
            partition['val'].append(path)

        # Add items in test list to test partition
        for path,label  in malware_test_list:
            partition['test'].append(path)
    
    # Shuffle the partitions
    random.shuffle(partition['train'])
    random.shuffle(partition['test'])
    random.shuffle(partition['val'])

    return partition


''' This is the Dataset object.
A Dataset object loads the training or test data into memory.
Your custom Dataset class MUST inherit torch's Dataset
Your custom Dataset class should have 3 methods implemented (you can add more if you want but these 3 are essential):
__init__(self) : Performs data loading
__getitem__(self, index) :  Will allow for indexing into the dataset eg dataset[0]
__len__(self) : len(dataset)
'''
# Custom dataset class for the dataloader
class arm_telemetry_data(torch.utils.data.Dataset):

    def __init__(self, partition, labels, split, file_type, normalize=True):
        '''
            -labels = {file_path1 : 0, file_path2: 0, ...}

            -partition = {'train' : [file_path1, file_path2, ..],
                                'test' : [file_path1, file_path2, ..],
                                'val' : [file_path1, file_path2]}

            -split = 'train', 'test', or 'val'
            -file_type = 'dvfs' or 'simpleperf' [Different parsers for different file types]                    
        '''
        if(split not in ['train','test','val']):
            raise NotImplementedError('Can only accept Train, Val, Test')

        # Store the list of paths (ids) in the split
        self.path_ids = partition[split] 
        # print(f"- List of first 10 path ids : {self.path_ids[:10]}")
        
        # Store the list of labels
        self.labels = labels
        # Whether or not you want to normalize the data [default=True]
        self.normalize = normalize
        # File type for selecting the parser module
        self.file_type = file_type

    def __len__(self):
        return len(self.path_ids)
     
    def __getitem__(self, idx):
        # Select the sample [id = file path of the dvfs file]
        id = self.path_ids[idx]

        # Get the label
        y = self.labels[id]

        if self.file_type == 'dvfs':
            # Read and parse the file. NOTE: Each dvfs file has a different sampling frequency
            X = self.read_dvfs_file(id)
        elif self.file_type == 'simpleperf':
            # Read and parse the simpleperf file
            X = self.read_simpleperf_file(id)
            
        else:
            raise ValueError('Incorrect file type provided to the dataloader')
        
        X_std = X
        # Normalize
        if self.normalize:
            # X : Nchannel x num_data_points

            # Calculate mean of each channel
            mean_ch = torch.mean(X,dim=1)
            
            # Calculate std of each channel
            std_ch = torch.std(X,dim=1)
            floor_std_ch = torch.tensor([1e-12]*X.shape[0]) 
            std_ch = torch.maximum(std_ch,floor_std_ch) # To avoid the division by zero error
            
            # Normalize
            X_std = (X - torch.unsqueeze(mean_ch,1))/torch.unsqueeze(std_ch,1)
            
            ## Testing the normalization module
            # print(mean_ch.shape,std_ch.shape, floor_std_ch.shape)
            # print(X.shape, X_std.shape)
            # print(torch.max(X),torch.min(X))
            # print(torch.max(X_std),torch.min(X_std))
            # print(torch.std(X,1))
            # print(torch.mean(X,1))
            # print(torch.std(X_std,1))
            # print(torch.mean(X_std,1))
            
        # Return the dvfs/hpc tensor (X_std), the corresponding label (y), and the corresponding file path that contains the name (id)
        return X_std,y,id

    def read_simpleperf_file(self, f_path):
        '''
        Parses the simpleperf file at path = fpath, parses it and returns a tensor of shape (Nchannels,T)
        '''
        # Extract the rn value to select the perf channels
        rn_obj = re.search(r'.*_(rn\d*).txt', f_path, re.M|re.I)
        if rn_obj:
            rn_val = rn_obj.group(1).strip()

        # Dict storing the regex patterns for extracting the performance counter channels
        perf_channels = {
                    'rn1' : [r'(\d*),cpu-cycles',r'(\d*),instructions', r'(\d*),raw-bus-access'], 
                    'rn2' : [r'(\d*),branch-instructions',r'(\d*),branch-misses', r'(\d*),raw-mem-access'], 
                    'rn3' : [r'(\d*),cache-references',r'(\d*),cache-misses', r'(\d*),raw-crypto-spec'],
                    'rn4' : [r'(\d*),bus-cycles',r'(\d*),raw-mem-access-rd', r'(\d*),raw-mem-access-wr']
                }

        # Store the parsed performance counter data. Each item is one collection point constaining 3 performance counter [perf1,perf2,perf3]
        # perf_list = [[perf1_value1,perf2_value1,perf3_value1], [perf1_value2,perf2_value2,perf3_value2], [], ....]
        perf_list = [] 

        with open(f_path) as f:
            for line in f:
                ######################### Perform a regex search on this line #########################
                # Every new collection point starts with "Performance counter statistics,". We use this as a start marker.
                startObj = re.search(r'(Performance counter statistics)', line, re.M|re.I)
                if startObj: # A new collection point has started. Start an empty list for this collection point.
                    collection_point = []

                # Parse the first performance counter
                perf1Obj = re.search(perf_channels[rn_val][0], line, re.M|re.I)
                if perf1Obj: 
                    collection_point.append(float(perf1Obj.group(1).strip()))
                    
                # Parse the second performance counter
                perf2Obj = re.search(perf_channels[rn_val][1], line, re.M|re.I)
                if perf2Obj: 
                    collection_point.append(float(perf2Obj.group(1).strip()))
                
                # Parse the third performance counter
                perf3Obj = re.search(perf_channels[rn_val][2], line, re.M|re.I)
                if perf3Obj: 
                    collection_point.append(float(perf3Obj.group(1).strip()))

                # Every collection point ends with "Total test time" followed by the timestamp of collection
                endObj = re.search(r'Total test time,(.*),seconds', line, re.M|re.I)
                if endObj: # End of collection point reached
                    
                    collection_point = [float(endObj.group(1).strip())] + collection_point
                    
                    # Also perform a sanity check to make sure all the performance counters for the collection point are present.
                    if len(collection_point) != 4:
                        raise ValueError("Missing performance counter collection point")
                    else:
                        # We have all the data points. Add it to the list.
                        perf_list.append(collection_point)

        # Convert the list to a tensor (Shape : Num_data_points x Num_channels)
        perf_tensor = torch.tensor(perf_list, dtype=torch.float32)
        # Transpose the tensor to shape : Num_channels x Num_data_points
        perf_tensor_transposed = torch.transpose(perf_tensor, 0, 1)
        # Ignore the first channel (it is the time channel)
        perf_tensor_transposed = perf_tensor_transposed[1:]

        return perf_tensor_transposed
                    

    def read_dvfs_file(self, f_path):
        '''
        Reads the dvfs file at path = fpath, parses it, and returns a tensor of shape (Nchannels,T)
        '''
        # List containing the parsed lines of the file. Each element is one line of the file.
        dvfs_list = []

        with open(f_path) as f:
            next(f) #Skip the first line containing the timestamp

            for line in f:
                try:
                    dvfs_list.append(list(map(float,line.split())))
                except: # Skip the lines that throw error (Usually the last line)
                    print("************************************************** Parsing issue in DVFS ******************************************************************************")
                    print(f_path)
                        
        # Convert the list to a tensor (Shape : Num_data_points x Num_channels)
        try:
            dvfs_tensor = torch.tensor(dvfs_list[:-1]) # Skipping the last line as it may be incomplete in some files
        except:
            print("************************************************** Missing data ******************************************************************************")
            print(f_path)
            print([i for (i,item) in enumerate(dvfs_list) if len(item) != 17])

        # Transpose the tensor to shape : Num_channels x Num_data_points
        dvfs_tensor_transposed = torch.transpose(dvfs_tensor, 0, 1)

        # Substract successive values in chn_idx-13 (rx_bytes) . This channel has cumulative data. Need to convert it into individual points..
        dvfs_tensor_transposed[13] = dvfs_tensor_transposed[13] - torch.cat([torch.tensor([dvfs_tensor_transposed[13][0]]), dvfs_tensor_transposed[13][:-1]])
        # Substract successive values in chn_idx-14 (tx_bytes) . This channel has cumulative data. Need to convert it into individual points..
        dvfs_tensor_transposed[14] = dvfs_tensor_transposed[14] - torch.cat([torch.tensor([dvfs_tensor_transposed[14][0]]), dvfs_tensor_transposed[14][:-1]])
        # Discard the first 2 channels (they contain only the timestamps)
        filtered_dvfs_tensor = dvfs_tensor_transposed[2:]
    
        return filtered_dvfs_tensor    
        

# Returns the dataloader object that can be used to get batches of data
def get_dataloader(opt, partition, labels, custom_collate_fn, validation_present, normalize_flag = True, file_type = None, N=None):
    '''
    Input: -partition = {'train' : [file_path1, file_path2, ..],
                            'test' : [file_path1, file_path2, ..],
                            'val' : [file_path1, file_path2]}
                            
           -labels : {file_path1 : 0, file_path2: 1, ...}  (Benigns have label 0 and Malware have label 1)
           
           -custom_collate_fn : Custom collate function object (Resamples and creates a batch of spectrogram B,T_chunk,Nch,H,W)

           -validation_present : True if 'val' split is present in the parition dict. False otherwise.           
           
           -N  : [num_training_samples, num_validation_samples, num_testing_samples]
                  If N is specified, then we are selecting a subset of files from the dataset 

           -normalize_flag : Will normalize the data if set to True. [Should be set to True for 'dvfs' and False for 'simpleperf'] 

           -file_type : 'dvfs' or 'simpleperf' -> Different parsers used for each file_type

    Output: Dataloader object for training, validation, and test data.
    '''
    
    # Initialize the custom dataset class for training, validation, and test data
    ds_train_full = arm_telemetry_data(partition, labels, split='train', file_type= file_type, normalize=normalize_flag)
    
    if validation_present:
        ds_val_full = arm_telemetry_data(partition, labels, split='val', file_type= file_type, normalize=normalize_flag)
    
    ds_test_full = arm_telemetry_data(partition, labels, split='test', file_type= file_type, normalize=normalize_flag)

    # Check if you are using a subset of the complete dataset
    if N is not None:
        # You are using a subset of the complete dataset
        print(f'[Info] ############### Using Subset : Num_train = {N[0]}, Num_val = {N[1]}, Num_test = {N[2]} ##################')
        if len(N) != 3:
            raise NotImplementedError('Size of Array should be 3')

        if N[0] > ds_train_full.__len__():
            raise NotImplementedError(f"More samples than present in DS. Demanded : {N[0]} | Available: {ds_train_full.__len__()}")

        if validation_present:
            if N[1] > ds_val_full.__len__():
                raise NotImplementedError(f'More samples than present in DS. Demanded : {N[1]} | Available: {ds_val_full.__len__()}')

        if N[2] > ds_test_full.__len__():
            raise NotImplementedError(f'More samples than present in DS. Demanded : {N[2]} | Available: {ds_test_full.__len__()}')

        
        indices = torch.arange(N[0])
        ds_train = data_utils.Subset(ds_train_full, indices)

        if validation_present:
            indices = torch.arange(N[1])
            ds_val = data_utils.Subset(ds_val_full, indices)

        indices = torch.arange(N[2])
        ds_test = data_utils.Subset(ds_test_full, indices)

    else: # Using the complete dataset
        ds_train = ds_train_full
        
        if validation_present:
            ds_val = ds_val_full
        
        ds_test = ds_test_full

    # Create the dataloader object for training, validation, and test data
    trainloader = torch.utils.data.DataLoader(
        ds_train,
        num_workers=opt.num_workers,
        batch_size=opt.train_batchsz,
        collate_fn=custom_collate_fn,
        shuffle=opt.train_shuffle,
    )

    if validation_present:    
        valloader = torch.utils.data.DataLoader(
            ds_val,
            num_workers=opt.num_workers,
            batch_size=opt.test_batchsz,
            collate_fn=custom_collate_fn,
            shuffle=opt.test_shuffle,
            sampler = torch.utils.data.SequentialSampler(ds_val)
        )

    
    testloader = torch.utils.data.DataLoader(
        ds_test,
        num_workers=opt.num_workers,
        batch_size=opt.test_batchsz,
        collate_fn=custom_collate_fn,
        shuffle=opt.test_shuffle,
        sampler = torch.utils.data.SequentialSampler(ds_test)
    )

    if not validation_present:
        valloader = None

    return trainloader, valloader, testloader

# Returns the dataloader object that can be used to get batches of data
def get_dataloader_only_testloader(opt, partition, labels, custom_collate_fn, validation_present, normalize_flag = True, file_type = None, N=None):
    '''
    Returns only the dataloader for the testsamples.

    Input: -partition = {'train' : [file_path1, file_path2, ..],
                            'test' : [file_path1, file_path2, ..],
                            'val' : [file_path1, file_path2]}
                            
           -labels : {file_path1 : 0, file_path2: 1, ...}  (Benigns have label 0 and Malware have label 1)
           
           -custom_collate_fn : Custom collate function object (Resamples and creates a batch of spectrogram B,T_chunk,Nch,H,W)

           -N  : [num_training_samples, num_validation_samples, num_testing_samples]
                  If N is specified, then we are selecting a subset of files from the dataset 

           -normalize_flag : Will normalize the data if set to True. [Should be set to True for 'dvfs' and False for 'simpleperf'] 

           -file_type : 'dvfs' or 'simpleperf' -> Different parsers used for each file_type

    Output: Dataloader object for training, validation, and test data.
    '''
    
    ds_test_full = arm_telemetry_data(partition, labels, split='test', file_type= file_type, normalize=normalize_flag)

    # Check if you are using a subset of the complete dataset
    if N is not None:
        # You are using a subset of the complete dataset
        print(f'[Info] ############### Using Subset : Num_train = {N[0]}, Num_val = {N[1]}, Num_test = {N[2]} ##################')
        if len(N) != 3:
            raise NotImplementedError('Size of Array should be 3')

        if N[2] > ds_test_full.__len__():
            raise NotImplementedError('More samples than present in DS')

        indices = torch.arange(N[2])
        ds_test = data_utils.Subset(ds_test_full, indices)

    else: # Using the complete dataset
        ds_test = ds_test_full

    
    testloader = torch.utils.data.DataLoader(
        ds_test,
        num_workers=opt.num_workers,
        batch_size=opt.test_batchsz,
        collate_fn=custom_collate_fn,
        shuffle=opt.test_shuffle,
        sampler = torch.utils.data.SequentialSampler(ds_test)
    )

    # Returning in a specific format due to dependencies
    return None, None, testloader


class custom_collator(object):
    def __init__(self, collection_duration=40, chunk_time = 1, truncated_duration = 30, 
                custom_num_datapoints = 150000, resampling_type = 'custom',
                reduced_frequency_size = 10, reduced_time_size = 5, reduced_feature_flag = False, n_fft = 1024, file_type = None, histogram_bins = 32, resample=True):

        self.cd = collection_duration # Duration for which data is collected
        self.chunk_time = chunk_time # Window size (in s) over which the spectrogram will be calculated
        
        # Parameters for truncating the dvfs time series
        self.truncated_duration = truncated_duration # Consider the first truncated_duration seconds of the iteration
        
        # Parameters for resampling DVFS
        self.custom_num_datapoints = custom_num_datapoints # Number of data points in the resampled time series
        self.resampling_type = resampling_type # Type of resampling. Can take one of the following values: ['max', 'min', 'custom']
        self.resample = resample # Whether or not to resample. Default : True

        # Parameters for feature reduction (for DVFS file_type)
        self.reduced_frequency_size = reduced_frequency_size # dimension of frequency axis after dimensionality reduction
        self.reduced_time_size = reduced_time_size # dimension of time axis after dimensionality reduction
        self.reduced_feature_flag = reduced_feature_flag # If True, then we perform feature reduction. Defaule is False.
        self.n_fft = n_fft # Order of fft for stft

        # For selecting file_type : "dvfs" or "simpleperf"
        self.file_type = file_type

        # Feature engineering parameters for simpleperf files
        self.histogram_bins = histogram_bins

    def __call__(self, batch):
        '''
        Takes a batch of files, outputs a tensor of of batch, the corresponding labels, and the corresponding file paths
        - If reduced_feature_flag is False, then will return a list instead of a stacked tensor, for both dvfs and simpleperf
        '''
        if self.file_type == "dvfs":
            # batch_dvfs : [iter1, iter2, ... , iterB]  (NOTE: iter1 - Nchannels x T1 i.e. Every iteration has a different length. Duration of data collection is the same. Sampling frequency is different for each iteration)
            # batch_labels : [iter1_label, iter2_label, ...iterB_label]
            # batch_paths : [iter1_path, iter2_path, ...iterB_path]
            batch_dvfs, batch_labels, batch_paths = list(zip(*batch))

            ############################################################## Truncating module ###################################################################### 
            """ Truncates each sample to the first truncated_duration seconds of the iteration """    

            #######################################################################################################################################################
            
            if self.resample:
                # Resample so that each iteration in the batch has the same number of datapoints
                resampled_batch_dvfs, target_fs = self.resample_dvfs(batch_dvfs)
            else:
                resampled_batch_dvfs = batch_dvfs 
            # ######################## Testing the resampling module #######################################
            ## To plot the original and the resampled data (for verifying the resampling)
            # print('********************* Testing the resampling module *********************')
            # for i in range(15):
            #     chn_idx = i
            #     btch_idx = 0
            #     fig, axs = plt.subplots(2)
            #     axs[0].plot([i for i rin range(len(batch_dvfs[btch_idx][chn_idx]))],batch_dvfs[btch_idx][chn_idx])
            #     axs[1].plot([i for i in range(len(resampled_batch_dvfs[btch_idx][chn_idx]))],resampled_batch_dvfs[btch_idx][chn_idx])
            #     plt.savefig('pics/resampling_channel'+str(i)+'_.png', dpi=300)
            #     plt.close()
            ################################################################################################


            # Perform feature reduction on the batch of resampled dataset, so that number of features for every sample = 40
            if self.reduced_feature_flag: #If reduced_feature_flag is set to True, then perform feature reduction
                batch_tensor = self.perform_feature_reduction(resampled_batch_dvfs, target_fs)
            
            else: # Just pass the resampled dvfs data
                batch_tensor = resampled_batch_dvfs
            # print(batch_tensor.shape)
        
            ################################################################################################

        elif self.file_type == "simpleperf":
            # batch_hpc : [iter1, iter2, ... , iterB]  (NOTE: iter1 - Nchannels x T1 i.e. Every iteration has a different length. Duration of data collection is the same. Sampling frequency is different for each iteration)
            # batch_labels : [iter1_label, iter2_label, ...iterB_label]
            # batch_paths : [iter1_path, iter2_path, ...iterB_path]
            batch_hpc, batch_labels, batch_paths = list(zip(*batch))

            if self.reduced_feature_flag:
                # Stores the dimension reduced hpc for each patch
                reduced_batch_hpc = []

                # Divide the individual variates of the tensor into histogram_bins number of intervals. And sum over the individual intervals to form feature size of 32 for each variate.
                for hpc_iter_tensor in batch_hpc:
                    ## hpc_intervals : [chunks of size - Nchannels x self.histogram_bins]
                    hpc_intervals = torch.split(hpc_iter_tensor, self.histogram_bins, dim=1)
                    
                    # Take sum along the time dimension for each chunk to get chunks of size -  Nchannels x 1
                    sum_hpc_intervals = [torch.sum(hpc_int,dim=1, keepdim=False) for hpc_int in hpc_intervals]
                    
                    # Concatenate the bins to get the final feature tensor
                    hpc_feature_tensor = torch.cat(sum_hpc_intervals, dim=0)
                    
                    reduced_batch_hpc.append(torch.unsqueeze(hpc_feature_tensor, dim=0)) # Adding one dimension for channel [for compatibility purpose]. N_Channel = 1 in this case.

                batch_tensor = torch.stack(reduced_batch_hpc, dim=0)
            
            else:
                batch_tensor = batch_hpc # NOTE: This is not a tensor. It is a list of the iterations.

        return batch_tensor, torch.tensor(batch_labels), batch_paths


    def perform_feature_reduction(self, resampled_batch_dvfs, resampled_fs):
        '''
        Function to reduce the number of features for every iteration to 'reduced_feature_size'
            Input:  -resampled_batch_dvfs - list of resampled iterations [iter1, iter2, ... , iterB] where shape of iter1 = Nch x Num_data_points
                    -resampled_fs - sampling frequency of iteration (same for all the iterations)
            Output : - feature_reduced_batch_tensor - Tensor of shape (B, Nch, reduced_feature_size)
        '''
        # Stores the dimension reduced dvfs
        reduced_batch_dvfs = []

        for idx, iter in enumerate(resampled_batch_dvfs):
            
            # print(f'Shape of the multivariate time series (without dimensionality reduction) : {iter.shape}')
            
            ###################################################### FFT based feature reduction ######################################################
            # Perform windowed FFT on each iteration (shape: Nch, N_Freq_steps, N_time_steps, 2) Ref : https://pytorch.org/docs/stable/generated/torch.stft.html
            # Last dimension contains the real and the imaginary part
            stft_transform = torch.stft(input=iter, n_fft = self.n_fft, return_complex=False)
            # print(f'Shape of the stft : {stft_transform.shape}')

            # Get the magnitude from the stft (shape : Nch, N_Freq_steps, N_time_steps)
            stft_transform = torch.sqrt((stft_transform**2).sum(-1))
            Nch,_,_ = stft_transform.shape
            # print(f'Shape of the stft magnitude : {stft_transform.shape}')

            channel_dimension_reduced = []
            # Perform PCA on the time axis to reduce the number of time
            for channel in stft_transform:
                # Current axis orientation is (frequency, time). Perform dimensionality reduction on frequency. So swap the orientation. 
                channel = np.transpose(channel)
                # Orientation is now (time,frequency)
                # print(f"- Axis orientation is (time, frequency) : {channel.shape}")

                # Initialize the PCA. Reduce the frequency dimension to self.reduced_frequency_size
                pca = PCA(n_components=self.reduced_frequency_size)
                frequency_reduced = pca.fit_transform(channel)
                
                # print(f"- Shape of frequency reduced tensor (time,reduced_frequency_size) : {frequency_reduced.shape}")
                # print(f"  - Variance : {pca.explained_variance_ratio_}")
                # print(f"  - Sum of Variance : {sum(pca.explained_variance_ratio_)}")
                
                # Current axis orientation is (time,frequency). Perform dimensionality reduction on time. So swap the orientation (frequency,time). 
                frequency_reduced = np.transpose(frequency_reduced)
                # print(f"- Shape of transposed frequency reduced tensor (reduced_frequency_size,time) : {frequency_reduced.shape}")
                
                # Initialize the PCA. Reduce the time dimension to self.reduced_time_size
                pca = PCA(n_components=self.reduced_time_size)
                time_frequency_reduced = pca.fit_transform(frequency_reduced)

                # print(f"- Shape of time-reduced frequency-reduced tensor (reduced_frequency_size,reduced_time_size) : {time_frequency_reduced.shape}")
                # print(f"  - Variance : {pca.explained_variance_ratio_}")
                # print(f"  - Sum of Variance : {sum(pca.explained_variance_ratio_)}")
               
                # Current axis orientation is (frequency,time).  Change orientation to (time, frequency)
                time_frequency_reduced = np.transpose(time_frequency_reduced)
                # print(f"- Shape of time-reduced frequency-reduced tensor (reduced_time_size, reduced_frequency_size) : {time_frequency_reduced.shape}")
                
                # Flatten the array (Shape : reduced_frequency_size*reduced_time_size)
                time_frequency_reduced = time_frequency_reduced.flatten()
                # print(f"- Shape of flattened time-reduced frequency-reduced tensor (reduced_frequency_size*reduced_time_size) : {time_frequency_reduced.shape}")
                
                channel_dimension_reduced.append(time_frequency_reduced)
            
            channel_dimension_reduced_tensor = np.stack(channel_dimension_reduced, axis=0)
            # print(f"Shape of channel_dimension_reduced_tensor (Nch, reduced_frequency_size*reduced_time_size) : {channel_dimension_reduced_tensor.shape}")

            #############################################################################################################################################

            reduced_batch_dvfs.append(channel_dimension_reduced_tensor)
        
        # Shape : B,Nch,reduced_feature_size
        reduced_batch_dvfs_tensor = np.stack(reduced_batch_dvfs, axis=0)
        # print(f"Shape of final batch tensor (B, Nch, reduced_frequency_size*reduced_time_size) : {reduced_batch_dvfs_tensor.shape}")
        # sys.exit()
        return torch.tensor(reduced_batch_dvfs_tensor)

    def truncate_dvfs(self, batch_dvfs):
        """ 
        Truncate the dvfs time series to truncated duration 
        params:
            - batch_dvfs: list of dvfs time series
            - List of truncated dvfs time series
        """
        # Get the number of datapoints for each iteration in the batch
        num_data_points = [b_dvfs.shape[1] for b_dvfs in batch_dvfs]
        # print(f"- Number of data points per iteration : {num_data_points}")
        
        # Calculate the sampling frequency for each iteration in the batch
        fs_batch = [ndp//self.cd for ndp in num_data_points]
        # print(f"- Sampling frequency per iteration : {fs_batch}")
        
        # Calculate the number of datapoints for each iteration based on the truncated duration
        pass

    def resample_dvfs(self, batch_dvfs):
        '''
        Function to resample a batch of dvfs iterations
            -Input: batch of iterations
            -Output : List of resampled batch of iterations, target_frequency (frequency of the resampled batch)
        '''

        # Get the number of datapoints for each iteration in the batch
        num_data_points = [b_dvfs.shape[1] for b_dvfs in batch_dvfs]
        # print(f"- Number of data points per iteration : {num_data_points}")
        
        # Calculate the sampling frequency for each iteration in the batch
        fs_batch = [ndp//self.cd for ndp in num_data_points]
        # print(f"- Sampling frequency per iteration : {fs_batch}")
        
        # Get the max and min sampling frequency (Will be used for resampling)
        max_fs, min_fs= [max(fs_batch), min(fs_batch)]
        # print(f"- Maximum and Minimum sampling frequency in the batch : {max_fs,min_fs}")
        
        # Check what kind of resampling needs to be performed, and set the corresponding target frequency
        if(self.resampling_type == 'max'):
            target_fs = max_fs
        elif(self.resampling_type == 'min'):
            target_fs = min_fs
        elif(self.resampling_type == 'custom'):
            target_fs = self.custom_num_datapoints/self.cd
        else:
            raise NotImplementedError('Incorrect resampling argument provided')
        
        # print(f"- Sampling mode : {self.resampling_type} | Target sampling frequency : {target_fs}")

        # Resample each iteration in the batch using the target_fs
        resampled_batch_dvfs = []

        for idx,iter in enumerate(batch_dvfs):
            # print(f" -- Shape of the iteration {idx} pre resampling : {iter.shape}")
            # Initialize the resampler
            resample_transform = torchaudio.transforms.Resample(orig_freq=fs_batch[idx], new_freq=target_fs, lowpass_filter_width=6, resampling_method='sinc_interpolation')

            r_dvfs = resample_transform(iter)
            # print(f" -- Shape of the iteration {idx} post sampling : {r_dvfs.shape}")
            
            resampled_batch_dvfs.append(r_dvfs)

        return resampled_batch_dvfs, target_fs

class dataset_generator:
    def __init__(self, filter_values, dataset_type):
        """
        Dataset generator : Downloads the dataset from the dropbox.

        params:
            - filter_values : Filter values for the logcat files
                            Format : [runtime_per_file, num_logcat_lines_per_file, freq_logcat_event_per_file]
            - dataset_type : Type of dataset that you want to create
                            Can take one of the following values : ["std-dataset","cd-dataset","bench-dataset"]
            
        """
        self.filter_values = filter_values
        self.dataset_type = dataset_type

        # Root directory of xmd
        self.root_dir_path = os.path.dirname(os.path.realpath(__file__)).replace("/src","")

        #################################################### Dataset info ######################################################## 
        # Information about the std dataset and the cd dataset
        self.std_cd_dataset_info = {
                "std_malware":{"dbx_path":"/results_android_zoo_malware_all_rerun", "app_type":"malware", "dtype":"std_malware"},
                "std_benign":{"dbx_path":"/results_android_zoo_benign_with_reboot", "app_type":"benign", "dtype":"std_benign"},
                "cd_malware":{"dbx_path":"/results_android_zoo_unknown_malware", "app_type":"malware", "dtype":"cd_malware"},
                "cd_benign":{"dbx_path":"/results_android_zoo_unknown_benign", "app_type":"benign", "dtype":"cd_benign"}
                }
        
        # Information about the bench dataset. Benchmark logs are divided over three different folders.
        self.bench_dataset_info={"bench1":"/results_benchmark_benign_with_reboot_using_benchmark_collection_module",
                    "bench2":"/results_benchmark_benign_with_reboot_using_benchmark_collection_module_part2",
                    "bench3":"/results_benchmark_benign_with_reboot_using_benchmark_collection_module_part3"}
        ###########################################################################################################################

        ############################### Generating black list for malware apks in the std-dataset #################################
        vt_malware_report_path = os.path.join(self.root_dir_path, "res", "virustotal", "hash_virustotal_report_malware")
        
        # If the black list already exists, then it will load the previous black list. To generate the new blacklist, delete
        # the not_malware_hashlist at "xmd/res/virustotal"
        self.std_dataset_malware_blklst = self.get_black_list_from_vt_report(vt_malware_report_path)
        ###########################################################################################################################

    def get_black_list_from_vt_report(self, vt_malware_report_path):
        """
        Just for the std-dataset: Gets the list of malware apks with 0 or 1 vt positives. We will not process the logs 
        from these apks as malware.

        params:
            - vt_malware_report_path : Path of the virustotal report of the malware
         
        Output:
            - not_malware : List of hashes of apks with 0 or 1 vt positive
        """
        # Location where the not_malware list is stored
        not_malware_list_loc = os.path.join(self.root_dir_path,"res","virustotal","not_malware_hashlist")

        # Check if the not_malware_hashlist is already created. If yes, then return the previous list
        if os.path.isfile(not_malware_list_loc):
            with open(not_malware_list_loc, 'rb') as fp:
                not_malware = pickle.load(fp)
            return not_malware
        
        # Load the virustotal report
        with open(file=vt_malware_report_path) as f:
            report = json.load(f)

        # List storing the malware with 0 or 1 positive results
        not_malware = []

        # Parsing the virustotal report
        malware_details = {}
        for hash, hash_details in report.items():
            # Store the malware hash, positives, total, percentage positive
            malware_details[hash] = {'positives':hash_details['results']['positives'],
                                    'total':hash_details['results']['total'],
                                    'percentage_positive':round((float(hash_details['results']['positives'])/float(hash_details['results']['total']))*100,2),
                                    'associated_malware_families':[avengine_report['result'] for _,avengine_report in hash_details['results']['scans'].items() if avengine_report['result']]}
            
            # Identify the malware apks with 0 or 1 vt_positives
            if int(hash_details['results']['positives']) == 0 or int(hash_details['results']['positives']) == 1 :
                not_malware.append(hash)

        # Save the not_malware list as a pickled file
        with open(not_malware_list_loc, 'wb') as fp:
            pickle.dump(not_malware, fp)
            
        return not_malware

    @staticmethod
    def extract_hash_from_filename(file_list):
        """
        Extract hashes from the shortlisted files [Used for counting the number of apks for the std-dataset and the cd-dataset].
        params:
            - file_list : List of files from which the hashes needs to be extracted
        Output:
            - hash_list : List of hashes that is extracted from the file list
        """
        # To store the list of hashes
        hash_list = []

        for fname in file_list:
            # Extract the hash from the filename
            hashObj = re.search(r'.*_(.*).apk.*', fname, re.M|re.I)
            hash_ = hashObj.group(1)

            if hash_ not in hash_list:
                hash_list.append(hash_)

        return hash_list

    def filter_shortlisted_files(self, file_list):
        """
        Filters out the blacklisted files from the shortlisted files. 
        [Used for filtering out the blacklisted files from the malware apks in the std-dataset].
        params:
            - file_list : List of file names on which the filter needs to be applied
        
        Output:
            - filtered_file_list : List of file names after the filter has been applied
        """
        # For tracking the number of files that are filtered out
        num_files_filtered_out = 0
        
        # Storing the file names post filter
        filtered_file_list = []

        for fname in file_list:
            # Extract the hash from the filename
            hashObj = re.search(r'.*_(.*).apk.*', fname, re.M|re.I)
            hash_ = hashObj.group(1)

            # If the hash is not in the blklst, then add it to the filtered list
            if hash_ not in  self.std_dataset_malware_blklst:
                filtered_file_list.append(fname)
            else:
                num_files_filtered_out += 1

        print(f"- Number of malware files that are filtered out: {num_files_filtered_out}")
        return filtered_file_list

    def create_shortlisted_files(self, parser_info_loc, apply_filter = True):
        '''
        Function to create a list of shortlisted files, based on logcat
        Input: 
            - parser_info_loc : Location of the parser_info file
            - apply_filter : If True, then applies the filter. Else, no filter (in the case of benchmark benign files)
            
        Output: 
            - shortlisted_files : List containing the dropbox location of the shortlisted files
            - logcat_attributes_list : List containing the corresponding logcat attributes of the shortlisted files
        '''
        # List of locations of the shortlisted files [Output of this method]
        shortlisted_files = []
        # List of corresponding logcat attributes for the shortlisted files
        logcat_attributes_list = []

        # Load the JSON containing the parsed logcat info for each iteration of data collection (You need to run codes/dropbox_module.py to generate the file)
        with open(parser_info_loc,"r") as fp:
            data=json.load(fp)

        # Extracting the threshold values
        if apply_filter:
            # If cd-dataset or std-dataset, then apply the logcat filter
            runtime_thr, num_logcat_event_thr, freq_logcat_event_thr = self.filter_values
        else: 
            # No need to filter the benchmark dataset since benchmarks run to completion always
            runtime_thr, num_logcat_event_thr, freq_logcat_event_thr = [0,0,0]

        for apk_folder,value in data.items():
            # apk_folder = Path of apk logcat folder (Contains the apk name)
            # value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]

            for ind in range(value[0]): # Value[0] = number of logcat files for each apk. Each logcat file has its own dict.
                i = ind + 1 # For indexing into the corresponding dict in the list.
                
                for file_name,logcat_attributes in value[i].items():
                    # file_name = Name of the logcat file
                    # logcat_attributes = [avg_freq, num_logcat_lines, time_diff]

                    if((logcat_attributes[0] > freq_logcat_event_thr) and (logcat_attributes[1] > num_logcat_event_thr) and (logcat_attributes[2] > runtime_thr)):
                        # File satisfies all the threshold, add the full location of the file to the list
                        shortlisted_files.append(apk_folder+'/'+file_name) 
                        logcat_attributes_list.append([logcat_attributes[0],logcat_attributes[1],logcat_attributes[2]])

        return shortlisted_files, logcat_attributes_list

    ######################################## Helper methods to download the files from dropbox ########################################
    @staticmethod
    def create_dropbox_location(shortlisted_files, file_type):
        '''
        Function to create a list of dropbox locations and corresponding locations on the local machine
        from the shortlisted files based on the file_type (dvfs, logcat, simpleperf)
        Input :
                - shortlisted_files : Full dropbox locations of the logcat files of the shortlisted files
                - file_type : (dvfs, logcat, simpleperf)
                
        Output : 
                -shortlisted_files_mod (List of dropbox locations for the given file_type)
                -localhost_loc (List of corresponding file locations on the local host)
        '''

        shortlisted_files_mod = [] # Contains the location in dropbox
        localhost_loc = [] # Contains the location of the file in the local host

        for location in shortlisted_files:

            # Extract the iter, rn, and base_locations
            inputObj = re.search(r'(\/.*\/)logcat\/(.*logcat)(.*)', location, re.M|re.I)
            base_loc = inputObj.group(1)
            file_loc = inputObj.group(2)
            iter_rn = inputObj.group(3)
            
            # Extract the rn number [Will be used for separating the HPC data into buckets]
            rn_obj = re.search(r'.*\_(.*)\.txt', iter_rn, re.M|re.I)
            rn_num = rn_obj.group(1) # Will be one of the following : ['rn1','rn2','rn3','rn4']
            
            # Extract the apk hash [Will inject the hash into the file name to accurately track the apks the logs are collected from]
            hash_obj = re.search(r'.*_(.*)\.apk', base_loc, re.M|re.I)
            apk_hash = hash_obj.group(1)
            
            if file_type == 'dvfs':
                new_loc = base_loc+'dvfs/'+file_loc.replace('logcat','devfreq_data')+iter_rn # Dropbox location
                rem_loc = 'dvfs/'+apk_hash+'_'+file_loc.replace('logcat','devfreq_data')+iter_rn # Location on the local host
            elif file_type == 'logcat':
                new_loc = location
                rem_loc = 'logcat/'+apk_hash+'_'+file_loc+iter_rn
            
            # For performanc counter, we have 4 buckets : rn1, rn2, rn3, rn4. 
            elif file_type == 'simpleperf':
                new_loc = base_loc+'simpleperf/'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                
                # Create local location depending on the rn bucket
                if (rn_num == 'rn1'):
                    rem_loc = 'simpleperf/rn1/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                elif (rn_num == 'rn2'):
                    rem_loc = 'simpleperf/rn2/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                elif (rn_num == 'rn3'):
                    rem_loc = 'simpleperf/rn3/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                elif (rn_num == 'rn4'):
                    rem_loc = 'simpleperf/rn4/'+apk_hash+'_'+file_loc.replace('_logcat','')+iter_rn.replace('iter_','it')
                else :
                    raise ValueError('Parser returned an incorrect run number')     
            
            else: 
                raise ValueError('Incorrect file type provided')

            shortlisted_files_mod.append(new_loc)
            localhost_loc.append(rem_loc)

        return shortlisted_files_mod, localhost_loc
    
    def download_shortlisted_files(self, shortlisted_files, file_type, app_type):
        '''
        Function to download the shortlisted files from dropbox
        Input : -shortlisted_files (List containing the dropbox location of the shortlisted files)
                -file_type (the file type that you want to download : 'logcat', 'dvfs', or, 'simpleperf')
                -app_type ('malware' or 'benign')
               
        Output : Downloads the shortlisted files in <root_dir>/data/<dataset_type> 
                       
        '''
        # Create the download location on the local host
        base_download_location = os.path.join(self.root_dir_path, "data", self.dataset_type, app_type)
        os.system(f'mkdir -p {os.path.join(base_download_location, file_type)}')

        # Get the dropbox api key
        with open(os.path.join(self.root_dir_path,"src","dropbox_api_key")) as f:
            access_token = f.readlines()[0]

        # Authenticate with Dropbox
        print('Authenticating with Dropbox...')
        dbx = dropbox.Dropbox(access_token)
        print('...authenticated with Dropbox owned by ' + dbx.users_get_current_account().name.display_name)

        # If file_type is simpleperf then create rn bucket folders for each of them
        if (file_type == 'simpleperf'):
            os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn1')}")
            os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn2')}")
            os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn3')}")
            os.system(f"mkdir -p {os.path.join(base_download_location, file_type, 'rn4')}")

        # Create the dropbox location for the give file_type from the shortlisted_files
        dropbox_location, localhost_loc = dataset_generator.create_dropbox_location(shortlisted_files, file_type)

        # Counter to see how many files were not downloaded
        not_download_count = 0

        # Start the download
        for i, location in enumerate(dropbox_location):
            try:
                # print(f'-------Dropbox location : {location}')
                print(f'******* Local host location :{os.path.join(base_download_location, localhost_loc[i])} *******')
                download(dbx, location, os.path.join(base_download_location, localhost_loc[i]))
            except:
                not_download_count+=1
                traceback.print_exc()
                print(f'File not downloaded : Count = {not_download_count}')

        # Print the total files not downloaded
        print(f" ******************* Total files not downloaded : {not_download_count} *******************")
        

    ###################################################################################################################################
    def count_number_of_apks(self):
        """
        Count the number of apks (hashes) in the benign and malware file_list.
        params:
            - file_list: List of file names (including location)

        Output: 
            - num_apk_benign, num_apk_malware : Number of benign and malware apks
        """

        shortlisted_files_benign,shortlisted_files_malware = self.generate_dataset(download_file_flag=False)

        # Get the hash_list for benign and malware
        hashlist_benign = dataset_generator.extract_hash_from_filename(shortlisted_files_benign)
        hashlist_malware = dataset_generator.extract_hash_from_filename(shortlisted_files_malware)

        return len(hashlist_benign), len(hashlist_malware)

    
    def generate_dataset(self, download_file_flag):
        """
        Generates the dataset (benign,malware) based on the dataset_type and filter_values
        params:
            - download_file_flag : If True, then will download all the shortlisted files

        Output:
            - Generated dataset at the specified location
            - returns shortlisted_files_benign, shortlisted_files_malware (Corresponding dvfs and simpleperf files will be downloaded
                if download_file_flag is True.)
        """
        # 1. Create shortlisted files based on the logcat filter and dataset type
        if self.dataset_type == "std-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_std_benign.json")
            parser_info_malware = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_std_malware.json")

            # Create shortlisted files for benign and malware
            shortlisted_files_benign, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign, apply_filter=True)
            shortlisted_files_malware, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware, apply_filter=True)

            # Filter out the blacklisted files from the malware file list
            shortlisted_files_malware = self.filter_shortlisted_files(shortlisted_files_malware)

            
        elif self.dataset_type == "cd-dataset":
            # Get the location of the parser info files
            parser_info_benign = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_cd_benign.json")
            parser_info_malware = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_cd_malware.json")

            # Create shortlisted files for benign and malware
            shortlisted_files_benign, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign, apply_filter=True)
            shortlisted_files_malware, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware, apply_filter=True)

        elif self.dataset_type == "bench-dataset":
            # Get the location of the parser info files
            # Benchmark files are distributed over three different locations
            parser_info_benign1 = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_bench1.json")
            parser_info_benign2 = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_bench2.json")
            parser_info_benign3 = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_bench3.json")
            parser_info_malware = os.path.join(self.root_dir_path, "res/parser_info_files", f"parser_info_std_malware.json")

            # Create shortlisted files for benign and malware
            shortlisted_files_benign1, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign1, apply_filter=False)
            shortlisted_files_benign2, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign2, apply_filter=False)
            shortlisted_files_benign3, _ = self.create_shortlisted_files(parser_info_loc=parser_info_benign3, apply_filter=False)
            shortlisted_files_malware, _ = self.create_shortlisted_files(parser_info_loc=parser_info_malware, apply_filter=True)

            # Merge all the benchmark files to get one single list
            shortlisted_files_benign = shortlisted_files_benign1+shortlisted_files_benign2+shortlisted_files_benign3

            # Filter out the blacklisted files from the malware file list
            shortlisted_files_malware = self.filter_shortlisted_files(shortlisted_files_malware)

            
        else:
            raise(ValueError("Incorrect dataset type specified"))

        #################### Dataset Info ####################
        print(f"Information for the dataset : {self.dataset_type}")
        print(f"- Number of benign files : {len(shortlisted_files_benign)}")
        print(f"- Number of malware files : {len(shortlisted_files_malware)}")
        ###################################################### 
        
        
        # 2. Download the shortlisted files at <root_dir>/data/<dataset_type> 
        if download_file_flag:
            print("--------- Downloading all the shortlisted files ---------")
            # Downloading the shortlisted dvfs files [Needs to be executed only once to download the files]
            malware_dvfs_path =  self.download_shortlisted_files(shortlisted_files_malware, file_type= 'dvfs', app_type= 'malware')
            benign_dvfs_path =  self.download_shortlisted_files(shortlisted_files_benign, file_type= 'dvfs', app_type= 'benign')
            
            # Downloading the shortlisted performance counter files [Needs to be executed only once to download the files]
            malware_simpeperf_path =  self.download_shortlisted_files(shortlisted_files_malware, file_type= 'simpleperf', app_type= 'malware')
            benign_simpleperf_path =  self.download_shortlisted_files(shortlisted_files_benign, file_type= 'simpleperf', app_type= 'benign')

        return shortlisted_files_benign,shortlisted_files_malware


def main():
    # # STD-Dataset
    dataset_generator_instance = dataset_generator(filter_values= [15,50,2], dataset_type="std-dataset")
    # # CD-Dataset
    # dataset_generator_instance = dataset_generator(filter_values= [15,50,2], dataset_type="cd-dataset")
    # # Bench-Dataset
    # dataset_generator_instance = dataset_generator(filter_values= [15,50,2], dataset_type="bench-dataset")
    
    # dataset_generator_instance.generate_dataset(download_file_flag=True)
    print(dataset_generator_instance.count_number_of_apks())
    exit()

    # NOTE: There is a need to introduce randomness while generating the dataloader lists, so that we can perform cross validation
    
    
    # ################################# Trim the benign and malware to create a balanced dataset [Only for unknown testing] #################################
    # # Less number of files for malware so get number of malware files
    # num_malware = len(shortlisted_files_malware)
    # # Slice the benign list to have equal number of benign
    # shortlisted_files_benign = shortlisted_files_benign[:num_malware]
    
    # # Sanity check for balanced dataset
    # if len(shortlisted_files_malware) != len(shortlisted_files_benign):
    #     raise ValueError(f"Unbalanced dataset : benign - {len(shortlisted_files_benign)} | malware - {len(shortlisted_files_malware)}")
    # ####################################################################################################################################################### 
    
    ############################------------------------- Testing the dataloader components for fusing HPC and DVFS -------------------------############################
    # Folders containing the benign logs [HPC and DVFS]
    benign_dvfs_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/benign/dvfs'
    benign_simpleperf_rn1_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/benign/simpleperf/rn1'
    benign_simpleperf_rn2_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/benign/simpleperf/rn2'
    benign_simpleperf_rn3_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/benign/simpleperf/rn3'
    benign_simpleperf_rn4_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/benign/simpleperf/rn4'

    # Folders containing the malware logs [HPC and DVFS]
    malware_dvfs_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/malware/dvfs'
    malware_simpleperf_rn1_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/malware/simpleperf/rn1'
    malware_simpleperf_rn2_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/malware/simpleperf/rn2'
    malware_simpleperf_rn3_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/malware/simpleperf/rn3'
    malware_simpleperf_rn4_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/malware/simpleperf/rn4'

    # Folders to fuse when generating the HPC fusion predictions
    hpc_fusion_benign_path_list = [benign_simpleperf_rn1_path, benign_simpleperf_rn2_path, benign_simpleperf_rn3_path, benign_simpleperf_rn4_path]
    hpc_fusion_malware_path_list = [malware_simpleperf_rn1_path, malware_simpleperf_rn2_path, malware_simpleperf_rn3_path, malware_simpleperf_rn4_path]

    # Folders to fuse when generating the DVFS+HPC fusion predictions
    total_fusion_benign_path_list = [benign_dvfs_path]+hpc_fusion_benign_path_list
    total_fusion_malware_path_list = [malware_dvfs_path]+hpc_fusion_malware_path_list

    test_path_list = [benign_simpleperf_rn1_path, benign_simpleperf_rn2_path]

    # common_app_hashes_benign = get_common_apps([malware_simpleperf_rn1_path, malware_dvfs_path])
    # common_app_hashes_malware = get_common_apps(total_fusion_benign_path_list)
    
    # hash_list = get_hpc_dvfs_file_list(benign_simpleperf_rn2_path, benign_dvfs_path)

    base_location = "/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/"
    # b_matched_list, m_matched_list = create_matched_lists(base_location)

    all_datasets = create_all_datasets(base_location=base_location)
    # sys.exit()

    rn_val_minus_one = 2
    ds_train_full = arm_telemetry_data(partition = all_datasets[1][0][rn_val_minus_one], labels = all_datasets[1][1][rn_val_minus_one], split='train', file_type= 'simpleperf', normalize=False)
    

    tensor,label,path = ds_train_full.__getitem__(159)

    # print(tensor.shape)
    # print(tensor.type)
    # print(label)
    # print(path)
    # # print(tensor)

    mts = pd.DataFrame(data=np.transpose(tensor), columns=['perf1','perf2','perf3'], dtype= float)
    # mts['perf1']=mts['perf1'].astype(float)
    # mts['perf2']=mts['perf2'].astype(float)
    # mts['perf3']=mts['perf3'].astype(float)
    print(mts.dtypes)
    # Plot all the columns of the dataframe
    plt.figure()
    
    mts.plot(subplots=True,
             figsize = (10,10),
             legend = False )
    
    plt.savefig('/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/preprocess/test_hpc.png', dpi=300)
    plt.close('all')

    sys.exit()

    custom_collate_fn = custom_collator(truncated_duration=30, 
                reduced_feature_flag=True, 
                reduced_frequency_size=12, 
                reduced_time_size=4,
                file_type="simpleperf")
    
    trainloader = torch.utils.data.DataLoader(
        ds_train_full,
        num_workers=1,
        batch_size=5,
        collate_fn=custom_collate_fn,
        shuffle=False,
        sampler = torch.utils.data.SequentialSampler(ds_train_full)
    )
    iter_trainloader = iter(trainloader)
    batch_spec_tensor, labels, f_paths = next(iter_trainloader)
    print(batch_spec_tensor.shape, labels, f_paths)
    sys.exit()
    
    ############################-------------------------------------------------------------------------------------------------------------############################

    # Create the labels for the benign and malware files
    benign_dvfs_path = '/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/benign/dvfs'
    malware_dvfs_path ='/data/hkumar64/projects/arm-telemetry/conv-lstm/ConvLSTM_pytorch/data/data_updated/malware/dvfs'
    benign_label, malware_label = create_labels(benign_dvfs_path, malware_dvfs_path)

    # Create the labels dict by merging the benign and malware labels
    # Format : {file_path1 : 0, file_path2: 1, ...}
    labels = {**benign_label,**malware_label}

    # Create the partition dict
    partition = create_splits(benign_label, malware_label, partition_dist=None)

    ###################################### Testing the dataloader ######################################
    ds_train_full = arm_telemetry_data(partition, labels, split='train', normalize=True)
    # x,y = ds_train_full.__getitem__(0)
    # print(x.shape)

    # Intitialize the custom collator
    # Intitialize the custom collator
    custom_collate_fn = custom_collator(truncated_duration=30, 
                reduced_feature_flag=True, 
                reduced_frequency_size=12, 
                reduced_time_size=4)

    trainloader = torch.utils.data.DataLoader(
        ds_train_full,
        num_workers=1,
        batch_size=5,
        collate_fn=custom_collate_fn,
        shuffle=False,
        sampler = torch.utils.data.SequentialSampler(ds_train_full)
    )
    iter_trainloader = iter(trainloader)
    batch_spec_tensor, labels, f_paths = next(iter_trainloader)
    print(batch_spec_tensor.shape, labels, f_paths)
    print("---------------------------------------------------------")
    batch_spec_tensor, labels, f_paths = next(iter_trainloader)
    print(batch_spec_tensor.shape, labels, f_paths)
    ###################################################################################################
        



if __name__ == '__main__':
    main()