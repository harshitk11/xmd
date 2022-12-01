"""
Python script to train and evaluate all the first stage and the second stage models of XMD.
"""
import argparse
import datetime
# import stat
# import torch
from utils import Config
import os
import shutil
import numpy as np
import pickle
import collections
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

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

# class train_classifiers:
#     """
#     Class contains all the methods for training the GLOBL and the HPC base classifiers.
#     """

#     def __init__(self, args, xmd_base_folder_location) -> None:
#         # Channel names [in the same order as present in the dataset]
#         self.globl_channel_names = ['gpu_load','cpu_freq_lower_cluster','cpu_freq_higher_cluster', 'cpu_bw', 'gpu_bw',
#                     'kgsl-busmon','l3-cpu0', 'l3-cpu4', 'llccbw', 'memlat_cpu0',
#                     'memlat_cpu4', 'rx_bytes', 'tx_bytes', 'current_now', 'voltage_now']
#         self.args = args
#         self.xmd_base_folder_location = xmd_base_folder_location        
        
#         # Generate the list of candidate logcat threshold runtimes
#         self.logcat_rtimeThreshold_list = [i for i in range(0, args.collected_duration, args.step_size_logcat_runtimeThreshold)]

#         # Generate the list of truncated durations
#         self.truncatedRtimeList = [i for i in range(args.step_size_truncated_duration, args.collected_duration+args.step_size_truncated_duration, args.step_size_truncated_duration)]

#         # Read the json that stores the paths of all the feature engineered datasets
#         with open(os.path.join(xmd_base_folder_location,"res","featureEngineeredDatasetDetails","info.json")) as fp:
#             self.fDatasetAllDatasets = json.load(fp)

#         # Get the type of dataset [base classifiers are trained for std-dataset and bench-dataset]
#         self.dsGen_dataset_type, _ = dataloader_generator.get_dataset_type_and_partition_dist(dataset_type = args.dataset_type)
#         if (self.dsGen_dataset_type != "std-dataset") or (self.dsGen_dataset_type != "bench-dataset"):
#             raise ValueError("Incorrect dataset type specified for training the base classifiers")


        

#     def generate_all_base_classifier(self):
#         """
#         Generates base classifiers (DVFS and HPC) for all logcat_rtime_thresholds and truncated_durations.
#         """        
#         for logcatRtime in self.logcat_rtimeThreshold_list:
#             for truncatedRtime in self.truncatedRtimeList:
#                 # Train the GLOBL base classifiers
#                 self.train_base_classifiers_GLOBL(logcatRtime=str(logcatRtime), truncatedRtime=str(truncatedRtime))
#                 # Train the HPC base classifiers

        
        # For each logcat_rtime_threshold
        #       For each truncated duration
        #           Fetch the training channel channel bins
        #           train and tune the models using the training channel bins
        #           save the model 

    # def trainAndTuneRandomForest(self, channel_bins_train, labels_train):
    #     """
    #     Trains and generates the ensemble of classifiers, with one classifier per channel. For GLOBL telemetry we have 15 channels. For HPC telemetry, we have one channel.
    #     params : 
    #         - channels_bins_train : Shape - (N_channels, N_samples, feature_size) - Dataset for training
    #         - labels_train : Shape - (N_samples, 1) - Corresponding labels for training  
    #         - epochs :  Number of epochs of training
    #     """
    #     # Stores the trained classifier for each channel 
    #     channel_clf_list = []

    #     # Get the number of channels
    #     numChannels = channel_bins_train.shape[0]

    #     # Train random forest model for each channel
    #     for chn_indx in range(numChannels): 
    #         print(f"*************** Training model for channel number : {chn_indx} *************** ")

    #         # Training dataset for this channel
    #         X_train = channel_bins_train[chn_indx] 
            
            
    #         '''Training and Validation'''
    #         val_accuracy_list = [] # Stores the list of validation scores over all the epochs [Used for selecting the best model]
    #         best_chn_model = None # Stores the best model for this channel

    #         for _ in range(epochs):
    #             model,val_accuracy  = base_random_forest(channel_bins_train[chn_indx],labels_train,channel_bins_val[chn_indx],labels_val,n_estimators=100)
    #             val_accuracy_list.append(val_accuracy)

    #             # Save the best model based on accuracy
    #             if val_accuracy >= max(val_accuracy_list):
    #                 print(f"-------------- New best model found with accuracy : {val_accuracy} -------------- ")
    #                 best_chn_model = model


    #         '''Add the best model to the list of ensemble'''
    #         if best_chn_model is None:
    #             raise ValueError('Expected a model. Found None.')
    #         channel_clf_list.append(best_chn_model)

    #     if not os.path.isdir(os.path.join(results_path,'models')):
    #         os.makedirs(os.path.join(results_path,'models'))

    #     # Check if channel_bins_train has only one channel [that means that this is hpc dataset for a given rn]
    #     # We use a different naming convention for hpc individual classifiers
    #     if channel_bins_train.shape[0] == 1:
    #         # Write the channel_clf_list into a pickle file [so that you don't have to train the model every time]
    #         with open(os.path.join(results_path,'models',f'channel_models_{channels_of_interest[0]}'), 'wb') as fp:
    #             pickle.dump(channel_clf_list, fp)

    #     else:
    #         # Write the channel_clf_list into a pickle file [so that you don't have to train the model every time]
    #         with open(os.path.join(results_path,'models','channel_models'), 'wb') as fp:
    #             pickle.dump(channel_clf_list, fp)

    #     return channel_clf_list


    # def train_base_classifiers_GLOBL(self, logcatRtime, truncatedRtime):
    #     """
    #     Trains the GLOBL base classifiers for all the channels.

    #     params:
    #         - logcatRtime, truncatedRtime, self.dsGen_dataset_type: Used for fetching the training dataset 
    #     """
        
    #     ####################################################### Fetch the training datasets ####################################################### 
    #     channel_bins_path, labels_path, fPath_path = self.fDatasetAllDatasets[self.args.dataset_type][logcatRtime][truncatedRtime]["DVFS_individual"]["all"]["train"]
        
    #     # Fetch the training dataset 
    #     channel_bins_train = np.load(channel_bins_path)
    #     labels_train = np.load(labels_path)
    #     with open (fPath_path, 'rb') as fp:
    #         f_paths_train = np.array(pickle.load(fp), dtype="object")

    #     # Info about the dataset
    #     print(f"Shape of train channel bins (N_ch, N_samples, feature_size) and labels (N_samples,) : {channel_bins_train.shape, labels_train.shape}")
    #     print(f"Num Malware and Benign: {np.unique(labels_train, return_counts = True)}")
    #     ############################################################################################################################################

    #     ####################################################### Train and tune the model ###########################################################
    #     channel_clf_list = self.trainAndTuneRandomForest(channel_bins_train = channel_bins_train, labels_train = labels_train)
    #     ############################################################################################################################################ 
    #     # Save the model

    # def train_base_classifiers_HPC(self, logcatRtime, truncatedRtime):
    #     """
    #     Trains the HPC base classifiers for all the HPC groups.
    #     """
    #     # Fetch the training dataset.
    #     # Train and tune the model. 
    #     # Save the model.
    #     pass

    # def generate_stage1_decisions_GLOBL(self, globl_base_classifiers, classifier_input, labels):
    #     """
    #     Generates stage 1 decisions from the GLOBL base classifiers
    #     """
    #     # Loads the saved models
    #     # Calculate stage 1 decisions
    #     # Calculate the f1 score using the labels [if required]
    #     pass

    # def generate_stage1_decisions_HPC(self, hpc_base_classifiers, classifier_input, labels):
    #     """
    #     Generates stage 1 decisions from the HPC base classifiers
    #     """
    #     # Loads the saved models
    #     # Calculate stage 1 decisions
    #     # Calculate the f1 score using the labels [if required]
    #     pass

    # def generate_all_stage2_classifier():
    #     """
    #     Generates the MLP model for all logcat_rtime_thresholds and truncated_durations.
    #     """
    #     # For each logcat_rtime_threshold
    #     #       For each truncated duration
    #     #           Fetch the trainingSG channel channel bins
    #     #           train MLP
        
    #     pass    
    # def train_MLP(self, training_dataset, training_labels):
    #     """
    #     Trains the MLP model
    #     """
    #     #      generate_stage1_decisions_GLOBL() and generate_stage1_decisions_HPC()
    #     #      Train the MLP 
    #     pass


class baseRFmodel:
    """
    Object for the base Random Forest model. Tracks the different attributes of the base-classifiers (GLOBL or HPC).
    Contains all the methods for training, evaluation, saving, and loading the model.
    """
    
    def __init__(self, args, channelName=None, channelType=None) -> None:
        # Contains the parameters for training
        self.args = args
        # Channel for which the RF model is created
        self.channelName = channelName
        # Type of the channel : HPC or GLOBL
        self.channelType = channelType
        # Hyper parameter grid over which the tuning needs to be performed
        self.hyperparameterGrid = baseRFmodel.generate_hyperparameter_grid()
        
        ##################### Populated after training or loading a trained model ##################### 
        # List of validation scores for different hyperparameters
        self.validationScoreList = None
        # Stores the trained RF model
        self.trainedRFmodel = None
        # Stores the parameters of the best model
        self.bestModelParams = None
        ###############################################################################################

    def train(self, Xtrain, Ytrain):
        """
        Trains the Random Forest model. Tunes the trained model if modelTuneFlag is passed.
        
        params:
            - Xtrain: dataset (Nsamples, feature_size)
            - Ytrain: labels (Nsamples,) 
        """
        # Sanity check
        if len(Xtrain.shape)>2:
            raise ValueError(f"Shape of training data array incorrect : {Xtrain.shape}")

        rf_clf=RandomForestClassifier()
        # Train and tune the model
        print(self.hyperparameterGrid)
        rf_random = RandomizedSearchCV(estimator = rf_clf, 
                                        param_distributions = self.hyperparameterGrid, 
                                        n_iter = self.args.num_comb_model_tuning, 
                                        cv = self.args.num_cv, 
                                        verbose=1, 
                                        random_state=self.args.seed, 
                                        n_jobs = self.args.n_jobs,
                                        scoring='f1_weighted')

        # Fit the random search model
        rf_random.fit(Xtrain, Ytrain)
        
        # Save the best model and its details
        self.bestModelParams = rf_random.best_params_
        self.trainedRFmodel = rf_random.best_estimator_
        self.validationScoreList = rf_random.cv_results_['mean_test_score'] 

    def eval(self, Xtest, Ytest = None, print_performance_metric = False):
        """
        Generates predictions from the trainedRFmodel. If Ytest is passed, then will generate accuracy metrics for the trained model.
        
        params:
            - Xtest: dataset (Nsamples, feature_size)
            - Ytest: labels (Nsamples,)

        Output:
            - predict_labels: Predicted labels from the trained model (Nsamples,)
            - test_performance_metric: Contains the performance metrics (f1, precision, recall) if the true labels are passed
        """
        test_performance_metric = None
        # Sanity check
        if len(Xtest.shape)>2:
            raise ValueError(f"Shape of testing data array incorrect : {Xtest.shape}")

        # Getting the prediction from the model over the test_data
        predict_labels = self.trainedRFmodel.predict(Xtest)

        if Ytest is not None:
            # Get the classification report of the prediction
            class_results = classification_report(y_true= Ytest,
                                                   y_pred= predict_labels, 
                                                   output_dict=True)

            test_performance_metric = {'f1': class_results['weighted avg']['f1-score'],
                                        'precision': class_results['weighted avg']['precision'],
                                        'recall': class_results['weighted avg']['recall']}

            if print_performance_metric:
                # Print the classification report and the confusion matrix
                print(f" ----- Evaluation performation metrics -----")
                print(classification_report(Ytest,predict_labels))
                print(confusion_matrix(Ytest, predict_labels))

        return predict_labels, test_performance_metric 

    def save_model(self, fpath):
        """
        Saves the model and the model details to the specified path.

        params:
            - fpath: full path where the model should be saved
        """
        # Used for saving and loading the object
        model_pickle_file = {"channelName":self.channelName,
                                "channelType":self.channelType,
                                "validationScoreList":self.validationScoreList,
                                "trainedRFmodel":self.trainedRFmodel,  
                                "bestModelParams":self.bestModelParams}

        # Write the pickle file
        with open(fpath, 'wb') as handle:
            pickle.dump(model_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    def load_model(self, fpath):
        """
        Loads the model from the specified path. And populates all the corresponding model details of this object.

        params:
            - fpath: full path from which to load the RF model
        """
        # Load the dict from fpath and update the instance attributes
        with open(fpath, 'rb') as handle:
            model_pickle_file = pickle.load(handle)

        self.channelName = model_pickle_file["channelName"]
        self.channelType = model_pickle_file["channelType"]
        self.validationScoreList = model_pickle_file["validationScoreList"]
        self.trainedRFmodel = model_pickle_file["trainedRFmodel"]
        self.bestModelParams = model_pickle_file["bestModelParams"]


    @staticmethod
    def generate_hyperparameter_grid():
        """
        Generates the hyperparameter grid over which model tuning needs to be performed.
        
        Output:
            - Populates self.hyperparameterGrid
        """
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 80, stop = 200, num = 10)]
        # Criterion
        criterion = ["entropy","gini"]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2', None]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        randomGrid = {'n_estimators': n_estimators,
                    'criterion': criterion,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        
        return randomGrid

    @staticmethod
    def unit_test_baseRFmodel(args):
        """
        unit test for the base RF model object
        """
        X_train = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/0/15/DVFS_individual/all/channel_bins_train.npy")
        Y_train = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/0/15/DVFS_individual/all/labels_train.npy")

        X_test = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/10/15/DVFS_individual/all/channel_bins_test.npy")
        Y_test = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/10/15/DVFS_individual/all/labels_test.npy")
        
        print(f" - Shape of the training data and label : {X_train.shape, Y_train.shape}")
        print(f" - Shape of the test data and label : {X_test.shape, Y_test.shape}")

        print(f" - Training the model -")
        baseModelInst = baseRFmodel(args=args, channelName="test", channelType="dvfs")
        baseModelInst.train(Xtrain=X_train[0],Ytrain=Y_train)

        print(f" - Evaluating the model -")
        baseModelInst.eval(Xtest=X_test[0],Ytest=Y_test,print_performance_metric=True)

        print(f" - Saving the model -")
        baseModelInst.save_model(fpath="testmodel.pkl")

        print(f" - Loading and testing the model -")
        newBaseModelInst = baseRFmodel(args=args)
        newBaseModelInst.load_model(fpath="testmodel.pkl")
        newBaseModelInst.eval(Xtest=X_test[0],Ytest=Y_test,print_performance_metric=True)

class late_stage_fusion:
    """
    Object for tracking the stage-1 and stage-2 predictive performance.
    """
    globlChannelNameList = ['gpu_load','cpu_freq_lower_cluster','cpu_freq_higher_cluster', 'cpu_bw', 'gpu_bw',
                    'kgsl-busmon','l3-cpu0', 'l3-cpu4', 'llccbw', 'memlat_cpu0',
                    'memlat_cpu4', 'rx_bytes', 'tx_bytes', 'current_now', 'voltage_now']
    
    hpcGroupNameList = ["hpc-group-1", "hpc-group-2", "hpc-group-3", "hpc-group-4"]

    def __init__(self, args) -> None:
        self.args = args
        # Stores a dict of baseRFmodel objects. One for every globl channel.
        self.globlChannelBaseClf = None
        # Stores a dict of baseRFmodel objects. One for every hpc group.
        self.hpcGroupBaseClf = None
        # Stage 2 MLP model for fusion
        self.stage2mlp = None
        # List of performance metrics for all the GLOBL channels and HPC groups
        self.stage1ClassifierPerformanceMetrics = {chnName: None for chnName in (late_stage_fusion.globlChannelNameList+late_stage_fusion.hpcGroupNameList)}
        # Performance metric after the stage 2 models
        self.fusionPerformanceMetric = {"dvfs":{"ensemble": None, "sg": None}, "globl":{"ensemble": None, "sg": None}}
        

    def stage1trainGLOBL(self, XtrainGLOBL, YtrainGLOBL):
        """
        Trains all the base-classifiers for all the GLOBL channels.
        
        params:
            - XtrainGLOBL: dataset (Nchannels, Nsamples, feature_size)
            - YtrainGLOBL: labels (Nsamples,)
        
        Output:
            - Populates the self.globlChannelBaseClf
        """
        # Dict of baseRFmodel objects
        modelDict = {}
        # Train a classifier for every channel
        for channelIndex, channelName in enumerate(self.globlChannelNameList): 
            baseModelInst = baseRFmodel(args=self.args, channelName=channelName, channelType="globl")
            baseModelInst.train(Xtrain=XtrainGLOBL[channelIndex],Ytrain=YtrainGLOBL)
            modelDict[channelName] = baseModelInst

        self.globlChannelBaseClf = modelDict
    

    def stage1evalGLOBL(self, XtestGLOBL, YtestGLOBL, updateObjectPerformanceMetrics):
        """
        Evaluates all the base-classifiers for all the GLOBL channels.
        
        params:
            - XtestGLOBL: dataset (Nchannels, Nsamples, feature_size)
            - YtestGLOBL: labels (Nsamples,)
            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the globl base classifiers.
            
        Output:
            - globl_predict_labels: Predicted labels from the trained model (Nsamples, Nchannels)
        """
        # Stores the predictions of all the channels over all the samples
        allChannelPredictions = []

        for channelIndex, (channelName, channelModel) in enumerate(self.globlChannelBaseClf.items()): 
            # Get the prediction from the channel model
            predict_labels, test_performance_metric = channelModel.eval(Xtest = XtestGLOBL[channelIndex], 
                                                                        Ytest = YtestGLOBL, 
                                                                        print_performance_metric = True)
            allChannelPredictions.append(predict_labels)
            
            if updateObjectPerformanceMetrics:
                self.stage1ClassifierPerformanceMetrics[channelName] = test_performance_metric
            
        globl_predict_labels = np.stack(allChannelPredictions, axis=-1)
        return globl_predict_labels

    def stage1trainHPC(self, XtrainHPC, YtrainHPC):
        """
        Trains all the base-classifiers for all the HPC groups.
        
        params:
            - XtrainHPC: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - YtrainHPC: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)

        Output:
            - Populates the self.hpcGroupBaseClf
        """
        # Dict of baseRFmodel objects
        modelDict = {}
        
        # Train a classifier for every group
        for groupNumber, groupName in enumerate(self.hpcGroupNameList): 
            # Fetch trainingData (Nsamples, feature_size) and trainingLabel (Nsamples,)
            trainingData = XtrainHPC[groupNumber].squeeze() 
            trainingLabel = YtrainHPC[groupNumber]
            
            baseModelInst = baseRFmodel(args=self.args, channelName=groupName, channelType="hpc")
            baseModelInst.train(Xtrain=trainingData,Ytrain=trainingLabel)
            modelDict[groupName] = baseModelInst

        self.hpcGroupBaseClf = modelDict

    def stage1evalHPC(self, XtestHPC, YtestHPC, updateObjectPerformanceMetrics):
        """
        Evaluates all the base-classifiers for all the HPC groups.
        
        params:
            - XtestHPC: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - YtestHPC: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the globl base classifiers.
            
        Output:
            - allGroupPredictions: Predicted labels from the trained model 
                                [labels_group1, labels_group2, labels_group3, labels_group4] | labels-shape: (Nsamples,)
        """
        # Stores the predictions of all the groups over their corresponding test dataset
        allGroupPredictions = []
        
        # Train a classifier for every group
        for groupNumber, (groupName,groupModel) in enumerate(self.hpcGroupBaseClf.items()): 
            # Fetch testingData (Nsamples, feature_size) and trainingLabel (Nsamples,)
            testData = XtestHPC[groupNumber].squeeze() 
            testLabel = YtestHPC[groupNumber]
            
            # Get the prediction from the group model
            predict_labels, test_performance_metric = groupModel.eval(Xtest = testData, 
                                                                        Ytest = testLabel, 
                                                                        print_performance_metric = True)
            allGroupPredictions.append(predict_labels)
            
            if updateObjectPerformanceMetrics:
                self.stage1ClassifierPerformanceMetrics[groupName] = test_performance_metric
            
        return allGroupPredictions
        

    def stage1evalAll(self, XtestGLOBL_HPC, YtestGLOBL_HPC, generate_XtrainSG, updateObjectPerformanceMetrics):
        """
        Evaluates the individual predictions of all the stage 1 classifiers (all GLOBL channels and HPC groups)
        
        params:
            - XtestGLOBL_HPC: {"globl":XtestGLOBL, "hpc":XtestHPC}
            - YtestGLOBL_HPC: {"globl":YtestGLOBL, "hpc":YtestHPC}
            - generate_XtrainSG: True or False. If True, then will generate a stacked dataset for training the second stage model.

        Output:
            - XtrainSG: {"group1":{"Xtrain":dataset, "Ytrain":labels}, ...}
                        dataset shape - (Nsample, Nchannel+1)
                        label shape - (Nsample, 1)

        """
        # Calls stage1evalHPC and stage1evalGLOBL
        allGLOBL_predict_labels = self.stage1evalGLOBL(XtestGLOBL = XtestGLOBL_HPC["globl"], 
                            YtestGLOBL = YtestGLOBL_HPC["globl"], 
                            updateObjectPerformanceMetrics = updateObjectPerformanceMetrics)
        
        allHPC_predict_labels = self.stage1evalHPC(XtestHPC = XtestGLOBL_HPC["hpc"], 
                             YtestHPC = YtestGLOBL_HPC["hpc"], 
                             updateObjectPerformanceMetrics = updateObjectPerformanceMetrics)
        
        # TODO: Continue from here.
        # If generate_XtrainSG, then create XtrainSG for the second stage model
        # Populates the self.stage1ClassifierPerformanceMetrics
        pass

    def stage2_ensemble(self, XtrainSG, includeHPC, globlChannelsOfInterest, generateRocCurve):
        """
        Generates the performance metrics of the fused classification by using majority voting.
        """
        pass

    def stage2_mlpTrain(self, XtrainSG, includeHPC, globlChannelsOfInterest):
        """
        Trains the stage2 MLP model. 
        
        params:
            - XtrainSG : Decisions of the stage1 models for GLOBL channels and HPC group
                        {"group1":{"Xtrain":dataset, "Ytrain":labels}, ...}
                            dataset shape - (Nsample, Nchannel+1)
                            label shape - (Nsample, 1)
        """
        pass

    def stage2_mlpEval(self, XtestSG, includeHPC, globlChannelsOfInterest):
        """
        Generates the stage2 performance metrics using majority decision. Option to generate the ROC curve. 
        """
        pass

    def save_fusion_object(self, fpath):
        """
        Saves the model and the model details to the specified path.

        params:
            - fpath: full path where the model should be saved
        """
        pass

    def load_fusion_object(self, fpath):
        """
        Loads the model from the specified path. And populates all the corresponding model details of this object.
        """
        pass

def main_worker(args, xmd_base_folder_location):
    """
    Worker node that performs the complete analysis.
    params:
        - args: easydict storing the experimental parameters
        - xmd_base_folder_location: Location of base folder of xmd
    """
    # # Train the HPC and the GLOBL base classifiers
    # training_inst = train_classifiers(args=args, xmd_base_folder_location=xmd_base_folder_location)
    # training_inst.generate_all_base_classifier()

    baseRFmodel.unit_test_baseRFmodel(args=args)


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
    main_worker(args=args, xmd_base_folder_location= base_folder_location)

if __name__ == '__main__':
    main()