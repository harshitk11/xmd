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
from scipy.stats import mode
import re
from collections import Counter
from imblearn.over_sampling import RandomOverSampler


BENIGN_LABEL = 0
MALWARE_LABEL = 1

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

class resample_dataset:
    """
    Contains all the methods for resampling the datasets to achieve the desired malware percentage.
    """

    def __init__(self, malwarePercent) -> None:
        # % of malware in the resampled dataset. The resampling is done by oversampling the benign class.
        self.malwarePercent = malwarePercent
    
    def __call__(self, y):
        """
        Returns a dict containing the number of samples for the benign and the malware class
        """
        target_stats = Counter(y)
        Bo = int((1-self.malwarePercent)*target_stats[MALWARE_LABEL]/self.malwarePercent)
        Mo = target_stats[MALWARE_LABEL]
        resampled_stats = {MALWARE_LABEL:Mo, BENIGN_LABEL:Bo}
        return resampled_stats

    def generate_sampling_indices(self, X, y):
        """
        Generates indices of the samples from the original dataset that are used in the new dataset.
        params:
            - X : (Nsamples, Nfeature)
            - y : (Nsamples, )

        Output:
            - ros.sample_indices_ : Indices of the samples selected. ndarray of shape (n_new_samples,)
        """
        rmInst = resample_dataset(malwarePercent=self.malwarePercent)
        if self.malwarePercent == 0.1:
            ros = RandomOverSampler(random_state=42, sampling_strategy=rmInst)
        elif self.malwarePercent == 0.5:
            ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
        else:
            raise ValueError(f"Expecting malwarePercent of 0.1 or 0.5. Got {self.malwarePercent}.")

        _, _ = ros.fit_resample(X, y)
        return ros.sample_indices_

    def resampleBaseTensor(self, X, y):
        """
        Resamples the Globl/Hpc Tensor.
        params:
            - X: dataset (Nchannels, Nsamples, feature_size)
            - y: labels (Nsamples,) 
        Output:
            - X_res, y_res : Resampled dataset
        """
        # Get the sampling indices
        sampIndx = self.generate_sampling_indices(X[0], y)

        # Resample the dataset
        X_res = X[:,sampIndx,:]
        y_res = y[sampIndx]

        return X_res, y_res

    def resampleHpcTensor(self, Xlist, yList):
        """
        Resamples the dataset for all the HPC groups.
        
        params:
            - Xlist: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - yList: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)

        Output:
            - Xlist_res, yList_res : Resampled dataset
        """
        Xlist_res = []
        yList_res = []

        for grpIndx, _ in enumerate(Xlist):
            X_res, y_res = self.resampleBaseTensor(X=Xlist[grpIndx], y=yList[grpIndx])
            Xlist_res.append(X_res)
            yList_res.append(y_res)

        return Xlist_res, yList_res

    def resampleFusionTensor(self, XtestGLOBL_HPC, YtestGLOBL_HPC):
        """
        Resamples the matched HPC and GLOBL tensor for the fusion tasks.
        params:
            - XtestGLOBL_HPC: {"globl":XtestGLOBL, "hpc":XtestHPC} | XtestGLOBL,XtestHPC = [matched_dataset_group1, matched_dataset_group2, matched_dataset_group3, matched_dataset_group4]
            - YtestGLOBL_HPC: {"globl":YtestGLOBL, "hpc":YtestHPC} | YtestGLOBL,YtestHPC = [labels_group1, labels_group2, labels_group3, labels_group4]
        
        Output:
            - XtestGLOBL_HPC_res, YtestGLOBL_HPC_res : Resampled dataset and labels
        """
        XtestGLOBL_HPC_res = {"globl":[], "hpc":[]}
        YtestGLOBL_HPC_res = {"globl":[], "hpc":[]}

        # For each HPC group
        for grpIndx,(globlDatasetTensor,hpcDatasetTensor) in enumerate(zip(XtestGLOBL_HPC["globl"], XtestGLOBL_HPC["hpc"])):
            # Create sampling indices
            sampIndx = self.generate_sampling_indices(globlDatasetTensor[0], YtestGLOBL_HPC["globl"][grpIndx])

            # Resample GLOBL and HPC using the sampling indices
            globlDatasetTensor_res = globlDatasetTensor[:,sampIndx,:]
            globlLabelTensor_res = YtestGLOBL_HPC["globl"][grpIndx][sampIndx]
            hpcDatasetTensor_res = hpcDatasetTensor[:,sampIndx,:]
            hpcLabelTensor_res = YtestGLOBL_HPC["hpc"][grpIndx][sampIndx]
            
            # Add it to the resampled dict
            XtestGLOBL_HPC_res["globl"].append(globlDatasetTensor_res)
            XtestGLOBL_HPC_res["hpc"].append(hpcDatasetTensor_res)
            YtestGLOBL_HPC_res["globl"].append(globlLabelTensor_res)
            YtestGLOBL_HPC_res["hpc"].append(hpcLabelTensor_res)

        return XtestGLOBL_HPC_res, YtestGLOBL_HPC_res

    @staticmethod
    def unitTestResampler():
        ########### Testing the individual GLOBL and HPC resampling ###########
        # Load dataset
        dfvs_X_train = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/DVFS_individual/all/channel_bins_train.npy")
        dfvs_Y_train = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/DVFS_individual/all/labels_train.npy")
        print(f"- Shape of the original dataset and labels array: {dfvs_X_train.shape, dfvs_Y_train.shape}")
        print(f"- [Pre] Number of malware and benign samples : {Counter(dfvs_Y_train)}")
        # Perform resampling
        rmInst = resample_dataset(malwarePercent=0.5)
        X_res, y_res = rmInst.resampleBaseTensor(X=dfvs_X_train, y=dfvs_Y_train)
        print(f"- Shape of the resampled dataset and labels array: {X_res.shape, y_res.shape}")
        print(f"- [Post] Number of malware and benign samples : {Counter(y_res)}")

        ########### Testing the fusion GLOBL and HPC resampling ###########
        # Loading the datasets for testing the fusion modules
        hpc_x_test = []
        hpc_y_test = []
        globl_x_test = []
        globl_y_test = []
        for group in ["rn1","rn2","rn3","rn4"]:
            hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_partition_for_HPC_DVFS_fusion/{group}/channel_bins_test.npy"))
            hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_partition_for_HPC_DVFS_fusion/{group}/labels_test.npy"))
            globl_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/DVFS_partition_for_HPC_DVFS_fusion/{group}/channel_bins_test.npy"))
            globl_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/DVFS_partition_for_HPC_DVFS_fusion/{group}/labels_test.npy"))

        XtestGLOBL_HPC= {"globl":globl_x_test, "hpc":hpc_x_test}
        YtestGLOBL_HPC= {"globl":globl_y_test, "hpc":hpc_y_test}
    
        print(" [Pre] Details of HPC and globl matched data")
        print(f" - Shape of the hpc data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_test,hpc_y_test)]}")
        print(f" - Shape of the globl data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(globl_x_test,globl_y_test)]}")
        print(" - [Pre] Number of malware and benign samples")
        print([Counter(yxx) for yxx in YtestGLOBL_HPC["globl"]])
        print([Counter(yxx) for yxx in YtestGLOBL_HPC["hpc"]])

        # Perform resampling
        XtestGLOBL_HPC_res, YtestGLOBL_HPC_res = rmInst.resampleFusionTensor(XtestGLOBL_HPC=XtestGLOBL_HPC, YtestGLOBL_HPC=YtestGLOBL_HPC)
        print(" -------------------------------------------------------------------------------------------------")
        print(" [Post] Details of HPC and globl matched data")
        print(f" - Shape of the hpc data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(XtestGLOBL_HPC_res['hpc'],YtestGLOBL_HPC_res['hpc'])]}")
        print(f" - Shape of the globl data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(XtestGLOBL_HPC_res['globl'],YtestGLOBL_HPC_res['globl'])]}")
        print(" - [Post] Number of malware and benign samples")
        print([Counter(yxx) for yxx in YtestGLOBL_HPC_res["globl"]])
        print([Counter(yxx) for yxx in YtestGLOBL_HPC_res["hpc"]])


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
        print(f"  - Validation score during training: {self.validationScoreList}")

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

            test_performance_metric = {'f1': class_results['macro avg']['f1-score'],
                                        'precision': class_results['macro avg']['precision'],
                                        'recall': class_results['macro avg']['recall']}

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

        # print(f" - Evaluating the model -")
        # baseModelInst.eval(Xtest=X_test[0],Ytest=Y_test,print_performance_metric=True)

        # print(f" - Saving the model -")
        # baseModelInst.save_model(fpath="testmodel.pkl")

        # print(f" - Loading and testing the model -")
        # newBaseModelInst = baseRFmodel(args=args)
        # newBaseModelInst.load_model(fpath="testmodel.pkl")
        # newBaseModelInst.eval(Xtest=X_test[0],Ytest=Y_test,print_performance_metric=True)

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
        
        # Stores a dict of baseRFmodel objects. One for every globl channel. {chnName: baseRFModel object, ...}
        self.globlChannelBaseClf = None
        # Stores a dict of baseRFmodel objects. One for every hpc group. {grpName: baseRFModel object, ...}
        self.hpcGroupBaseClf = None
        # Stage 2 MLP model for fusion
        self.stage2mlp = None
        
        ########################################## Performance metricts  ##########################################
        # List of file hashes used for testing the performance of hpc-globl fusion models [Used for VT comparison]
        self.hpcGloblTestFileHashlist = {groupName:None for groupName in late_stage_fusion.hpcGroupNameList}

        # List of performance metrics for all the base classifiers of GLOBL channels and HPC groups
        self.stage1ClassifierPerformanceMetrics = {chnGrpName: None for chnGrpName in (late_stage_fusion.globlChannelNameList+late_stage_fusion.hpcGroupNameList)}
        
        # Performance metric for globl fusion models ("dvfs" created with channels 1-11, "globl" created with channels 1-15)
        self.globlFusionPerformanceMetric = {"dvfs":None, "globl":None}

        # Performance metric for hpc-globl fusion models for all groups
        hpcGloblFusionMetric = {"hpc":None, "hpc-dvfs-ensemble":None, "hpc-dvfs-sg":None, "hpc-globl-ensemble":None, "hpc-globl-sg":None}
        self.hpcGloblFusionPerformanceMetricAllGroup = {groupName:hpcGloblFusionMetric.copy() for groupName in late_stage_fusion.hpcGroupNameList}
        ###########################################################################################################
    
    def stage1trainGLOBL(self, XtrainGLOBL, YtrainGLOBL, updateObjectPerformanceMetrics):
        """
        Trains all the base-classifiers for all the GLOBL channels.
        
        params:
            - XtrainGLOBL: dataset (Nchannels, Nsamples, feature_size)
            - YtrainGLOBL: labels (Nsamples,)
            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the globl base classifiers.
            
        Output:
            - Populates the self.globlChannelBaseClf
            - Updates the self.stage1ClassifierPerformanceMetrics if the flag is passed
        """
        # Dict of baseRFmodel objects
        modelDict = {}
        # Train a classifier for every channel
        for channelIndex, channelName in enumerate(self.globlChannelNameList): 
            print(f" - Training baseRF model for globl-channel: {channelName}")
            baseModelInst = baseRFmodel(args=self.args, channelName=channelName, channelType="globl")
            baseModelInst.train(Xtrain=XtrainGLOBL[channelIndex],Ytrain=YtrainGLOBL)
            modelDict[channelName] = baseModelInst

            # Update the stage1 classifier performance metric if the flag is passed
            if updateObjectPerformanceMetrics:
                _performance_metric = {'f1': max(baseModelInst.validationScoreList),
                                        'precision': None,
                                        'recall': None}
                self.stage1ClassifierPerformanceMetrics[channelName] = _performance_metric
               
        self.globlChannelBaseClf = modelDict
    
    def stage1trainHPC(self, XtrainHPC, YtrainHPC, updateObjectPerformanceMetrics):
            """
            Trains all the base-classifiers for all the HPC groups.
            
            params:
                - XtrainHPC: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
                - YtrainHPC: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
                - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the globl base classifiers.
                
            Output:
                - Populates the self.hpcGroupBaseClf
                - Updates the self.stage1ClassifierPerformanceMetrics if the flag is passed
            """
            # Dict of baseRFmodel objects
            modelDict = {}
            
            # Train a classifier for every group
            for groupNumber, groupName in enumerate(self.hpcGroupNameList): 
                print(f" - Training baseRF model for hpc-group: {groupName}")

                # Fetch trainingData (Nsamples, feature_size) and trainingLabel (Nsamples,)
                trainingData = XtrainHPC[groupNumber].squeeze() 
                trainingLabel = YtrainHPC[groupNumber]
                
                baseModelInst = baseRFmodel(args=self.args, channelName=groupName, channelType="hpc")
                baseModelInst.train(Xtrain=trainingData,Ytrain=trainingLabel)
                modelDict[groupName] = baseModelInst

                if updateObjectPerformanceMetrics:
                    _performance_metric = {'f1': max(baseModelInst.validationScoreList),
                                            'precision': None,
                                            'recall': None}
                    self.stage1ClassifierPerformanceMetrics[groupName] = _performance_metric
                

            self.hpcGroupBaseClf = modelDict

    def stage1evalGLOBL(self, XtestGLOBL, YtestGLOBL, updateObjectPerformanceMetrics, print_performance_metric):
        """
        Evaluates all the base-classifiers for all the GLOBL channels.
        
        params:
            - XtestGLOBL: dataset (Nchannels, Nsamples, feature_size)
            - YtestGLOBL: labels (Nsamples,)
            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the globl base classifiers.
            - print_performance_metric: If True, then will print the performance metrics to stdout

        Output:
            - globl_predict_labels: Predicted labels from all the stage1 trained models (Nsamples, Nchannels)
        """
        # Stores the predictions of all the channels over all the samples
        allChannelPredictions = []

        for channelIndex, (channelName, channelModel) in enumerate(self.globlChannelBaseClf.items()): 
            print(f" - Evaluating baseRF model for globl-channel: {channelName}")
            
            # Get the prediction from the channel model
            predict_labels, test_performance_metric = channelModel.eval(Xtest = XtestGLOBL[channelIndex], 
                                                                        Ytest = YtestGLOBL, 
                                                                        print_performance_metric = print_performance_metric)
            allChannelPredictions.append(predict_labels)
            if updateObjectPerformanceMetrics:
                self.stage1ClassifierPerformanceMetrics[channelName] = test_performance_metric
            
        globl_predict_labels = np.stack(allChannelPredictions, axis=-1)
        assert globl_predict_labels.shape[-1] == len(late_stage_fusion.globlChannelNameList), f"Shape of globl_predict_labels (Nsamples, Nchannels) is incorrect {globl_predict_labels.shape}"
        
        return globl_predict_labels

    def stage1evalHPC(self, XtestHPC, YtestHPC, updateObjectPerformanceMetrics, print_performance_metric):
        """
        Evaluates all the base-classifiers for all the HPC groups.
        
        params:
            - XtestHPC: [dataset_group1, dataset_group2, dataset_group3, dataset_group4] | dataset shape: (1, Nsamples, feature_size)
            - YtestHPC: [labels_group1, labels_group2, labels_group3, labels_group4] | labels_shape: (Nsamples,)
            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the hpc-group base-classifiers.
            - print_performance_metric: If True, then will print the performance metrics to stdout

        Output:
            - allGroupPredictions: Predicted labels from the trained model 
                                [labels_group1, labels_group2, labels_group3, labels_group4] | labels-shape: (Nsamples,)
        """
        # Stores the predictions of all the groups over their corresponding test dataset
        allGroupPredictions = []
        
        # Test a classifier for every group
        for groupNumber, (groupName,groupModel) in enumerate(self.hpcGroupBaseClf.items()): 
            print(f" - Evaluating baseRF model for hpc-group: {groupName}")
            
            # Fetch testingData (Nsamples, feature_size) and trainingLabel (Nsamples,)
            testData = XtestHPC[groupNumber].squeeze() 
            testLabel = YtestHPC[groupNumber]
            
            # Get the prediction from the group model
            predict_labels, test_performance_metric = groupModel.eval(Xtest = testData, 
                                                                        Ytest = testLabel, 
                                                                        print_performance_metric = print_performance_metric)
            allGroupPredictions.append(predict_labels)
            
            if updateObjectPerformanceMetrics:
                self.stage1ClassifierPerformanceMetrics[groupName] = test_performance_metric
            
        return allGroupPredictions
        
    def generateTrainSGdataset(self, XtestGLOBL_HPC, YtestGLOBL_HPC):
        """
        Generates the TrainSG dataset which is the training dataset for the second stage MLP model. 
        We will have four datasets, one for each HPC-GLOBL-group.

        params:
            - XtestGLOBL_HPC: {"globl":XtestGLOBL, "hpc":XtestHPC} | XtestGLOBL,XtestHPC = [matched_dataset_group1, matched_dataset_group2, matched_dataset_group3, matched_dataset_group4]
            - YtestGLOBL_HPC: {"globl":YtestGLOBL, "hpc":YtestHPC} | YtestGLOBL,YtestHPC = [labels_group1, labels_group2, labels_group3, labels_group4]
            
        Output:
            - XtrainSG: {"hpc-group-1":{"Xtrain":dataset, "Ytrain":labels}, ...}
                        dataset shape - (Nsample, Nchannel+1)
                        label shape - (Nsample, 1)

        """
        XtrainSG = {}
        # Sanity check to ensure that the matched dataset for globl and hpc are aligned, i.e., the labels are same
        for globlGrouplabels, hpcGrouplabels in zip(YtestGLOBL_HPC["globl"],YtestGLOBL_HPC["hpc"]):
            assert (globlGrouplabels==hpcGrouplabels).all(), f"Files not aligned for GLOBL HPC fusion"

        # Get stage-1 predictions of the globl base-classifiers for each matched dataset. 
        allGLOBL_predict_labels = []
        for trainDataset, trainLabels in zip(XtestGLOBL_HPC["globl"], YtestGLOBL_HPC["globl"]):
            allGLOBL_predict_labels.append(self.stage1evalGLOBL(XtestGLOBL = trainDataset, 
                                                                YtestGLOBL = trainLabels, 
                                                                updateObjectPerformanceMetrics = False,
                                                                print_performance_metric=False))   

        # Get stage-1 predictions of the hpc base-classifiers for each matched dataset. 
        allHPC_predict_labels = self.stage1evalHPC(XtestHPC = XtestGLOBL_HPC["hpc"], 
                                                    YtestHPC = YtestGLOBL_HPC["hpc"], 
                                                    updateObjectPerformanceMetrics = False,
                                                    print_performance_metric=False)

        # Merge the HPC and globl predictions for each of the matched dataset
        for groupIndex,(globlPredictionGroup, hpcPredictionGroup) in enumerate(zip(allGLOBL_predict_labels,allHPC_predict_labels)):
            assert globlPredictionGroup.shape[0] == hpcPredictionGroup.shape[0], f"Number of samples in the matched data is not same: {globlPredictionGroup.shape,hpcPredictionGroup.shape}"
            
            # Add the HPC group to the globl channel
            total_predictions = np.concatenate((globlPredictionGroup, hpcPredictionGroup[:,np.newaxis]), axis = -1)
            
            XtrainSG[self.hpcGroupNameList[groupIndex]]={"Xtrain":total_predictions, "Ytrain":YtestGLOBL_HPC["globl"][groupIndex]}
        
        return XtrainSG

    def ensemble_eval_using_majority_vote(self, stage1Decisions, Ytrue, print_performance_metric):
        """
        Generates the final decision from stage1Decisions using majority voting. 
        If true label is passed, then will generate performance metrics.

        params:        
            - stage1Decisions: Decisions from the participating base-classifiers (Nsample, Nchannel) 
            - Ytrue: True labels (Nsamples,)
            - print_performance_metric: If True, then will print the performance metrics to stdout

        Output:
            - final_ensemble_decision: combined decision using majority voting (Nsamples,)
            - test_performance_metric: Contains the performance metrics (f1, precision, recall) if the true labels are passed
        """
        # Get the majority vote of all the decisions of the classifiers
        final_ensemble_decision = mode(stage1Decisions, axis=1)[0]

        if Ytrue is not None:
            # Get the classification report of the prediction
            class_results = classification_report(y_true= Ytrue,
                                                   y_pred= final_ensemble_decision, 
                                                   output_dict=True)

            test_performance_metric = {'f1': class_results['macro avg']['f1-score'],
                                        'precision': class_results['macro avg']['precision'],
                                        'recall': class_results['macro avg']['recall']}

            if print_performance_metric:
                # Print the classification report and the confusion matrix
                print(f" ----- [Majority Voting] Evaluation performation metrics -----")
                print(classification_report(Ytrue,final_ensemble_decision))
                print(confusion_matrix(Ytrue, final_ensemble_decision))

        return final_ensemble_decision, test_performance_metric 
    
    
    def stage2_globlFusion_ensemble_eval(self, Xeval, Yeval, updateObjectPerformanceMetrics, print_performance_metric):
        """
        Generates the evaluation scores for globl fusion : ("dvfs" created with channels 1-11, "globl" created with channels 1-15)
        
        params:
            - Xeval: dataset (Nchannels, Nsamples, feature_size)
            - Yeval: labels (Nsamples,)
            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the hpc-group base-classifiers.
            - print_performance_metric: If True, then will print the performance metrics to stdout

        Output:
            - Updates the self.globlFusionPerformanceMetric if updateObjectPerformanceMetrics is True
        """
        # Generate the predicted labels for all the channels
        allGLOBL_predict_labels = self.stage1evalGLOBL(XtestGLOBL = Xeval, 
                                                        YtestGLOBL = Yeval, 
                                                        updateObjectPerformanceMetrics = False,
                                                        print_performance_metric=False) 

        # Get the indices for "dvfs" channels and "globl" channels
        dvfs_channels = late_stage_fusion.globlChannelNameList[:11]
        globl_channels = late_stage_fusion.globlChannelNameList
        dvfsChnIndex = [late_stage_fusion.globlChannelNameList.index(coi) for coi in dvfs_channels]
        globlChnIndex = [late_stage_fusion.globlChannelNameList.index(coi) for coi in globl_channels]

        # Filter the predictions for the "dvfs" channels and the "globl" channels
        stage1DecisionsFilteredDVFS = allGLOBL_predict_labels[:,dvfsChnIndex]
        stage1DecisionsFilteredGLOBL = allGLOBL_predict_labels[:,globlChnIndex]

        # Get the evaluation metrics
        print(" *** Fusing the DVFS channels ***")
        _, dvfsFusion_performance_metric = self.ensemble_eval_using_majority_vote(stage1Decisions = stage1DecisionsFilteredDVFS, 
                                                                                Ytrue = Yeval, 
                                                                                print_performance_metric = print_performance_metric)
        print(" *** Fusing the GLOBL channels ***")
        _, globlFusion_performance_metric = self.ensemble_eval_using_majority_vote(stage1Decisions = stage1DecisionsFilteredGLOBL, 
                                                                                Ytrue = Yeval, 
                                                                                print_performance_metric = print_performance_metric)

        if updateObjectPerformanceMetrics:
            self.globlFusionPerformanceMetric = {"dvfs":dvfsFusion_performance_metric, "globl":globlFusion_performance_metric}

    
    def stage2_hpcGlobl_ensemble_eval(self, XtrainSG, updateObjectPerformanceMetrics, print_performance_metric):
        """
        Generates the evaluation scores for each of the four hpc-globl-groups.
        Updates the self.hpcGloblFusionPerformanceMetricAllGroup if updateObjectPerformanceMetrics is True
        
        params:        
            - XtrainSG: {"hpc-group-1":{"Xtrain":dataset, "Ytrain":labels}, ...}
                                        dataset shape - (Nsample, Nchannel+1) -> Includes both globl and hpc channels
                                        label shape - (Nsample, 1)

            - updateObjectPerformanceMetrics: If True, then will update the performance metrics of the hpc-group base-classifiers.
            - print_performance_metric: If True, then will print the performance metrics to stdout

        Output:
            - Updates the self.hpcGloblFusionPerformanceMetricAllGroup if updateObjectPerformanceMetrics is True
        """   
        for hpcGroupName, dataLabelDict in XtrainSG.items():
            print(f" ----------------------- HPC+DVFS+GLOBL fusion for group : {hpcGroupName} -----------------------")
            stage1Decisions = dataLabelDict["Xtrain"]
            Ytrue = dataLabelDict["Ytrain"]
            
            # Get the test performance metric for only HPC
            print(" *** Only HPC channel ***")
            hpcPrediction = stage1Decisions[:,[-1]]
            _, hpc_performance_metric = self.ensemble_eval_using_majority_vote(stage1Decisions = hpcPrediction, 
                                                                                Ytrue = Ytrue, 
                                                                                print_performance_metric = print_performance_metric)
        
            # Get the test performance metric for HPC+DVFS
            print(" *** Fusing the HPC+DVFS channels ***")
            dvfs_channels = late_stage_fusion.globlChannelNameList[:11]
            dvfsChnIndex = [late_stage_fusion.globlChannelNameList.index(coi) for coi in dvfs_channels]
            channels_of_interest_index = dvfsChnIndex + [-1]
            stage1DecisionsFiltered = stage1Decisions[:,channels_of_interest_index]
            _, hpcDvfs_performance_metric = self.ensemble_eval_using_majority_vote(stage1Decisions = stage1DecisionsFiltered, 
                                                                                Ytrue = Ytrue, 
                                                                                print_performance_metric = print_performance_metric)

            # Get the test performance metric for HPC+GLOBL
            print(" *** Fusing the HPC+GLOBL channels ***")
            globl_channels = late_stage_fusion.globlChannelNameList
            globlChnIndex = [late_stage_fusion.globlChannelNameList.index(coi) for coi in globl_channels]
            channels_of_interest_index = globlChnIndex + [-1]
            stage1DecisionsFiltered = stage1Decisions[:,channels_of_interest_index]
            _, hpcGlobl_performance_metric = self.ensemble_eval_using_majority_vote(stage1Decisions = stage1DecisionsFiltered, 
                                                                                Ytrue = Ytrue, 
                                                                                print_performance_metric = print_performance_metric)

            if updateObjectPerformanceMetrics:
                self.hpcGloblFusionPerformanceMetricAllGroup[hpcGroupName]["hpc"] = hpc_performance_metric
                self.hpcGloblFusionPerformanceMetricAllGroup[hpcGroupName]["hpc-dvfs-ensemble"] = hpcDvfs_performance_metric
                self.hpcGloblFusionPerformanceMetricAllGroup[hpcGroupName]["hpc-globl-ensemble"] = hpcGlobl_performance_metric

    def pretty_print_performance_metric(self):
        """
        Prints the performance metric to stdout.
        """
        print("----------- GLOBL and DVFS fusion performance metric -----------")
        for fType,perfMetric in self.globlFusionPerformanceMetric.items():
            print(f" Fusion type: {fType} | F1 : {perfMetric['f1']} | precision : {perfMetric['precision']} | recall : {perfMetric['recall']}")
        print("----------- HPC and DVFS fusion performance metric -----------")
        for gName,allPerf in self.hpcGloblFusionPerformanceMetricAllGroup.items():
            print(f"- Group : {gName}")
            for fType, perfMetric in allPerf.items():    
                try:
                    print(f" Fusion type: {fType} | F1 : {perfMetric['f1']} | precision : {perfMetric['precision']} | recall : {perfMetric['recall']}")
                except:
                    continue    



    @staticmethod
    def get_hashList_from_fileList(file_paths):
        """
        Returns a list of hashes from the file list.
        params:
            - file_paths (list): List of file paths

        Output:
            - hashList (list): List of hashes extracted from the file paths
        """
        hashList = []
        regex_pattern = r'.*\/(.*)__.*'

        # Parse this file list to extract the hashes
        for filename in file_paths:
            file_hash_obj = re.search(regex_pattern, filename, re.M|re.I)
            file_hash_string = file_hash_obj.group(1).strip()
            hashList.append(file_hash_string)

        return hashList

    def save_fusion_object(self, fpath):
        """
        Saves the model and the model details to the specified path.

        params:
            - fpath: full path where the model should be saved
        """
        # Used for saving and loading the object
        model_pickle_file = {"globlChannelBaseClf":self.globlChannelBaseClf,
                                "hpcGroupBaseClf":self.hpcGroupBaseClf,
                                "stage2mlp":self.stage2mlp,
                                "hpcGloblTestFileHashlist": self.hpcGloblTestFileHashlist,
                                "stage1ClassifierPerformanceMetrics":self.stage1ClassifierPerformanceMetrics,  
                                "globlFusionPerformanceMetric":self.globlFusionPerformanceMetric,
                                "hpcGloblFusionPerformanceMetricAllGroup":self.hpcGloblFusionPerformanceMetricAllGroup}

        # Write the pickle file
        with open(fpath, 'wb') as handle:
            pickle.dump(model_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        

    def load_fusion_object(self, fpath):
        """
        Loads the model from the specified path. And populates all the corresponding model details of this object.
        """
        # Load the dict from fpath and update the instance attributes
        with open(fpath, 'rb') as handle:
            model_pickle_file = pickle.load(handle)

        self.globlChannelBaseClf = model_pickle_file["globlChannelBaseClf"]
        self.hpcGroupBaseClf = model_pickle_file["hpcGroupBaseClf"]
        self.stage2mlp = model_pickle_file["stage2mlp"]
        self.hpcGloblTestFileHashlist = model_pickle_file["hpcGloblTestFileHashlist"]
        self.stage1ClassifierPerformanceMetrics = model_pickle_file["stage1ClassifierPerformanceMetrics"]
        self.globlFusionPerformanceMetric = model_pickle_file["globlFusionPerformanceMetric"]
        self.hpcGloblFusionPerformanceMetricAllGroup = model_pickle_file["hpcGloblFusionPerformanceMetricAllGroup"]


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

    @staticmethod
    def unit_test_lateStageFusion(args):
        """
        unit test for the late_stage_fusion object
        """
        ############################### Testing the base classifier modules ###############################
        # Loading the unmatched datasets for the unit tests
        hpc_x_train = []
        hpc_y_train = []
        hpc_x_test = []
        hpc_y_test = []
        for group in ["rn1","rn2","rn3","rn4"]:
            hpc_x_train.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/HPC_individual/{group}/channel_bins_train.npy"))
            hpc_y_train.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/HPC_individual/{group}/labels_train.npy"))
            hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_individual/{group}/channel_bins_test.npy"))
            hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_individual/{group}/labels_test.npy"))
        dfvs_X_train = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/DVFS_individual/all/channel_bins_train.npy")
        dfvs_Y_train = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/std-dataset/15/40/DVFS_individual/all/labels_train.npy")
        dfvs_X_test = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/DVFS_individual/all/channel_bins_test.npy")
        dfvs_Y_test = np.load("/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/DVFS_individual/all/labels_test.npy")
        
        print("Details of GLOBL training and test data")
        print(f" - Shape of the training data and label : {dfvs_X_train.shape, dfvs_Y_train.shape}")
        print(f" - Shape of the test data and label : {dfvs_X_test.shape, dfvs_Y_test.shape}")
        print("Details of HPC training and test data")
        print(f" - Shape of the training data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_train,hpc_y_train)]}")
        print(f" - Shape of the test data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_test,hpc_y_test)]}")
        
        #################### Resampler ####################
        rmInst = resample_dataset(malwarePercent=0.1)
        dfvs_X_train, dfvs_Y_train = rmInst.resampleBaseTensor(X=dfvs_X_train, y=dfvs_Y_train)
        dfvs_X_test, dfvs_Y_test = rmInst.resampleBaseTensor(X=dfvs_X_test, y=dfvs_Y_test)
        hpc_x_train, hpc_y_train = rmInst.resampleHpcTensor(Xlist=hpc_x_train, yList=hpc_y_train)
        hpc_x_test, hpc_y_test = rmInst.resampleHpcTensor(Xlist=hpc_x_test, yList=hpc_y_test)
        ###################################################
        
        # Testing the training module
        print(f" - Training the hpc and globl base classifiers -")
        lateFusionInstance = late_stage_fusion(args=args)
        lateFusionInstance.stage1trainGLOBL(XtrainGLOBL=dfvs_X_train, YtrainGLOBL=dfvs_Y_train, updateObjectPerformanceMetrics= True)
        lateFusionInstance.stage1trainHPC(XtrainHPC=hpc_x_train, YtrainHPC=hpc_y_train, updateObjectPerformanceMetrics=True)
        
        print(" - Summary performance metric of all the models [post training] -")
        for chnName, performanceMetric in lateFusionInstance.stage1ClassifierPerformanceMetrics.items():
            print(chnName, performanceMetric)

        # Testing the evaluation module
        print(f" - Evaluating the hpc and globl base classifiers -")
        globl_predict_labels = lateFusionInstance.stage1evalGLOBL(XtestGLOBL=dfvs_X_test, YtestGLOBL=dfvs_Y_test, updateObjectPerformanceMetrics=True, print_performance_metric=True)
        allGroupPredictions = lateFusionInstance.stage1evalHPC(XtestHPC=hpc_x_test, YtestHPC=hpc_y_test, updateObjectPerformanceMetrics=True, print_performance_metric=True)

        print(" - Shape of the predicted labels post evaluation -")
        print(f" - Globl predicted labels (Nsamples, Nchannels): {globl_predict_labels.shape}")
        print(f" - HPC predicted labels for all groups (Nsamples, ): {[grp.shape for grp in allGroupPredictions]}")

        print(" - Summary performance metric of all the models [post evaluation] -")
        for chnName, performanceMetric in lateFusionInstance.stage1ClassifierPerformanceMetrics.items():
            print(chnName, performanceMetric)

        # Testing the loading and saving module
        print(f" - Saving the object -")
        lateFusionInstance.save_fusion_object(fpath="testmodel.pkl")
        
        print(f" - Loading the object and testing the models -")
        lateFusionInstance = late_stage_fusion(args=args)
        lateFusionInstance.load_fusion_object(fpath="testmodel.pkl")
        lateFusionInstance.stage1evalGLOBL(XtestGLOBL=dfvs_X_test, YtestGLOBL=dfvs_Y_test, updateObjectPerformanceMetrics=False, print_performance_metric=True)
        lateFusionInstance.stage1evalHPC(XtestHPC=hpc_x_test, YtestHPC=hpc_y_test, updateObjectPerformanceMetrics=False, print_performance_metric=True)

        print(" - Summary performance metric of all the models [post evaluation post loading] -")
        for chnName, performanceMetric in lateFusionInstance.stage1ClassifierPerformanceMetrics.items():
            print(chnName, performanceMetric)

        ############################### Testing the fusion modules ###############################
        # Loading the datasets for testing the fusion modules
        hpc_x_test = []
        hpc_y_test = []
        hpc_file_paths = []
        globl_x_test = []
        globl_y_test = []
        globl_file_paths = []
        for group in ["rn1","rn2","rn3","rn4"]:
            hpc_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_partition_for_HPC_DVFS_fusion/{group}/channel_bins_test.npy"))
            hpc_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_partition_for_HPC_DVFS_fusion/{group}/labels_test.npy"))
            with open(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/HPC_partition_for_HPC_DVFS_fusion/{group}/file_paths_test.npy", 'rb') as fp:
                hpc_file_paths.append(np.array(pickle.load(fp), dtype="object"))
            
            globl_x_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/DVFS_partition_for_HPC_DVFS_fusion/{group}/channel_bins_test.npy"))
            globl_y_test.append(np.load(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/DVFS_partition_for_HPC_DVFS_fusion/{group}/labels_test.npy"))
            with open(f"/data/hkumar64/projects/arm-telemetry/xmd/data/featureEngineeredDataset/cd-dataset/15/40/DVFS_partition_for_HPC_DVFS_fusion/{group}/file_paths_test.npy", 'rb') as fp:
                globl_file_paths.append(np.array(pickle.load(fp), dtype="object"))
            
        print("Details of HPC and globl matched data")
        print(f" - Shape of the hpc data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(hpc_x_test,hpc_y_test)]}")
        print(f" - Shape of the globl data and label : {[(dataset.shape,labels.shape) for dataset,labels in zip(globl_x_test,globl_y_test)]}")
        print(f" - Shape of the file paths hpc and globl : {[(fHpc.shape,fGlobl.shape) for fHpc,fGlobl in zip(hpc_file_paths,globl_file_paths)]}")
        
        # Testing the file hash extractor
        for fList in hpc_file_paths:
            print(len(late_stage_fusion.get_hashList_from_fileList(fList)))

        # Tesing the XTrainSG dataset generation
        XtestGLOBL_HPC= {"globl":globl_x_test, "hpc":hpc_x_test}
        YtestGLOBL_HPC= {"globl":globl_y_test, "hpc":hpc_y_test}
        
        #################### Resampler ####################
        rmInst = resample_dataset(malwarePercent=0.1)
        XtestGLOBL_HPC, YtestGLOBL_HPC = rmInst.resampleFusionTensor(XtestGLOBL_HPC=XtestGLOBL_HPC, YtestGLOBL_HPC=YtestGLOBL_HPC)
        ###################################################
        
        print(f" - Testing XTrainSG generation -")
        lateFusionInstance = late_stage_fusion(args=args)
        lateFusionInstance.load_fusion_object(fpath="testmodel.pkl")
        XtrainSG = lateFusionInstance.generateTrainSGdataset(XtestGLOBL_HPC=XtestGLOBL_HPC, YtestGLOBL_HPC = YtestGLOBL_HPC)
        for hpcGroupName, dataLabelDict in XtrainSG.items():
            print(hpcGroupName, dataLabelDict["Xtrain"].shape,dataLabelDict["Ytrain"].shape)

        # Testing the globl fusion module
        lateFusionInstance.stage2_globlFusion_ensemble_eval(Xeval=dfvs_X_test, 
                                                            Yeval=dfvs_Y_test, 
                                                            updateObjectPerformanceMetrics=True, 
                                                            print_performance_metric=True)
        # Testing the hpc-globl fusion module
        lateFusionInstance.stage2_hpcGlobl_ensemble_eval(XtrainSG = XtrainSG, 
                                                        updateObjectPerformanceMetrics=True, 
                                                        print_performance_metric=True)

        # Testing the performance metric
        lateFusionInstance.pretty_print_performance_metric()

def main_worker(args, xmd_base_folder_location):
    """
    Worker node that performs the complete analysis.
    params:
        - args: easydict storing the experimental parameters
        - xmd_base_folder_location: Location of base folder of xmd
    """
    # resample_dataset.unitTestResampler()
    # baseRFmodel.unit_test_baseRFmodel(args=args)
    late_stage_fusion.unit_test_lateStageFusion(args=args)


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