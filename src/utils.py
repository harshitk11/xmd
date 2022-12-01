"""
Contains all the utility classes and functions.
"""

import argparse
from genericpath import isfile
import json
import yaml
from easydict import EasyDict as edict
import re
from datetime import datetime
from datetime import timedelta
import statistics
from virus_total_apis import PublicApi as VirusTotalPublicApi
import time
import os


class Config:
    """
    Python class to handle the config files.
    """
    def __init__(self, file) -> None:
        """
        Reads the base configuration file.
        
        params:
            - file: Path of the base configuration file.
        """
        # Reads the base configuration file
        self.args = self.read(file)
        print(f"---------------------------------------------------------------------------------------------------------------")
        print(f"Base configuration file : {file}")
        for key,val in self.args.items():
            print(f" - {key}: {val}")
        print(f"---------------------------------------------------------------------------------------------------------------")
    
    def update(self, updatefile):
        """
        This method updates the base-configuration file based on the values read from the updatefile.

        params:
            - updatefile: Path of the file containing the updated configuration parameters.
        """
        uArgs = self.read(updatefile)
        print(f"Updating the configuration using the file : {updatefile}")
        for key, val in uArgs.items():
            self.args[key] = val
            print(f" - {key} : {val}")

        print("Configuration file updated")
        print(f"---------------------------------------------------------------------------------------------------------------")


    @staticmethod
    def read(filename):
        """
        Reads the configuration file.
        """
        with open(filename, 'r') as f:
            parser = edict(yaml.load(f, Loader=yaml.FullLoader))
        return parser

    @staticmethod
    def print_config(parser):
        """
        Prints the args.
        params:
            - parser: edict object of the config file
        """
        print("========== Configuration File ==========")
        for key in parser:
            print(f" - {key}: {parser[key]}")

    def get_config(self):
        """
        Returns the currently stored config in the object.
        """
        return self.args

    def export_config(self, filename):
        """
        Writes the arguments in the file specified by filename (includes path)
        """
        with open(filename, 'w') as f:
            yaml.dump(dict(self.args), f)


class logcat_parser:
    """
    Class containing all the methods that aid in parsing the logcat file.
    
    We parse the logcat file to extract the following information from it:
        1) Number of lines 
        2) Timestamp difference to see how long the application executes
        3) Rate at which logcat events are happening
    """
    @staticmethod
    def extract_events(logcat_filename):
        """
        Function to read a logcat file and extract the list of timestamps
        
        params:
            - logcat_filename: specify with file path

        Output:
            - timestamp_list: list of timestamp of the events
        """
        # List to store the extracted time stamps
        timestamp_list = []
        rfile = open(logcat_filename, 'r', errors='ignore') # Ignore the encoding errors. Ref: 'https://docs.python.org/3/library/functions.html#open'

        # Extract the time stamp from each of the logcat event and store it in a list
        while True:
            try: 
                line = rfile.readline()
                
            except Exception as e: # Ignore the lines that throw errors [encoding issues with chinese characters]
                print(e)
                print("*************** Ignoring error ***************")
                continue
            
            else:    
                # Regex to extract the time stamp
                logcat_obj = re.match( r'\d\d-\d\d (\d\d:\d\d:\d\d\.\d\d\d)', line, re.M|re.I)

                if(logcat_obj):
                    # Add the timestamp to a list
                    timestamp_list.append(logcat_obj.group(1))
                    
                if not line: # If line is empty, you have reached the end of the file. (readline() returns an empty string when it reaches end of file)
                    break

        rfile.close()
        # Return list of timestamp of the events
        return timestamp_list
    
    @staticmethod
    def get_time_difference(tstamp_list):
        """
        Function to read the timestamp list and get the timestamp difference between the last and the first event
        params:
            - tstamp_list: List of timestamps
        Output:
            - timestamp difference
        """
        if (len(tstamp_list) > 0): # Need to have atleast one event in logcat to get the time difference
            start_time = tstamp_list[0]
            end_time = tstamp_list[-1]
            time_format = '%H:%M:%S.%f'
            
            t_delta = datetime.strptime(end_time, time_format) - datetime.strptime(start_time, time_format)

            # Corner case where interval might cross midnight
            if t_delta.days < 0:
                t_delta = timedelta(
                    days = 0,
                    seconds= t_delta.seconds,
                    microseconds= t_delta.microseconds
                )

            return t_delta.total_seconds()
        
        else:
            return 0

    @staticmethod
    def get_logcat_lines(tstamp_list):
        """ 
        Function to return the number of lines in the timestamp list
        params: 
            - tstamp_list: List of timestamps
        """
        return len(tstamp_list)

    @staticmethod
    def get_average_frequency(tstamp_list):
        """
        Function to calculate the average frequency of events using the timestamp list
        params: 
            - tstamp_list: List of timestamps
        """
        # You need to have atleast 2 timestamps in the list to get the time difference
        if (len(tstamp_list) > 1): 
            time_format = '%H:%M:%S.%f'
            
            # Calculate the time difference between successive events and store the difference in a list
            time_dif_list = [(datetime.strptime(tstamp_list[i], time_format)-datetime.strptime(tstamp_list[i-1], time_format)).total_seconds() for i in range(1, len(tstamp_list))]

            # Time difference between successive events can be negative sometimes [logcat issue], so we take mod before averaging
            time_dif_list = [abs(dif) for dif in time_dif_list]

            # Get the mean of the time difference 
            mean_time_dif = statistics.mean(time_dif_list)

            # Inverse of the time difference gives average frequency
            avg_freq = 1/mean_time_dif

        else: 
            avg_freq = 0

        return avg_freq

class malware_label_generator:
    """
    Contains helper functions to generate VT reports for AVClass [https://github.com/malicialab/avclass] (for malware label generation)
    """
    @staticmethod
    def generate_hashlist(metainfo_path):
        """
        Generates the list of hashes from the metainfo file
        params:
            - metainfo_path: Path of the metainfo file from which the hash list needs to be extracted
        Output:
            - hashList: List of hashes extracted from the metainfo file
        """
        hashList = []
        
        with open(metainfo_path,'rb') as f:
            mInfo = json.load(f)

        for apkHash in mInfo:
            hashList.append(apkHash)

        return hashList

    @staticmethod
    def get_vt_report(hashList, outputFilePath):
        """
        Takes as input list of hashes and outputs a dict with key = hash and value = report
        
        params:
            - hashList: List of hashes for which the vt report needs to be generated
            - outputFilePath: Path of the output file where the report_dict will be dumped

        Output:
            - report_dict: key = hash and value = report
        """
        # Dict containing the hash: report pair [This is the final output]
        report_dict = {}
        # Checkpointing: If the report file already exists, then read it into report_dict 
        if os.path.isfile(outputFilePath):
            with open(outputFilePath,'rb') as fp:
                report_dict = json.load(fp)

        # VT api key
        API_KEY = '24e16060310e84f88d071e79d4050b4021acf677480609993cc368a5879bb0ce'

        #Instantiate the VT API   
        vt = VirusTotalPublicApi(API_KEY)

        # MAX_REQUEST in a day-delay between the sample is adjusted accordingly
        MAX_REQUEST=480

        for indx, hash in enumerate(hashList):
            # Checkpointing: If the report already exists in report_dict then skip this hash
            if hash in report_dict:
                continue

            response = vt.get_file_report(hash)         
            if(response['response_code']==200 and response['results']['response_code']==1):
                report_dict[hash] = response
                with open(outputFilePath,'w') as fp:
                    json.dump(report_dict,fp, indent=2)
                positive=response['results']['positives']
                total=response['results']['total']
                print (f"- [{indx}] Hash : {hash} | Num positives : {positive} | Total : {total}")                
            else:
                print(response)
                print(f"- [{indx}] Skipping this app BAD Request or Not available in the repo : {hash}")

            # We want MAX_REQUEST requests in 1 day    
            time.sleep(int(24*60*60.0/MAX_REQUEST))

        return report_dict

    @staticmethod
    def generate_vt_report_all_malware(metaInfoPath, outputReportPath):
        """
        Generates VT report for all the malware in the all the datasets: STD, CDyear1, CDyear2, CDyear3
        
        params:
            - metaInfoPath: Base directory of the folder where all the meta info files are stored
            - outputReportPath: Path where the vt report file will be stored
        """
        dataset_type = ["std_vt10","cd_year1","cd_year2","cd_year3","std"]

        # Generating a combined hash list containing hashes of malware in all the datasets
        hashListAllMalware = []
        for datType in dataset_type:
            mPath = os.path.join(metaInfoPath, f"meta_info_{datType}_malware.json")
            hashListAllMalware += malware_label_generator.generate_hashlist(metainfo_path = mPath)

        # Now generate the vt report by querying VirusTotal
        malware_label_generator.get_vt_report(hashList = hashListAllMalware, 
                                            outputFilePath = outputReportPath)

    @staticmethod
    def generate_vt_detection_distribution(VTReportPath):
        """
        Reads the virustotal report and outputs the distribution of the vt detections vs number of applications

        params:
            - VTReportPath: Path of the VT report
        Output:
            - detectionDistribution: Dict with key=vt detection and value= # of apks
        """
        detectionDistribution = {}

        # Get the report
        with open(VTReportPath,"rb") as f:
            vt_rep = json.load(f)

        for hash, vtReport in vt_rep.items():
            numPositives = vtReport['results']['positives']
            
            if numPositives in detectionDistribution:
                detectionDistribution[numPositives] += 1
            else:
                detectionDistribution[numPositives] = 1

        # Sort on the basis of num detections
        detectionDistribution = {k:v for k,v in sorted(detectionDistribution.items(), key = lambda item: item[1], reverse=True)}

        print(f"#Detections\t#Apks")
        for numDetection, numApps in detectionDistribution.items():
            print(f"{numDetection}\t{numApps}")
                
        return detectionDistribution


    @staticmethod
    def read_vt_and_convert_to_avclass_format(infile, outfile):
        """
        Reads the vt report and converts it into a format that Euphony can process.
        Euphony reades a sequence of reports from VirusTotal formatted as JSON records (one per line)

        params:
            - infile: Path of the vt report that should be converted to the simplified JSON format used by AVClass2 
            - outfile: Path of the output file
        """
        with open(infile,"rb") as f:
            vt_rep = json.load(f)

        # simplified json format used by avclass {md5, sha1, sha256, av_labels}
        avclass_list = []
        for _,vt_report in vt_rep.items():

            # Generate av labels for each antivirus
            avclass_avlabel_entry = []
            for av, avReport in vt_report["results"]["scans"].items():
                if avReport["detected"] == True:
                    avclass_avlabel_entry.append([av,avReport["result"]])
            
            # If no av detections then skip this file 
            if avclass_avlabel_entry:
                avclass_entry = {}
                avclass_entry["sha1"] = vt_report["results"]["sha1"]
                avclass_entry["md5"] = vt_report["results"]["md5"]
                avclass_entry["sha256"] = vt_report["results"]["sha256"]
                avclass_entry["av_labels"] = avclass_avlabel_entry
                avclass_list.append(avclass_entry)
        
        # Output the list into a file with one report per line
        with open(outfile,'w') as f:
            f.write("\n".join(map(str,avclass_list)).replace("'",'"'))
                
        
def main():
    # Current directory [where the script is executing from]
    cur_path = os.path.dirname(os.path.realpath(__file__))
    
    # Base folder of xmd
    xmd_base_folder = cur_path.replace("/src","")
    
    # Folder storing the metaInfo files
    metaInfoPath = os.path.join(xmd_base_folder, "baremetal_data_collection_framework", "androzoo", "metainfo")

    # Path where the final vt report will be saved
    vtReportSavePath = os.path.join(xmd_base_folder,"res","virustotal","hash_VT_report_all_malware_vt10.json")

    # Generate the VT report
    eParse = malware_label_generator()
    eParse.generate_vt_report_all_malware(metaInfoPath = metaInfoPath, outputReportPath = vtReportSavePath)
    exit()
    # Get the detection distribution
    eParse.generate_vt_detection_distribution(VTReportPath="/data/hkumar64/projects/arm-telemetry/xmd/res/virustotal/hash_VT_report_all_malware.json")
    ########################################## Generating VT report for feeding to AVClass ################################################################
    # eParse.read_vt_and_convert_to_avclass_format(infile= "/data/hkumar64/projects/arm-telemetry/xmd/res/virustotal/hash_virustotal_report_malware", 
    #                                             outfile="/data/hkumar64/projects/arm-telemetry/xmd/res/virustotal/avclass_virustotal_report_malware.vt")
    #######################################################################################################################################################
if(__name__=="__main__"):
	main()

