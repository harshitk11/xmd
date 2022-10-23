"""
Contains all the utility classes and functions.
"""

import argparse
import yaml
from easydict import EasyDict as edict
import re
from datetime import datetime
from datetime import timedelta
import statistics


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
        print(f"------------------------------------------------------------------------")
        print(f"Base configuration file : {file}")
        for key,val in self.args.items():
            print(f"{key}: {val}")
        print(f"------------------------------------------------------------------------")
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
            print(f"{key} : {val}")

        print("Configuration file updated")
        print(f"------------------------------------------------------------------------")


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
            print(f"{key}: {parser[key]}")

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
