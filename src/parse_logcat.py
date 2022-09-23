# Script to parse the logcat file to extract the following information from it:
# 1) Number of lines 
# 2) Timestamp difference to see how long the application executes
# 3) Rate at which logcat events are happening

import re
from datetime import datetime
from datetime import timedelta
import statistics

# Class containg all the methods that aid in parsing the logcat file
class logcat_parser:
    
    # Function to read a logcat file and extract the list of timestamps
    # Argument : filename (specify with file path)
    @staticmethod
    def extract_events(logcat_filename):
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
    #-------------------------------------------------------------------------------------------------------------------#
    # Function to read the timestamp list and get the timestamp difference between the last and the first event
    # Argument : List of timestamps
    @staticmethod
    def get_time_difference(tstamp_list):
        if (len(tstamp_list) > 0): # Need to have atleast one event in logcat to get the time difference
            start_time = tstamp_list[0]
            end_time = tstamp_list[-1]
            time_format = '%H:%M:%S.%f'
            
            # print(datetime.strptime(end_time, time_format))
            # print(datetime.strptime(start_time, time_format))
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

    #-------------------------------------------------------------------------------------------------------------------#
    # Function to return the number of lines in the timestamp list
    # Argument : List of timestamps
    @staticmethod
    def get_logcat_lines(tstamp_list):
        return len(tstamp_list)

    #-------------------------------------------------------------------------------------------------------------------#
    # Function to calculate the average frequency of events using the timestamp list
    # Argument : List of timestamps
    @staticmethod
    def get_average_frequency(tstamp_list):
        if (len(tstamp_list) > 1): # You need to have atleast 2 timestamps in the list to get the time difference
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
    #-------------------------------------------------------------------------------------------------------------------#    



def main():
    # print(logcat_parser.extract_time_difference(logcat_parser.extract_events('/data/hkumar64/projects/arm-telemetry/codes/_com_dealmoon_android_logcat_iter_0_rn3.txt')))
    print(logcat_parser.get_average_frequency(logcat_parser.extract_events('/data/hkumar64/projects/arm-telemetry/logcat_files/results_android_zoo_malware_all_rerun/com_taomi_beauty_130AD68D523D08F47F2A776CFFBE62E4645AD3E6629FF741654DCFC8361A6F1A.apk/logcat/_com_taomi_beauty_logcat_iter_0_rn1.txt')))
    # print(logcat_parser.get_logcat_lines(logcat_parser.extract_events('/data/hkumar64/projects/arm-telemetry/codes/_com_dealmoon_android_logcat_iter_0_rn3.txt')))
    
if(__name__=="__main__"):
    main()

