# Repository for the codebase and the dataset of XMD.

Steps to generate the dataset:
1. Create json with info about the different runs: The first step is to parse the logcat files and create json file which has the apk folder location and the different attributes of the logcat files based on which filtering will be performed.
    - To do this we run `python dropbox_module.py`
        - This will generate json files (parser_info_*) in the directory `res/parser_info_files` which will be used for filtering out the logs of application runs with not sufficient runtimes.

2. Filter out the apk runs using filter values for run-time, and download the shortlisted runs from Dropbox.
    - To do this we run `python create_dataset.py`
        - Specifically, we run the `generate_dataset()` method of the `dataset_generator` class. The filter values are passed as arguments when an instance of the `dataset_generator` class is created. 


## Functionalities provided by each script in [source directory](/src)
- `parse_logcat.py`:-
    1. `logcat_parser`: Class to parse the logcat file to extract the following information from it: (1) Number of lines,  (2) Timestamp difference to see how long the application executes, and (3) Rate at which logcat events are happening.
    2. 'analyse_logcat_json' : To plot and observe the statistical trends from all the parsed logcat files.


- `dropbox_module.py`:-
    1. `create_parser_info_for_all_datasets()`: Python module to generate parser_info dict for all the datasets: `std-dataset`, `cd-dataset`, and `bench-dataset`.

        parser_info : dict for storing all the parsed information for each of the apk folder
                    ## key = Path of apk logcat folder (Contains the apk name)
                    ## value = [Number of logcat files, {logcat_file_1: [avg_freq, num_logcat_lines, time_diff]}, {logcat_file_2: [avg_freq, num_logcat_lines, time_diff]}, ...]

- `create_dataset.py`:- 
    1. `dataset_generator` : Class that filters out the runs (based on the filter) and downloads the dataset from Dropbox.
