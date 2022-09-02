# Repository for the codebase and the dataset of XMD.

Steps to generate the dataset:
1. The first step is to parse the logcat files and create json file which has the apk folder location and the different attributes of the logcat files based on which filtering will be performed.
    - To do this we run `python dropbox_module.py`
        - This will generate json files (parser_info_*) in the directory `res/parser_info_files` which will be used for filtering out the logs of application runs with not sufficient runtimes.