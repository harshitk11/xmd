"""
Script to download the malware and the benign apks from androzoo.
"""
import csv
from genericpath import isfile
import os
import json

def read_csvfile(filter, num_files):
    """
    Reads the row of a very large csv file
    params:
        - filter: dict storing the filter values used for downloading the apks
        - num_files: number of hashes to be returned in the list

    Output:
        - apk_list : List of hashes (apks)
        - meta_info : Dict with key = hash and value = meta-info of the apk (sha256, sha1, md5, apk_size, dex_size, dex_date, pkg_name, vercode, 
                                                                            vt_detection, vt_scan_date, markets)
    """
    
    apk_list = []
    meta_info = {}

    # This file needs to be downloaded from Androzoo (https://androzoo.uni.lu/lists)
    filename = "latest.csv"
    # To track the number of apks added in the list
    nApk = 0

    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)

        # Skip the header row
        next(datareader)
        
        for indx,row in enumerate(datareader):
            try:
                ############################## Get the info from the row ##############################
                vt_detection = int(row[7])
                vt_scan_date = row[8]
                # market_flag set to True only if the target market is found in the list of markets
                market_flag = any([m == filter["market"] for m in row[10].split("|")])
                #######################################################################################

                # Check if the row passes the filter
                if filter["vt_detection_threshold_low"] == None:
                    # Apply filter for benign applications
                    if ((vt_detection == filter["vt_detection_threshold_high"]) and 
                        (vt_scan_date > filter["vt_scandate_threshold_start"]) and 
                        (vt_scan_date < filter["vt_scandate_threshold_end"]) and (market_flag)):
                        
                        # Add the apk hash to the list
                        apk_list.append(row[0])
                        meta_info[row[0]] = row
                        nApk += 1
                        print(f"[{nApk}] Adding {row[0]} to the list")
                
                elif filter["vt_detection_threshold_high"] == None:
                    # Apply filter for malware applications
                    if ((vt_detection >= filter["vt_detection_threshold_low"]) and 
                        (vt_scan_date > filter["vt_scandate_threshold_start"]) and 
                        (vt_scan_date < filter["vt_scandate_threshold_end"]) and (market_flag)):
                        
                        # Add the apk hash to the list
                        apk_list.append(row[0])
                        meta_info[row[0]] = row
                        nApk += 1
                        print(f"[{nApk}] Adding {row[0]} to the list")


                # If we have enough apks, then terminate
                if nApk >= num_files:
                    break
            
            except:
                continue

    return apk_list, meta_info
    

def get_filter(dataset_type):
    """
    Returns the filter for the dataset_type provided as input.
    params:
        - dataset_type : Can take one of the following values {std_benign, std_malware, cd_year1_benign, cd_year1_malware,
                                                                cd_year2_benign, cd_year2_malware, cd_year3_benign, cd_year3_malware}

    Output:
        - filter : Dict representing the filter values
    """
    ############################### Filters for the STD-Dataset ###############################
    if dataset_type=='std_benign':
        std_benign_filter = {"vt_detection_threshold_low":None,    # For confirming malware
                        "vt_detection_threshold_high":0,    # For identifying malware and benign apks
                        "vt_scandate_threshold_start":'2018-01-01 00:00:00',    # For the concept drift study
                        "vt_scandate_threshold_end":'2020-01-01 00:00:00',
                        "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return std_benign_filter
    
    elif dataset_type=='std_malware':
        std_malware_filter = {"vt_detection_threshold_low":2,    # For confirming malware
                        "vt_detection_threshold_high":None,    # For identifying malware and benign apks
                        "vt_scandate_threshold_start":'2018-01-01 00:00:00',    # For the concept drift study
                        "vt_scandate_threshold_end":'2020-01-01 00:00:00',
                        "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return std_malware_filter
    
    ############################### Filters for the CD-Dataset year 1 ###############################
    elif dataset_type=='cd_year1_benign':
        cd_year1_benign_filter = {"vt_detection_threshold_low":None,    # For confirming malware
                            "vt_detection_threshold_high":0,    # For identifying malware and benign apks
                            "vt_scandate_threshold_start":'2020-01-01 00:00:00',    # For the concept drift study
                            "vt_scandate_threshold_end":'2021-01-01 00:00:00',
                            "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return cd_year1_benign_filter
    
    elif dataset_type=='cd_year1_malware':
        cd_year1_malware_filter = {"vt_detection_threshold_low":2,    # For confirming malware
                            "vt_detection_threshold_high":None,    # For identifying malware and benign apks
                            "vt_scandate_threshold_start":'2020-01-01 00:00:00',    # For the concept drift study
                            "vt_scandate_threshold_end":'2021-01-01 00:00:00',
                            "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return cd_year1_malware_filter
    
    ############################### Filters for the CD-Dataset year 2 ###############################
    elif dataset_type=='cd_year2_benign':
        cd_year2_benign_filter = {"vt_detection_threshold_low":None,    # For confirming malware
                            "vt_detection_threshold_high":0,    # For identifying malware and benign apks
                            "vt_scandate_threshold_start":'2021-01-01 00:00:00',    # For the concept drift study
                            "vt_scandate_threshold_end":'2022-01-01 00:00:00',
                            "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return cd_year2_benign_filter
    
    elif dataset_type=='cd_year2_malware':
        cd_year2_malware_filter = {"vt_detection_threshold_low":2,    # For confirming malware
                            "vt_detection_threshold_high":None,    # For identifying malware and benign apks
                            "vt_scandate_threshold_start":'2021-01-01 00:00:00',    # For the concept drift study
                            "vt_scandate_threshold_end":'2022-01-01 00:00:00',
                            "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return cd_year2_malware_filter
        
    ############################### Filters for the CD-Dataset year 3 ###############################
    elif dataset_type=='cd_year3_benign':
        cd_year3_benign_filter = {"vt_detection_threshold_low":None,    # For confirming malware
                            "vt_detection_threshold_high":0,    # For identifying malware and benign apks
                            "vt_scandate_threshold_start":'2022-01-01 00:00:00',    # For the concept drift study
                            "vt_scandate_threshold_end":'2023-01-01 00:00:00',
                            "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return cd_year3_benign_filter

    elif dataset_type=='cd_year3_malware':    
        cd_year3_malware_filter = {"vt_detection_threshold_low":2,    # For confirming malware
                            "vt_detection_threshold_high":None,    # For identifying malware and benign apks
                            "vt_scandate_threshold_start":'2022-01-01 00:00:00',    # For the concept drift study
                            "vt_scandate_threshold_end":'2023-01-01 00:00:00',
                            "market":'play.google.com'}     # For identifying the market (to tackle the sampling bias)
        return cd_year3_malware_filter

def download_apks(apk_list, download_path):
    """
    Function to download the list of apks from Androzoo.
    params:
        - apk_list : List of hashes for the apk that needs to be downloaded
        - download_path : Download path on the local system
    """
    # Get the Androzoo API key
    with open('api_key','r') as f:
        APIKEY = f.read()

    # Download the apk files
    for SHA256 in apk_list:
        SHA256=SHA256.rstrip()
        path=os.path.join(download_path, f"{SHA256}.apk")
        cmd=f"curl -o {path} -O --remote-header-name -G -d apikey={APIKEY} -d sha256={SHA256} https://androzoo.uni.lu/api/download --max-time 900"
        print(f" - Downloading {SHA256}")
        os.system(cmd)

class top10k_benign_apps:
    @staticmethod
    def create_hashtable_from_androzooCSV(xmd_base_folder):
        """
        Reads the androzoo csv and creates a hash table with key=package_name. This will help in quickly accessing the entries of the csv file using the package name.
        NOTE: This is a memory intensive operation and requires >30 GB of available RAM.
        Output:
            - androzoo_hashtable: Key=package_name, Value=[csv_info1, ...] ---> Can have multiple entries in the list in case of hash collisions.
        """ 
        print(" - Creating hashtable from androzoo csv file.")
        androzoo_hashtable = {}

        filename = os.path.join(xmd_base_folder, "baremetal_data_collection_framework/androzoo", "latest.csv")
        with open(filename, "r") as csvfile:
            datareader = csv.reader(csvfile)

            #Skip the header row
            next(datareader)
            
            for indx,row in enumerate(datareader):
                try:
                    packageName = row[5]
                    print(f"[{indx}] Processing {packageName}")
                    if packageName in androzoo_hashtable:
                        # Hash collision. Append the info to the list.
                        androzoo_hashtable[packageName].append(row)
                    else:
                        # Add the info to the hash table
                        androzoo_hashtable[packageName] = [row]            
                except:
                    continue
               
        return androzoo_hashtable

    @staticmethod
    def match_top_apps_with_androzoo(xmd_base_folder):
        """
        Reads the list of top apps (present at /res/category_benign_malware_apk) and creates a hash table that merges the top apps info with 
        the info found on the androzoo csv.

        params:
            - xmd_base_folder: Location of the xmd base folder
        Output:
            - top_app_androzooInfo_dict: Hash table:-
                                    key = apk hash
                                    value = {"androzoo_info":[],
                                            "category": ... ,
                                            "rank": ...}
        """
        # Create hash table from the androzoo csv file
        androzoo_hashtable = top10k_benign_apps.create_hashtable_from_androzooCSV(xmd_base_folder=xmd_base_folder)

        # Output
        top_app_androzooInfo_dict = {}
        
        # Read top apps json file
        with open(os.path.join(xmd_base_folder,"res/category_benign_malware_apk/top_apps_metadata.json")) as f:
            topAppData = json.load(f)
        
        # For each app in the json file, fetch the corresponding info from androzoo csv file
        for apkDat in topAppData:
            if apkDat["pkg_name"] in androzoo_hashtable:
                # Entry found. Get the info of all the occurences.
                for androzoo_info in androzoo_hashtable[apkDat["pkg_name"]]:
                    shaHash =  androzoo_info[0]
                    # Add the entry
                    top_app_androzooInfo_dict[shaHash] = {"androzoo_info":androzoo_info, "category": apkDat["category"], "rank":apkDat["rank"]}
        
        # Save the dict
        with open(os.path.join(xmd_base_folder, "baremetal_data_collection_framework/androzoo", "top_app_androzoo_info.json"), 'w') as f:
            json.dump(top_app_androzooInfo_dict, f, indent=4)
        
        return top_app_androzooInfo_dict

    @staticmethod
    def get_merged_androzoo_top_app(xmd_base_folder):
        """
        If the top_app_androzoo_info.json file has been generated, then read it and return the dict.
        Else generate and return the dict.
        """
        if os.path.isfile(os.path.join(xmd_base_folder, "baremetal_data_collection_framework/androzoo", "top_app_androzoo_info.json")):
            with open(os.path.join(xmd_base_folder, "baremetal_data_collection_framework/androzoo", "top_app_androzoo_info.json"), 'r') as f:
                top_app_androzooInfo_dict = json.load(f)
                return top_app_androzooInfo_dict
        
        else:
            # Generate the json and return it
            return top10k_benign_apps.match_top_apps_with_androzoo(xmd_base_folder=xmd_base_folder)

    @staticmethod
    def check_category_violations(category_dict, category_name, category_limit=50):
        """
        Checks if the number of apps in a given category exceed the threshold. Increments the num apk if check is successful.
        Updates the category_dict as well.

        params:
            - category_dict: hash-table with key=category_name and value=# of apks in that category
            - category_limit: limit which should not be exceeded
            - category_name: name of the category for which check needs to be performed.

        Ouput: 
            - updated category_dict
            - True or False indicating if violation has happened
        """
        violation = False

        # Check if category_name exist in the dict
        if category_name in category_dict:
            # Check if limit has been violated
            if category_dict[category_name] <= category_limit:
                # No violation. Update the category dict
                category_dict[category_name] += 1 
            else:
                # Violation
                violation = True
        else:
            # Add the category in the dict
            category_dict[category_name] = 1

        return category_dict, violation    

    @staticmethod
    def generate_apklist_and_metainfo(xmd_base_folder, filter, num_files, dataset_type):
        """
        Downloads the benign apks and creates the corresponding meta-info files
        
        params:
            - xmd_base_folder:
            - filter: dict storing the filter values used for downloading the apks
            - num_files: number of hashes to be returned in the list
            - dataset_type: Type of dataset for which apklist is generated [Used for labelling the logs.]
        
        Output:
        - apk_list : List of hashes (apks)
        - meta_info : Dict with key = hash and value = meta-info of the apk (sha256, sha1, md5, apk_size, dex_size, dex_date, pkg_name, vercode, 
                                                                            vt_detection, vt_scan_date, markets)
        """
        apk_list = []
        meta_info = {}

        # Get the merged androzoo info and top app dict
        # top_app_androzooInfo_dict: Hash table:- key = apk-hash, value = {"androzoo_info":[], "category": ... , "rank": ...}
        top_app_androzooInfo_dict = top10k_benign_apps.get_merged_androzoo_top_app(xmd_base_folder=xmd_base_folder)

        # To track the packages that have been added in this dataset
        list_pkg_added = []
        # To track the number of apks added in the list
        nApk = 0
        
        # Category dict : key=category, val=# of apps included in the dataset that belongs to this category  
        # [Used for capping the number of apps from one category]
        category_dict = {}       
        
        for apkHash, apkInfo in top_app_androzooInfo_dict.items():
            try:
                # Packages should not be repeated [different versions of same package exist]
                packageName = apkInfo["androzoo_info"][5]
                if not packageName in list_pkg_added:
                    ############################## Get the info from the row ##############################
                    vt_detection = int(apkInfo["androzoo_info"][7])
                    vt_scan_date = apkInfo["androzoo_info"][8]
                    # market_flag set to True only if the target market is found in the list of markets
                    market_flag = any([m == filter["market"] for m in apkInfo["androzoo_info"][10].split("|")])
                    #######################################################################################

                    # Check if the apk passes the filter
                    if filter["vt_detection_threshold_low"] == None:
                        # Apply filter for benign applications
                        if ((vt_detection == filter["vt_detection_threshold_high"]) and 
                            (vt_scan_date > filter["vt_scandate_threshold_start"]) and 
                            (vt_scan_date < filter["vt_scandate_threshold_end"]) and (market_flag)):
                            
                            # Check if the number of apps in the category has exceeded the limit
                            category_dict, violationFlag = top10k_benign_apps.check_category_violations(category_dict=category_dict,
                                                                                                        category_name=apkInfo["category"],
                                                                                                        category_limit=50)
                            if not violationFlag:
                                # Add the apk hash to the list
                                apk_list.append(apkInfo["androzoo_info"][0])
                                meta_info[apkInfo["androzoo_info"][0]] = apkInfo["androzoo_info"]
                                nApk += 1
                                list_pkg_added.append(packageName)

                                print(f"[{nApk}] Adding {apkInfo['androzoo_info'][0]} to the list")
                
                # If we have enough apks, then terminate
                if nApk >= num_files:
                    break
            except:
                continue
        
        # Save the category dict
        with open(os.path.join(xmd_base_folder, "baremetal_data_collection_framework/androzoo/metainfo", f"benign_category_info_{dataset_type}.json"), 'w') as f:
            json.dump(category_dict, f, indent=4)

        # Print the category_dict
        for category,num_apk in category_dict.items():
            print(f"{category} -> {num_apk}")

        return apk_list, meta_info
        

def main():
    # Filter lookup table for the different datasets
    filter_dict = {
        0:{"name":'std_benign', "num_apk":2000 , "apk_type":"benign"},
        # 1:{"name":'std_malware', "num_apk":2000, "apk_type":"malware"},
        2:{"name":'cd_year1_benign', "num_apk":1000, "apk_type":"benign"},
        # 3:{"name":'cd_year1_malware', "num_apk":1000, "apk_type":"malware"},
        4:{"name":'cd_year2_benign', "num_apk":1000, "apk_type":"benign"},
        # 5:{"name":'cd_year2_malware', "num_apk":1000, "apk_type":"malware"},
        6:{"name":'cd_year3_benign', "num_apk":1000, "apk_type":"benign"},
        # 7:{"name":'cd_year3_malware', "num_apk":1000, "apk_type":"malware"}
    }


    # Current directory [where the script is executing from]
    cur_path = os.path.dirname(os.path.realpath(__file__))

    # Base folder of xmd
    xmd_base_folder = os.path.join(cur_path.replace("/baremetal_data_collection_framework/androzoo",""),"")

    # Directory for storing the meta-info
    metInfo_path = os.path.join(cur_path, "metainfo")
    os.system(f"mkdir -p {metInfo_path}")

    # Directory for storing the downloaded apks
    apkDownload_path = os.path.join(cur_path, "apks")
    os.system(f"mkdir -p {apkDownload_path}")

    for _, dataset_type in filter_dict.items():
        # Create the directory for this dataset
        download_path = os.path.join(apkDownload_path, dataset_type["name"])
        os.system("mkdir -p "+download_path)

        # Get the file-list and the meta-info for this dataset
        if dataset_type["apk_type"] == "malware":
            apk_list, meta_info = read_csvfile(filter=get_filter(dataset_type["name"]), num_files=dataset_type["num_apk"])
        elif dataset_type["apk_type"] == "benign":
            # For benign we are using the top 10k apps
            apk_list, meta_info = top10k_benign_apps.generate_apklist_and_metainfo(xmd_base_folder = xmd_base_folder, 
                                                                                filter=get_filter(dataset_type["name"]), 
                                                                                num_files=dataset_type["num_apk"],
                                                                                dataset_type=dataset_type["name"])
        # Save the meta_info
        with open(os.path.join(metInfo_path, f"meta_info_{dataset_type['name']}.json"),"w") as f:
            json.dump(meta_info, f, indent=4)

        # # Download the files
        # download_apks(apk_list, download_path)
    
    
if(__name__=="__main__"):
	main()



