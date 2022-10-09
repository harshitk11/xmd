"""
Script to download the malware and the benign apks from androzoo.
"""
import csv
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
                if nApk > num_files:
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

    

def main():
    # Filter lookup table for the different datasets
    filter_dict = {
        0:{"name":'std_benign', "num_apk":2000},
        1:{"name":'std_malware', "num_apk":2000},
        2:{"name":'cd_year1_benign', "num_apk":1000},
        3:{"name":'cd_year1_malware', "num_apk":1000},
        4:{"name":'cd_year2_benign', "num_apk":1000},
        5:{"name":'cd_year2_malware', "num_apk":1000},
        6:{"name":'cd_year3_benign', "num_apk":1000},
        7:{"name":'cd_year3_malware', "num_apk":1000}
    }

    # Current directory [where the script is executing from]
    cur_path = os.path.dirname(os.path.realpath(__file__))

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
        apk_list, meta_info = read_csvfile(filter=get_filter(dataset_type["name"]), num_files=dataset_type["num_apk"])
        
        # Save the meta_info
        with open(os.path.join(metInfo_path, f"meta_info_{dataset_type['name']}.json"),"w") as f:
            json.dump(meta_info, f, indent=4)

        # Download the files
        download_apks(apk_list, download_path)
    
    
if(__name__=="__main__"):
	main()



