# Host-Client bare-metal sandbox
This is the codebase for the bare-metal sandbox built for performing large-scale automated collection of hardware-telemetry logs. The logs are used to train Machine Learning models for classifying benign and malicious workloads.

The Client is a Google Pixel-3 mobile device (running Android OS) on which the benign and malicious samples are executed. The Host is a Linux OS-based PC that orchestrates the data collection. We leverage Android Debug Bridge [ADB](https://developer.android.com/studio/command-line/adb) for sending commands and transferring data between the host and the client. The network packets from the client were routed through the Host using [Gnirehtet](https://github.com/Genymobile/gnirehtet), providing unfiltered Internet access to the Client, which is crucial for the malware to perform essential functionalities like communicating with their command-and-control (C2) server. Since we use a bare-metal analysis environment, a custom checkpointing scheme is developed to prevent contamination of data from persistent malicious workloads.

***The codebase is meant to act like a reference that can be used when designing a bare-metal data-collection framework for automated large-scale collection of telemetry for malicious Android-OS-based applications.***

## APKs considered in the dataset.
We execute malware and benign Android apks that are downloaded from [Androzoo](https://androzoo.uni.lu). The malware apks have 2 or more VirusTotal detections. The benign apks belong to top 50 of their respective categories on Google Play Store. The apk files are further sub-divided into the following datasets:
- **STD-Dataset**: Used for training and tuning the ML models. This dataset has apks from 2018-01-01 to 2020-01-01. There are 2000 apks each for the malware and benign class.
- **CD-Year1-Dataset**: Used for evaluating the models trained on STD-Dataset. This dataset has apks from 2020-01-01 to 2021-01-01. There are 1000 apks each for the malware and benign class.
- **CD-Year2-Dataset**: Used for evaluating the models trained on STD-Dataset. This dataset has apks from 2021-01-01 to 2022-01-01. There are 1000 apks each for the malware and benign class.
- **CD-Year3-Dataset**: Used for evaluating the models trained on STD-Dataset. This dataset has apks from 2022-01-01 to 2023-01-01. There are 1000 apks each for the malware and benign class.

Please see [source code](/baremetal_data_collection_framework/androzoo/download_apk.py) for details on dataset creation. More info about the apks in the dataset can be found in the [meta-info files](/baremetal_data_collection_framework/androzoo/metainfo).

**NOTE:** We use the VirusTotal scan-date to timestamp the apk sample (instead of using the dex-date to timestamp the apk sample). As stated on the [Androzoo](https://androzoo.uni.lu) website:
> the dex_date is mostly unusable nowadays: The vast majority of apps from Google Play have a 1980 dex_date. 

Therefore, we use the VT scan-date option which provides the "First Submission" date when the apk sample was uploaded on VT. While this may not be the most accurate metric, it provides a rough indication of when the apk was actively in circulation [[Ref.]](https://www.sciencedirect.com/science/article/pii/S0957417422005863#!). 

## Data-Collection Architecture
The data collection architecture consists of three main components: the interaction module, the data-collection module, and the orchestrator module. The interaction module interacts with the application while it executes, the data-collection module collects the data logs in the background while the application is executing in the foreground, and the orchestrator is responsible for synchronizing sub-tasks. [Figure](/baremetal_data_collection_framework/data-collection-flowchart-cropped.pdf) summarizes the overall flow of profiling and collecting the hardware telemetry logs for a single iteration of data collection.

### Orchestrator Module
- `orchestrator.py` : Python script that executes on the Host. It is responsible for invocation and synchronization of the execution of the apks, the Data-Collection module, and the Interaction module. The orchestrator also performs the checksum extraction to verify the integrity of the userdata and system partitions and pushes the collected data logs to the cloud.
### Data-Collection Module
The following scripts contain the data-collection functionalities:
- `get_devfreq_val.c` : Used to read the operation states of all the GLOBL channels. The script is cross-compiled using the [NDK toolchain](https://developer.android.com/ndk/guides/other_build_systems) and runs natively in the Google Pixel-3 device.
- `main_dvfs_channel.py` : Used for executing the cross-complied binary on the Client, and pulling the GLOBL channel logs from the Client to the Host.
- `main_simpleperf.py` : Used for executing simpleperf on the Client to collect the Hardware Performance Counter (HPC) logs, and pull the logs from the Client to the Host.
- `main_logcat.py` : Used for collecting the logcat logs on the Client and pull the logs from the Client to the Host.
- `main_comb_devfreq_simpleperf.py` : Syncs the collection of HPC logs and GLOBL channel logs.
### Interaction Module
- `orchestrator.py` : `interact_with_app_touch_events()` and `interact_with_app_broadcast_events()` are the two methods used for interacting with the application while it is executing. The goal of the interaction is to ensure the activation of malicious payloads.
### Checkpointing scheme
- `firmware_flash_script.py` : Custom checkpointing scheme to restore the client to its clean state after data-collection was performed on an apk sample.




