# Host-Client bare-metal sandbox
This is the codebase for the bare-metal sandbox built for performing large-scale automated collection of hardware-telemetry logs. The logs are used to train Machine Learning models for classifying benign and malicious workloads.

The Client is a Google Pixel-3 mobile device (running Android OS) on which the benign and malicious samples are executed. The Host is a Linux OS-based PC that orchestrates the data collection. We leverage Android Debug Bridge [ADB](https://developer.android.com/studio/command-line/adb) for sending commands and transferring data between the host and the client. The network packets from the client were routed through the Host using [Gnirehtet](https://github.com/Genymobile/gnirehtet), providing unfiltered Internet access to the Client, which is crucial for the malware to perform essential functionalities like communicating with their command-and-control (C2) server. Since we use a bare-metal analysis environment, a custom checkpointing scheme is developed to prevent contamination of data from persistent malicious workloads.

## APKs considered in the dataset.
We execute malware and benign Android apks that are downloaded from [Androzoo](https://androzoo.uni.lu). The malware apks have 2 or more VirusTotal detections. The benign apks belong to top 50 of their respective categories on Google Play Store. The apk files are further sub-divided into the following datasets:
- STD-Dataset: Used for training and tuning the ML models. This dataset has apks from 2018-01-01 to 2020-01-01. There are 2000 apks each for the malware and benign class.
- CD-Year1-Dataset: Used for evaluating the models trained on STD-Dataset. This dataset has apks from 2020-01-01 to 2021-01-01. There are 1000 apks each for the malware and benign class.
- CD-Year2-Dataset: Used for evaluating the models trained on STD-Dataset. This dataset has apks from 2021-01-01 to 2022-01-01. There are 1000 apks each for the malware and benign class.
- CD-Year3-Dataset: Used for evaluating the models trained on STD-Dataset. This dataset has apks from 2022-01-01 to 2023-01-01. There are 1000 apks each for the malware and benign class.

Please see [source code](/baremetal_data_collection_framework/androzoo/download_apk.py) for more details.

**NOTE:** We use the VirusTotal scan-date to timestamp the apk sample (instead of using the dex-date to timestamp the apk sample). As stated on the [Androzoo](https://androzoo.uni.lu) website, "the dex_date is mostly unusable nowadays: The vast majority of apps from Google Play have a 1980 dex_date". Therefore, we use the VT scan-date option which provides the "First Submission" date when the apk sample was uploaded on VT. While this may not be the most accurate metric, it provides a rough indication of when the malware was actively in circulation [Ref](https://www.sciencedirect.com/science/article/pii/S0957417422005863#!). 

## Data-Collection Architecture
### Data-Collection Module

### Interaction Module

### Orchestrator Module



