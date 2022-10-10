# Data Collection Framework used to collect hardware telemetry logs from a Google Pixel-3 Android Smartphone

The data-collection architecture consists of three main components: the interaction module, the data-collection module, and the orchestrator module.
- The interaction module interacts with the application while it executes.
- The data-collection module collects the hardware telemetry logs in the bakcground while the application is executing in the foreground.
- The orchestrator is responsible for synchronizing all the different sub-tasks.

## Data-Collection Module

## Interaction Module

## Orchestrator Module

## APKs considered in the dataset.
The malware and the benign apks are downloaded from Androzoo. For each of the dataset, we apply the following filters:
- STD-Dataset.
    - Benign apks:
    - Malware apks:
- CD-Dataset.
    - Benign apks:
    - Malware apks:

NOTE: For the Concept Drift study, instead of using the dex-date to timestamp the apk sample, we use the VT scan-date to timestamp the apk sample. As stated on the Androzoo website, "the dex_date is mostly unusable nowadays: The vast majority of apps from Google Play have a 1980 dex_date". Therefore, we use the VT scan-date option which provides the "First Submission" date when the apk sample was uploaded on VT. While this may not be the most accurate metric, it provides a rough indication of when the malware was actively in circulation [Ref](https://www.sciencedirect.com/science/article/pii/S0957417422005863#!). 


