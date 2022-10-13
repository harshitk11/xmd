"""
Contains all the utility classes and functions.
"""

import argparse
import yaml
from easydict import EasyDict as edict

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

