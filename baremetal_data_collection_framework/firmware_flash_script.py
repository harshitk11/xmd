"""Script to flash the stock firmware of Pixel and flash the data partition using TWRP recovery."""

# First we flash the stock firmware of Google Pixel 3 with the patched boot.img file

# NOTE: ********************* RUN THIS FILE IN ROOT PRIVILEGE (else fastboot and adb won't work) ***************************** #

import os
import sys
import time

def flash_all_partition_except_data():
	# Need to reboot into the bootloader first
	print("Starting ADB server")
	os.system('adb start-server')

	time.sleep(2)

	print("ADB reboot bootloader <---------- command sent")
	os.system('adb reboot bootloader')
	print("ADB reboot bootloader <---------- command complete")

	time.sleep(2)

	# Then we use the flash-all script to flash the stock firmware
	# In the flash-all script you have to take care of the following two things:
	# 1) Make sure you are flashing the zip file with the patched boot img (for root access)
	# 2) You have to ensure that the '-w' flag is not there while flashing else you will wipe out the entire user data (this removes usb debugging and all your system settings)

	print("---> Flashing stock firmware with the patched boot img")
	os.system("cd /home/harshit/research/rooting_pixel/stock_pixel_image/blueline-pq3a.190801.002; ./flash-all.sh")
	print("---> Flashing stock firmware ended")
	time.sleep(60)  # Time it takes to reboot

def flash_data_partition():	
	#**************************************************************************************************************************************************************************#

	# Now you need to flash the data.img using twrp recovery (for restoring system settings)
	print("***********************FLASHING THE DATA IMG USING TWRP RECOVERY***********************")

	# First you go into the bootloader
	print("ADB reboot bootloader <---------- command sent")
	os.system('adb reboot bootloader')
	print("ADB reboot bootloader <---------- command complete")

	# Then you boot to twrp recovery using the img file on your local machine
	print("fastboot boot <twrp_img_location> <---------- command sent")
	os.system("cd /home/harshit/research/rooting_pixel; fastboot boot twrp-3.3.0-0-blueline.img")
	print("fastboot boot <twrp_img_location> <---------- command complete")

	# Booting into twrp recovery takes time, so wait for some time
	time.sleep(30) 

	# Restore the data img file
	print("---> Flashing data img started")
	os.system("adb shell twrp restore /data/media/0/TWRP/data_backup D")    # If you backup file changes, then you need to modify this.
	print("---> Flashing data img ended")

	time.sleep(15)

	# Reboot now
	print("---> Rebooting")
	os.system("adb shell twrp reboot")

	time.sleep(35)

	print("*************** Phone restoration complete ***************")


def flash_all():
	flash_all_partition_except_data()
	flash_data_partition()

def main():
	flash_all()
	# flash_data_partition()


if(__name__=="__main__"):
	main()
