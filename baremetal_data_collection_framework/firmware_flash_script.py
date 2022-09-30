"""Script to flash the stock firmware of Pixel and flash the data partition using TWRP recovery."""

# First we flash the stock firmware of Google Pixel 3 with the patched boot.img file

# NOTE: ********************* RUN THIS FILE IN ROOT PRIVILEGE (else fastboot and adb won't work) ***************************** #

import time
import subprocess
import re




def check_device_boot_up():
	"""
	Function to check if the device has booted up.
	Returns True if the device has booted up, else False.	
	"""
	command = "adb devices"
	get_devices = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
	time.sleep(1)
	devices_string = get_devices.read().decode()
	
	# If the device has booted up and is available to adb, then it will return a string of length 46. 
	if len(devices_string) > 40:
		return True

	return False
		
def get_local_time():
	"""
	Returns the current time of the system in HMS format
	"""
	return time.strftime("%H:%M:%S", time.localtime())

def flash_all_partition_except_data():
	# Need to reboot into the bootloader first
	print(f"[{get_local_time()}] Starting ADB server")
	subprocess.Popen('adb start-server', shell=True).wait()

	time.sleep(2)

	print(f"[{get_local_time()}] ADB reboot bootloader <---------- command sent")
	subprocess.Popen('adb reboot bootloader', shell=True).wait()
	print(f"[{get_local_time()}] ADB reboot bootloader <---------- command complete")

	time.sleep(2)

	# Then we use the flash-all script to flash the stock firmware
	# In the flash-all script you have to take care of the following two things:
	# 1) Make sure you are flashing the zip file with the patched boot img (for root access)
	# 2) You have to ensure that the '-w' flag is not there while flashing else you will wipe out the entire user data (this removes usb debugging and all your system settings)

	print(f"[{get_local_time()}] ---> Flashing stock firmware with the patched boot img")
	subprocess.Popen("cd /home/harshit/research/rooting_pixel/stock_pixel_image/blueline-pq3a.190801.002; ./flash-all.sh", shell=True).wait()
	print(f"[{get_local_time()}] ---> Flashing stock firmware ended")
	
	time.sleep(10)
	# If the device has not booted up, then wait.
	while not check_device_boot_up():
		print(f"[{get_local_time()}] Waiting for the device to boot up.")
		time.sleep(2)


def flash_data_partition():	
	#**************************************************************************************************************************************************************************#

	# Now you need to flash the data.img using twrp recovery (for restoring system settings)
	print("***********************FLASHING THE DATA IMG USING TWRP RECOVERY***********************")

	# First you go into the bootloader
	print(f"[{get_local_time()}] ADB reboot bootloader <---------- command sent")
	subprocess.Popen('adb reboot bootloader', shell=True).wait()
	print(f"[{get_local_time()}] ADB reboot bootloader <---------- command complete")

	# Then you boot to twrp recovery using the img file on your local machine
	print(f"[{get_local_time()}] fastboot boot <twrp_img_location> <---------- command sent")
	subprocess.Popen("cd /home/harshit/research/rooting_pixel; fastboot boot twrp-3.3.0-0-blueline.img", shell=True).wait()
	print(f"[{get_local_time()}] fastboot boot <twrp_img_location> <---------- command complete")

	# Booting into twrp recovery takes time, so wait for some time
	time.sleep(30) 

	# Restore the data img file
	print(f"[{get_local_time()}] ---> Flashing data img started")
	subprocess.Popen("adb shell twrp restore /data/media/0/TWRP/data_backup D", shell=True).wait()
	print(f"[{get_local_time()}] ---> Flashing data img ended")

	time.sleep(5)

	# Reboot now
	print(f"[{get_local_time()}] ---> Rebooting")
	subprocess.Popen("adb shell twrp reboot", shell=True).wait()

	# Wait for the device to boot up
	time.sleep(15)

	# If the device has not booted up, then wait.
	while not check_device_boot_up():
		print(f"[{get_local_time()}] Waiting for the device to boot up.")
		time.sleep(2)

	print("*************** Phone restoration complete ***************")
	time.sleep(5)

def flash_all():
	flash_all_partition_except_data()
	flash_data_partition()

def main():
	# flash_all()
	flash_data_partition()


if(__name__=="__main__"):
	main()
