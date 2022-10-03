import os 
import sys
import time

# arg1 for running the program
run = int(sys.argv[1])
# arg 2 for pulling the output file to laptop
adb_pull = int(sys.argv[2])
# arg 3 for removing the output file in the phone (to avoid cluttering)
remove = int(sys.argv[3])
# arg 4 is the destination folder in your computer where the data is to  pulled into
dest_folder = sys.argv[4]
# argv 5 is package_name 
package_name=sys.argv[5]

#Iteration number
iter_number = int(sys.argv[6])

#rum_time in sec
run_time = int(sys.argv[7])

# PID of the package
pid = sys.argv[8]
#-----------------------------------------------------------------------------------------------#
# One iteration of collection data with the iteration number labelled by 'iter_number' and 'run'. Maintain same labelling convention for every channel (dvfs,perf,network)

remote_location = '/data/local/tmp/'  # Location in the android device where the output log is saved
name_of_file = '_'+package_name.replace(".","_")+'_logcat_iter_'+str(iter_number)+'_rn'+str(run)+'.txt' # This is just the name of the output file


outfile = remote_location+name_of_file # Remote location + name of the output log


if run:	
	# Clear the logcat buffer
	os.system('adb shell logcat -b all -c')

	t_start=time.asctime( time.localtime(time.time()))

	# Start logcat 
	os.system('adb shell timeout '+str(run_time)+ " logcat --pid=" +pid+ " -f "+outfile)
	
if adb_pull:
	# If destination directory doesn't exist then make one and then pull
	if(not os.path.isdir(dest_folder)):
		os.system('mkdir '+dest_folder)
	
	# Now pull the output log
	os.system('adb pull '+outfile+' '+dest_folder + '/' + name_of_file)		

if remove:
	#print('Removing file')
	os.system('adb shell \"su -c \'rm '+outfile+"\'\"")	

#Add Time of execution to the first line of the log_file
os.system("sed -i \'1s/^/"+t_start+"\\n/\' "+dest_folder+"/"+name_of_file)



