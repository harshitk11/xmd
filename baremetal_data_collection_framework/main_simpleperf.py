import os
import sys
import time

# runtime = 30000000

# arg1 is enable for run | arg2 is enable for adb_pull | arg3 is enable for removing log file from the Android device 
# arg4 is the package name to be monitored

run = int(sys.argv[1])
adb_pull = int(sys.argv[2])
remove = int(sys.argv[3])
package_name = sys.argv[4] # This needs to be extracted from the apk file

#destination folder at the laptop
dest_folder=sys.argv[5]

#This is the sampling interval (in ms)  
samp_time=sys.argv[6]

#Duration for which you want simpleperf to collect the data
run_time=sys.argv[7]

#Iteration number
iter_number=sys.argv[8]

#---------------------------------------------------------------------------------------------------------------------------#
# Routine for recording simpleperf data. This will collect one iteration of data that will be labelled by 'iter_number' and 'run'
# We have 12 performance counters, but we can collect only 3 of them at a time.
# So, we have 4 groups of 3 performance counters. We will collect one of the four groups indexed by the value of 'run'
# ----> run == 1 : cpu-cycles, instructions, raw-bus-access
# ----> run == 2 : branch-instructions, branch-misses, raw-mem-access
# ----> run == 3 : cache-references, cache-misses, raw-crypto-spec
# ----> run == 4 : bus-cycles, raw-mem-access-rd, raw-mem-access-wr

remote_location = '/data/local/tmp/'  # Location in the android device where the output log is saved
name_of_file = '_'+package_name.replace(".","_")+'_it'+str(iter_number)+'_rn'+str(run)+'.txt' # This is just the name of the output file

outfile = remote_location+name_of_file

if run == 1:
	t_start=time.asctime( time.localtime(time.time()) )
	os.system('adb shell \"su -c \'./data/local/tmp/simpleperf stat --app '+package_name+' -e cpu-cycles,instructions,raw-bus-access --duration '+str(run_time)+' --interval '+str(samp_time)+' --interval-only-values --csv -o '+outfile+"\'\"")

elif run == 2:
	t_start=time.asctime( time.localtime(time.time()) )
	os.system('adb shell \"su -c \'./data/local/tmp/simpleperf stat --app '+package_name+' -e branch-instructions,branch-misses,raw-mem-access --duration '+str(run_time)+' --interval '+str(samp_time)+' --interval-only-values --csv -o '+outfile+"\'\"")

elif run == 3:
	t_start=time.asctime( time.localtime(time.time()) )
	os.system('adb shell \"su -c \'./data/local/tmp/simpleperf stat --app '+package_name+' -e cache-references,cache-misses,raw-crypto-spec --duration '+str(run_time)+' --interval '+str(samp_time)+' --interval-only-values --csv -o '+outfile+"\'\"")

elif run == 4:
	t_start=time.asctime( time.localtime(time.time()) )
	os.system('adb shell \"su -c \'./data/local/tmp/simpleperf stat --app '+package_name+' -e bus-cycles,raw-mem-access-rd,raw-mem-access-wr --duration '+str(run_time)+' --interval '+str(samp_time)+' --interval-only-values --csv -o '+outfile+"\'\"")

else:
	sys.exit("Incorrect value of argument run")


if adb_pull:
	# If destination directory doesn't exist then make one and then pull
	if(not os.path.isdir(dest_folder)):
		os.system('mkdir -p '+dest_folder)
	
	# Now pull the output log
	os.system('adb pull '+outfile+' '+dest_folder + '/' + name_of_file)
	

if remove:
	os.system('adb shell \"su -c \'rm '+outfile+"\'\"")


#Add Time of execution to the first line of the log_file
os.system("sed -i \'1s/^/"+t_start+"\\n/\' "+dest_folder+"/"+name_of_file)
