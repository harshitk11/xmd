#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <pthread.h>

#define _GNU_SOURCE
#define CLOCK CLOCK_MONOTONIC

/*
argv[1] : output log file in which you want to dump the dvfs data
argv[2] : duration for which you want to collect data
*/

int main(int argc,char** argv)
{
double elapsed=0;
double sTime,eTime,rTime;
char filedata[32];
char filedata1[32];
char filedata2[32];
char filedata3[32];
char filedata4[32];
char filedata5[32];
char filedata6[32];
char filedata7[32];
char filedata8[32];
char filedata9[32];
char filedata10[32];
char filedata11[32];
char filedata12[32];
char filedata13[32];
char filedata14[32];
char filedata15[32];
double data;
double data1;
double data2;
double data3;
double data4;
double data5;
double data6;
double data7;
double data8;
double data9;
double data10;
double data11;
double data12;
double data13;
double data14;
double data15;


struct timespec requestStart, requestEnd, req, refTime;
clock_gettime(CLOCK, &refTime);
FILE *writeStream = fopen(argv[1],"w");

while(1)
{
    clock_gettime(CLOCK, &requestStart);
    FILE *filestream = fopen("/sys/class/devfreq/5000000.qcom,kgsl-3d0/gpu_load","r");
    // CPUFreq data
    FILE *filestream1 = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq","r");
    FILE *filestream2 = fopen("/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq","r");
    // FILE *filestream3 = fopen("/sys/class/devfreq/aa00000.qcom,vidc:venus_bus_ddr/cur_freq","r");
    // FILE *filestream4 = fopen("/sys/class/devfreq/aa00000.qcom,vidc:venus_bus_llcc/cur_freq","r");
    FILE *filestream5 = fopen("/sys/class/devfreq/soc:qcom,cpubw/cur_freq","r");
    FILE *filestream6 = fopen("/sys/class/devfreq/soc:qcom,gpubw/cur_freq","r");
    FILE *filestream7 = fopen("/sys/class/devfreq/soc:qcom,kgsl-busmon/cur_freq","r");
    // FILE *filestream8 = fopen("/sys/class/devfreq/soc:qcom,l3-cdsp/cur_freq","r");
    FILE *filestream9 = fopen("/sys/class/devfreq/soc:qcom,l3-cpu0/cur_freq","r");
    FILE *filestream10 = fopen("/sys/class/devfreq/soc:qcom,l3-cpu4/cur_freq","r");
    FILE *filestream11 = fopen("/sys/class/devfreq/soc:qcom,llccbw/cur_freq","r");
    FILE *filestream12 = fopen("/sys/class/devfreq/soc:qcom,memlat-cpu0/cur_freq","r");
    FILE *filestream13 = fopen("/sys/class/devfreq/soc:qcom,memlat-cpu4/cur_freq","r");
    // FILE *filestream14 = fopen("/sys/class/devfreq/soc:qcom,mincpubw/cur_freq","r");
    // FILE *filestream15 = fopen("/sys/class/devfreq/soc:qcom,snoc_cnoc_keepalive/cur_freq","r");

    // To capture the network traffic flow
    // tun0 if for gnirehtet. If you are accessing internet using different network interface, then you need to modify this.
    FILE *filestream14 = fopen("/sys/class/net/tun0/statistics/rx_bytes","r");
    FILE *filestream15 = fopen("/sys/class/net/tun0/statistics/tx_bytes","r");

    // To capture the voltage_now and current_now values from the battery (will give the entire device level information)
    FILE *filestream3 = fopen("/sys/class/power_supply/battery/current_now","r");
    FILE *filestream4 = fopen("/sys/class/power_supply/battery/voltage_now","r");

    fgets(filedata,32,filestream);
    fgets(filedata1,32,filestream1);
    fgets(filedata2,32,filestream2);
    fgets(filedata3,32,filestream3);
    fgets(filedata4,32,filestream4);
    fgets(filedata5,32,filestream5);
    fgets(filedata6,32,filestream6);
    fgets(filedata7,32,filestream7);
    // fgets(filedata8,32,filestream8);
    fgets(filedata9,32,filestream9);
    fgets(filedata10,32,filestream10);
    fgets(filedata11,32,filestream11);
    fgets(filedata12,32,filestream12);
    fgets(filedata13,32,filestream13);
    fgets(filedata14,32,filestream14);
    fgets(filedata15,32,filestream15);


    sscanf(filedata,"%lf",&data);
    sscanf(filedata1,"%lf",&data1);
    sscanf(filedata2,"%lf",&data2);
    sscanf(filedata3,"%lf",&data3);
    sscanf(filedata4,"%lf",&data4);
    sscanf(filedata5,"%lf",&data5);
    sscanf(filedata6,"%lf",&data6);
    sscanf(filedata7,"%lf",&data7);
    // sscanf(filedata8,"%lf",&data8);
    sscanf(filedata9,"%lf",&data9);
    sscanf(filedata10,"%lf",&data10);
    sscanf(filedata11,"%lf",&data11);
    sscanf(filedata12,"%lf",&data12);
    sscanf(filedata13,"%lf",&data13);
    sscanf(filedata14,"%lf",&data14);
    sscanf(filedata15,"%lf",&data15);
    
    fclose(filestream);
    fclose(filestream1);
    fclose(filestream2);
    fclose(filestream3);
    fclose(filestream4);
    fclose(filestream5);
    fclose(filestream6);
    fclose(filestream7);
    // fclose(filestream8);
    fclose(filestream9);
    fclose(filestream10);
    fclose(filestream11);
    fclose(filestream12);
    fclose(filestream13);
    fclose(filestream14);
    fclose(filestream15);


    clock_gettime(CLOCK, &requestEnd);
    sTime = (requestStart.tv_sec) + (requestStart.tv_nsec)/1e9;
    eTime = (requestEnd.tv_sec) + (requestEnd.tv_nsec)/1e9;
    // fprintf(writeStream,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",sTime,eTime,data,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15);
    fprintf(writeStream,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",sTime,eTime,data,data1,data2,data5,data6,data7,data9,data10,data11,data12,data13,data14,data15,data3,data4);

    elapsed = ( requestEnd.tv_sec - refTime.tv_sec ) / 1e-6
                 + ( requestEnd.tv_nsec - refTime.tv_nsec ) / 1e3;
    if (elapsed > atoi(argv[2]))
    break;
    else
    continue;
}
fclose(writeStream);
return 0;
}
