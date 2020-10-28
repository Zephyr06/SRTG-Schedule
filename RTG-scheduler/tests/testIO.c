

#include "RTGS.h"
#include "RTGS_Global.h"
#include "string.h"

#include "../source/RTGS-file_handler.c"
#include "../source/RTGS-mode_1.c"



int main(int argc, char * argv[])
{
	GLOBAL_RTGS_DEBUG_MSG = 2;
	char jobsListFileName[] = "../testData/set1-jobs.txt";
    char releaseTimeFilename[] = "../testData/set1-jobReleaseTimes.txt";
	
	int schedulerMode=1;
	// profiler  - output name initialize, profiler initialize and shutdown
	GLOBAL_RTGS_MODE = schedulerMode;
	GLOBAL_KERNEL_FILE_NAME = jobsListFileName;
	if(GLOBAL_MAX_PROCESSORS == -1){ GLOBAL_MAX_PROCESSORS = MAX_GPU_PROCESSOR; }
	if(GLOBAL_DELAY_SCHEDULE_PROCESSOR_LIMIT == -1){ 
		GLOBAL_DELAY_SCHEDULE_PROCESSOR_LIMIT = (int) floor(GLOBAL_MAX_PROCESSORS * 0.6);
	}
	PROFILER_FILE_INITIALIZE(schedulerMode, jobsListFileName);
	PROFILER_INITIALIZE();

	// count the number of profiler, if it's smaller than the maximum possible, then count+1
	// else, shut down the profiler
	PROFILER_START(SRTG, RTG_Schedule) 

	int64_t start_t = RTGS_GetClockCounter();
	RTGS_Status status = scheduler_main(jobsListFileName, releaseTimeFilename, 1); // scheduler call
	int64_t end_t = RTGS_GetClockCounter();

	PROFILER_STOP(SRTG, RTG_Schedule)
	PROFILER_SHUTDOWN();
	status = scheduler_main(jobsListFileName, releaseTimeFilename, 1);
	printf("The Scheduler Mode 1 returned Status ->%d\n", status);
}
