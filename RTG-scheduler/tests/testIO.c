

#include "RTGS.h"
#include "RTGS_Global.h"
#include "string.h"

#include "../source/RTGS-file_handler.c"
#include "../source/RTGS-mode_1.c"



int main(int argc, char * argv[])
{
	RTGS_Status status = RTGS_SUCCESS;
	bool simulation = true, hardwareSupport = false;
	char *hardwareMode = NULL;
	int schedulerMode = 0;
	int error = 0;

	// global vaiable intitialize 
	GLOBAL_RTGS_MODE = 1;
	GLOBAL_KERNEL_FILE_NAME = NULL;
	GLOBAL_MAX_PROCESSORS = -1;
	GLOBAL_DELAY_SCHEDULE_PROCESSOR_LIMIT = -1;

	// get default debug msg control
	GLOBAL_RTGS_DEBUG_MSG = 2;
	char textBuffer[1024];
	// char jobsListFileName[] = "../testData/set1-jobs.txt";
    // char releaseTimeFilename[] = "../testData/set1-jobReleaseTimes.txt";
	char *jobsListFileName = NULL, *releaseTimeFilename = NULL;

	jobsListFileName = argv[2];
	releaseTimeFilename = argv[4];
	
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
	status = scheduler_main(jobsListFileName, releaseTimeFilename, 1); // scheduler call
	int64_t end_t = RTGS_GetClockCounter();

	PROFILER_STOP(SRTG, RTG_Schedule)
	PROFILER_SHUTDOWN();
	status = scheduler_main(jobsListFileName, releaseTimeFilename, 1);
	printf("The Scheduler Mode 1 returned Status ->%d\n", status);
}
