/*
* RTGS-mode_1.c
*      Author: Kiriti Nagesh Gowda
*/

#include"RTGS.h"


struct streamNode {
	int job_id;
	struct streamNode *next;
};

/***********************************************************************************************************
MODE 1 FUNCTIONS
**********************************************************************************************************/
static int Mode_1_book_keeper
(
	jobAttributes* jobAttributesList,
	int jobNumber,
	int processors_available,
	int present_time,
	scheduledResourceNode** processorsAllocatedList
)
{
	int processorsInUse = 0, processorReleaseTime = 0;
	int presentTime = present_time;
	int scheduleMethod = RTGS_SCHEDULE_METHOD_NOT_DEFINED;
	if (GLOBAL_RTGS_DEBUG_MSG > 1) {
		printf("Mode-1 Book Keeper:: Job::%d --> processor_req:%d execution_time:%d, deadline:%d, latest_schedulable_time:%d\n", jobNumber, jobAttributesList[jobNumber].processor_req, jobAttributesList[jobNumber].execution_time, jobAttributesList[jobNumber].deadline, jobAttributesList[jobNumber].latest_schedulable_time);
	}
	// If processors available is greater than the required processors by the jobAttributesList
	if (jobAttributesList[jobNumber].processor_req <= processors_available)
	{
		if (jobAttributesList[jobNumber].execution_time + presentTime <= jobAttributesList[jobNumber].deadline)
		{
			processors_available = processors_available - jobAttributesList[jobNumber].processor_req;
			processorsInUse = jobAttributesList[jobNumber].processor_req;
			processorReleaseTime = jobAttributesList[jobNumber].execution_time + presentTime;
			scheduleMethod = RTGS_SCHEDULE_METHOD_IMMEDIATE;
			// Job call for the GPU to handle the given Jobs and number of blocks
			queue_job_execution(processorsInUse, processorReleaseTime, presentTime, scheduleMethod, jobNumber, processorsAllocatedList);

			jobAttributesList[jobNumber].schedule_hardware = 1;
			jobAttributesList[jobNumber].rescheduled_execution = -1;
			jobAttributesList[jobNumber].scheduled_execution = present_time;
			jobAttributesList[jobNumber].completion_time = jobAttributesList[jobNumber].execution_time + present_time;
			GLOBAL_GPU_JOBS++;
			if (GLOBAL_RTGS_DEBUG_MSG > 1) {
				printf("Mode-1 Book Keeper:: Jobs ACCEPTED count --> %d\n", GLOBAL_GPU_JOBS);
			}
		}
		else
		{
			jobAttributesList[jobNumber].schedule_hardware = 2;
			jobAttributesList[jobNumber].rescheduled_execution = -1;
			jobAttributesList[jobNumber].completion_time = -1;
			jobAttributesList[jobNumber].scheduled_execution = -1;
			GLOBAL_CPU_JOBS++;
			if (GLOBAL_RTGS_DEBUG_MSG > 1) {
				printf("Mode-1 Book Keeper:: Job-%d will not complete before it's deadline, Job REJECTED\n", jobNumber);
				printf("Mode-1 Book Keeper:: Jobs REJECTED count --> %d\n", GLOBAL_CPU_JOBS);
			}
		}
	}
	else
	{
		jobAttributesList[jobNumber].schedule_hardware = 2;
		jobAttributesList[jobNumber].rescheduled_execution = -1;
		jobAttributesList[jobNumber].completion_time = -1;
		jobAttributesList[jobNumber].scheduled_execution = -1;
		GLOBAL_CPU_JOBS++;
		if (GLOBAL_RTGS_DEBUG_MSG > 1) {
			printf("Mode-1 Book Keeper:: No GCUs Available for Job-%d, Job REJECTED\n", jobNumber);
			printf("Mode-1 Book Keeper:: Jobs REJECTED count --> %d\n", GLOBAL_CPU_JOBS);
		}
	}

	return processors_available;
}

/**********************************************************************************************************
RTGS Mode 1 - - Simple GPU Schedulers
***********************************************************************************************************/
int RTGS_mode_1(char *jobsListFileName, char *releaseTimeFilename)
{
	jobAttributes jobAttributesList[MAX_JOBS] = {{0}}; // array, processor_req, execution_time, deadline, latest_schedulable_time
	jobReleaseInfo releaseTimeInfo[MAX_JOBS] = {{0}}; // array , release_time and num_job_released
	scheduledResourceNode *processorsAllocatedList = NULL;

	// global variables initialized
	GLOBAL_GPU_JOBS = 0;
	GLOBAL_CPU_JOBS = 0;

	int processorsAvailable = GLOBAL_MAX_PROCESSORS;
	int jobNumber = 0;
	

	int kernelMax = get_job_information(jobAttributesList, jobsListFileName);
	if (kernelMax <= RTGS_FAILURE) { printf("ERROR - get_job_information failed with %d code\n", kernelMax); return  RTGS_FAILURE; }
	int maxReleases = get_job_release_times(releaseTimeInfo, releaseTimeFilename);
	if (maxReleases <= RTGS_FAILURE) { printf("ERROR - get_job_release_times failed with %d code\n", maxReleases); return  RTGS_FAILURE; }

	if (GLOBAL_RTGS_DEBUG_MSG > 1) {
		printf("\n**************** The GPU Scheduler will Schedule %d Jobs ****************\n", kernelMax);
	}

	int TotalJob = sizeof(jobAttributesList)/sizeof(jobAttributes);
	int jobCursor=0;
	int numReleases = 0;
	struct streamNode* streamHead=NULL;
	struct streamNode* streamCursor=NULL;

	for (int present_time = 0; present_time < MAX_RUN_TIME; present_time++)
	{
		// Freeing-up processors
		processorsAvailable = Retrieve_processors(present_time, processorsAvailable, &processorsAllocatedList);
		if (processorsAvailable < 0) { printf("Retrieve_processors ERROR- GCUs Available:%d\n", processorsAvailable); return RTGS_ERROR_NOT_IMPLEMENTED; }

// **********************************My code begins***************************
		if(jobCursor > TotalJob)
			{
				// std::cout<<"All the jobs have been scheduled"<<std::endl;
				break;
			}
		if(jobAttributesList[jobCursor].release_time >= present_time )
		{
			// add this job to stream list
			struct streamNode *link = (struct streamNode*) malloc(sizeof(struct streamNode));
			link->job_id=jobAttributesList[jobCursor].job_id;
			link->next=NULL;
			if(!streamHead)
				{
					streamHead=link;
					streamCursor=streamHead;
				}
			else
				{
					streamCursor->next=link;
					streamCursor=streamCursor->next;					
				}
		}
		else
		{
			// nothing new to process
			;
		}

		// retrieve the job with highest priority from the stream list, and send it to Mode_1_book_keeper
		{
			struct streamNode* head=streamHead;
			int maxP=-1;
			int highest_job=-1;
			while(head)
			{
				if(jobAttributesList[jobAttributesList[head->job_id]].priority > maxP)
				{
					maxP=jobAttributesList[jobAttributesList[head->job_id]].priority;
					highest_job = head->job_id;
				}
			}
		}

// **********************************My code ends***************************
		if (releaseTimeInfo[numReleases].release_time == present_time) {

			if (releaseTimeInfo[numReleases].num_job_released == 1)
			{
				if (GLOBAL_RTGS_DEBUG_MSG > 1) {
					printf("\nRTGS Mode 1 -- Total GCUs Available at time %d = %d\n", present_time, processorsAvailable);
					printf("RTGS Mode 1 -- Job-%d Released\n", jobNumber);
				}
				jobAttributesList[jobNumber].release_time = present_time;
				// handling the released jobAttributesList by the book-keeper
				int64_t start_t = RTGS_GetClockCounter();
				processorsAvailable = Mode_1_book_keeper(jobAttributesList, jobNumber, processorsAvailable, present_time,
					&processorsAllocatedList);
				int64_t end_t = RTGS_GetClockCounter();
				int64_t freq = RTGS_GetClockFrequency();
				float factor = 1000.0f / (float)freq; // to convert clock counter to ms
				float SchedulerOverhead = (float)((end_t - start_t) * factor);
				jobAttributesList[jobNumber].schedule_overhead = SchedulerOverhead;
				jobNumber++;
			}
			// currently, we don't consider other release situations

			}
			else { printf("RTGS Mode 1 ERROR --  RTGS_ERROR_NOT_IMPLEMENTED\n"); return RTGS_ERROR_NOT_IMPLEMENTED; }

			numReleases++;
			if (numReleases > maxReleases) {
				printf("RTGS Mode 1 ERROR --  Job Release Time exceded Max Releases\n");
				return RTGS_ERROR_INVALID_PARAMETERS;
			}
		}
	}

	if (maxReleases != 0) {

		if (GLOBAL_RTGS_DEBUG_MSG) {
			printf("\n******* Scheduler Mode 1 *******\n");
			printf("GCUs Available -- %d\n", processorsAvailable);
			printf("Total Jobs Scheduled -- %d\n", kernelMax);
			printf("	GPU Scheduled Jobs    -- %d\n", GLOBAL_GPU_JOBS);
			printf("	Jobs Sent Back To CPU -- %d\n", GLOBAL_CPU_JOBS);
		}

		if (RTGS_PrintScheduleSummary(1, kernelMax, jobAttributesList)) {
			printf("\nSummary Failed\n");
		}

		if ((kernelMax != (GLOBAL_GPU_JOBS + GLOBAL_CPU_JOBS)) || processorsAvailable != GLOBAL_MAX_PROCESSORS) {
			return RTGS_FAILURE;
		}

		for (int j = 0; j <= kernelMax; j++) {
			jobAttributesList[j].processor_req = jobAttributesList[j].deadline = jobAttributesList[j].execution_time = jobAttributesList[j].latest_schedulable_time = 0;
		}
		kernelMax = 0; maxReleases = 0; jobNumber = 0; GLOBAL_GPU_JOBS = 0; GLOBAL_CPU_JOBS = 0;
	}

	if (processorsAllocatedList) {
		printf("\nERROR -- processorsAllocatedList Failed\n");
		return RTGS_FAILURE;
	}

	return RTGS_SUCCESS;
}