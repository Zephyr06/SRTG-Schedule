#include <CppUnitLite/TestHarness.h>

#include "RTGS.h"
#include "RTGS_Global.h"
#include "string.h"

#include "../source/RTGS-file_handler.c"

using namespace std;
char* string2char(string str)
{
    char* ptr=NULL;
    ptr= (char*)malloc(sizeof(char)*str.length()+1) ;
    strcpy(ptr, str.c_str());
    return ptr;
}

TEST(realerBrush, drawMark)
{

	jobAttributes jobAttributesList[MAX_JOBS] = {{0}}; // array, processor_req, execution_time, deadline, latest_schedulable_time
	jobReleaseInfo releaseTimeInfo[MAX_JOBS] = {{0}}; // array , release_time and num_job_released
	scheduledResourceNode *processorsAllocatedList = NULL;

	// global variables initialized
	GLOBAL_GPU_JOBS = 0;
	GLOBAL_CPU_JOBS = 0;

	int processorsAvailable = GLOBAL_MAX_PROCESSORS;
	int jobNumber = 0;

    string jobsListFileName = "../testData/set1-jobs.txt";
    string releaseTimeFilename = "../testData/set1-jobReleaseTimes.txt";

    char* f1= string2char(jobsListFileName);
    char* f2= string2char(releaseTimeFilename);


	int kernelMax = get_job_information(jobAttributesList, string2char(jobsListFileName));
	if (kernelMax <= RTGS_FAILURE) { printf("ERROR - get_job_information failed with %d code\n", kernelMax); return; }
	int maxReleases = get_job_release_times(releaseTimeInfo, string2char(releaseTimeFilename));
	if (maxReleases <= RTGS_FAILURE) { printf("ERROR - get_job_release_times failed with %d code\n", maxReleases); return; }

	if (GLOBAL_RTGS_DEBUG_MSG > 1) {
		printf("\n**************** The GPU Scheduler will Schedule %d Jobs ****************\n", kernelMax);
	}

    free(f1);
    free(f2);
}

int main()
{
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
