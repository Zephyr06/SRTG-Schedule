# Scheduling simulation for Deep Learning tasks on GPU devices

## Real-Time GPU Scheduler

A large part of the code developed in this project is based on RTG-Scheduler, which can be found [there](https://github.com/kiritigowda/SRTG-Schedule). Things we adopt mainly include reading task files, saving and visulaizing the scheduling results. The scheulder is mostly written by me.

## SRTG-JobCreator

DNN tasks are created by python script `generateReleaseDNN_AlexNet.py`. We use popular DNN framework pytorch to specify network structure there.

After creating DNN tasks, you should specify the generated file path to the scheduler, which include both a release file and a task file.

## RTG-Scheduler Usage

### Dependency

```
- Python3 and pytorch
- CMake
- C++
```

### Linux
```
cd build/bin
./schedulerTest [options] --j <jobs_file.csv>
                          --r <Release_Time_file.csv>
                          --m <option> 
                          --p <option> 
```

### Scheduler Options Supported
````
        --h/--help      -- Show full help
        --v/--verbose   -- Show detailed messages
````

### Scheduler Parameters
````
        --j/--jobs                 -- Jobs to be scheduled [required]
        --r/--releaseTimes         -- Release times for the jobs [required]
        --m/--mode                 -- Scheduler Mode [optional - default:5]
        --p/--maxProcessors        -- Max processors available on the GPU [optional - default:16]
        --d/--delayLimitPercentage -- Delay Schedule processor limit in percentage [optional - default:60]
        --s/--simulation 	   -- simulation mode turn ON/OFF [optional - default:ON]
        --h/--hardware 	           -- Jobs Scheduled on hardware <AMD/NVIDIA> - [optional - default:OFF]
````
#### Job Release file explanation
The Release Time File has the list of release times of the kernels, which include `Job_ID` and `release time` respectively in each line.

#### Job task file explanation
Each line represents a single task, the task parameters are `Job_ID`, `processor requirement`, `execution time`, `deadline`, `latest schedule time`, `dependency task's job ID`, `priority`.

#### Visualized results
The scheduling results visualization can be found under the folder `build\bin\RTGS-Summary`.
