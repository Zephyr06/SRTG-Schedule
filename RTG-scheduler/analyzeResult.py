import numpy as np

def readRes(mode=1):
    path="build/bin/RTGS-Summary/"+"RTGS-Mode-"+str(mode)+"-Job-Summary.csv"
    file=open(path,'r')
    lines=file.readlines()[1:]
    N=len(lines)
    res=np.zeros((N,3)) # completion-release, deadline-release
    i=0
    for line in lines:
        chunks=line.split(',')
        job=int(chunks[0])
        deadline=int(chunks[3])
        release=int(chunks[4])
        complete=int(chunks[8])
        

        res[i][0]=job
        if(complete==-1):
            res[i][1]=-1
        else:    
            res[i][1]=complete-release
        res[i][2]=deadline-release
        i=i+1

    return res

if __name__ == "__main__":
    res=readRes()
    print("The worst case response time among scheduled tasks is: ", np.amax(res[:,1]))
