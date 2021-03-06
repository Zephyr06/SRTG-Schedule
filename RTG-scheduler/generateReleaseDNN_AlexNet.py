import torch
import torch.nn as nn
from torchvision.models import alexnet
import copy

max_processor = 16
cost_scale = 5e-10

def write_jobs(release_file, path = 'myJob1.txt'):
    file=open(path,'w+')
    if(file.closed):
        print("File not exist!!!")
    for rel in release_file:
        file.write(str(int(rel.job_id))+','+str(int(rel.processor_req))+','+\
            "{:.0f}".format(rel.execution_time)+','+str(int(rel.deadline))+','+\
                   str(int(rel.latest_schedulable_time))+','+ str(int(rel.dependency))+ \
                   ','+ str(int(rel.priority))+'\n')
    file.close()

def write_release(release_file, path = 'myJob1_release.txt'):
    file=open(path,'w+')
    if(file.closed):
        print("File not exist!!!")
    for rel in release_file:
        file.write(str(int(rel.job_id))+','+str(int(rel.release_time))+'\n')
    file.close()

class Job_attributes:
    release_time = 0
    processor_req = 0
    execution_time = 0
    deadline = 0
    latest_schedulable_time = 0
    dependency = -1
    priority = 0

    def __init__(self, a,b,c,d, e):
        self.release_time=a
        self.processor_req=b
        self.execution_time=c
        self.deadline=d
        self.latest_schedulable_time=e
        self.job_id = 0
        self.dependency = -1
        self.priority = 0

def format(release_file):
    '''
    transform values to int
    '''
    remove_list=[]
    for i in range(len(release_file)):
        release_file[i].execution_time =  int(round(release_file[i].execution_time))
        if(release_file[i].execution_time ==0 ):
            remove_list.append(i)
    remove_list.reverse()
    for i in remove_list:
        release_file.remove(release_file[i])
    for i in range(len(release_file)):
        release_file[i].job_id = i
        release_file[i].dependency = i-1
    return release_file

def repeat(release_file, times=4):
    '''
    generate more tasks instances using the same NN
    '''
    big_release= copy.deepcopy(release_file)
    index=len(big_release)
    release_time = big_release[index-1].release_time
    for i in range(times-1):
        a=12
        for rel in release_file:
            rel2 = copy.deepcopy(rel)
            rel2.job_id = index
            rel2.release_time = release_time
            if(rel2.dependency!= -1):
                rel2.dependency = index -1
            index += 1
            release_time += 1
            big_release.append(rel2)
    return big_release

class MyAlexNet(nn.Module):
    def __init__(self):
        '''
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
        weights and biases of a layer to not require gradients.

        Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

        Note: Remove the last linear layer in Alexnet and add your own layer to
        perform 15 class classification.

        Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
        '''
        super().__init__()

        # self.cnn_layers = nn.Sequential()
        # self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        alexnet_model = alexnet(pretrained=False)

        self.cnn_layers = alexnet_model.features
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
        #
        # a=1


    def generate_layers_release(self, pict_size = 256):
        total_layers=len(self.cnn_layers)
        release_file=[]

        curr_pict_size=pict_size

        # convolution layer
        cnn_seq=[0,3,6,8,10]
        index = 0
        priority_index = 0
        for i in cnn_seq:
            processor_req=max_processor/pow(2,index)
            index+=1
            #  convolution layer
            in_channels=self.cnn_layers[i].in_channels
            out_channels = self.cnn_layers[i].out_channels
            kernel_size = self.cnn_layers[i].kernel_size[0]
            stride = self.cnn_layers[i].stride[0]
            padding=self.cnn_layers[i].padding[0]

            new_curr_pict_size = (curr_pict_size+padding*2-kernel_size+1)//stride

            cost = out_channels *new_curr_pict_size*new_curr_pict_size * (kernel_size*kernel_size*in_channels)

            job_temp = Job_attributes(i+0, processor_req, cost * cost_scale, 100000, 100000 )
            # generate equal priority sequence would bring second-level baseline



            # experiments for smaller scheduling units
            flag=1 # division units, could be 1, 2, 3
            if(flag==1):
                # job_temp.priority = 0
                priority_index = priority_index +1
                # job_temp.priority = 10-priority_index
                job_temp.priority = 0
                release_file.append(job_temp)
            elif (flag ==2):
                # job_temp.priority = 0
                priority_index = priority_index +1
                job_temp.priority = priority_index
                job_temp2=copy.deepcopy(job_temp)
                priority_index = priority_index +1
                job_temp2.priority = priority_index
                job_temp2.execution_time=job_temp2.execution_time//2
                job_temp.execution_time = job_temp.execution_time - job_temp2.execution_time

                release_file.append(job_temp)
                release_file.append(job_temp2)
            elif (flag ==3):
                job_temp2=copy.deepcopy(job_temp)
                job_temp2.execution_time=job_temp2.execution_time//3
                job_temp3 = copy.deepcopy(job_temp2)
                job_temp.execution_time = job_temp.execution_time - job_temp2.execution_time*2

                priority_index = priority_index +1
                job_temp.priority = priority_index
                priority_index = priority_index +1
                job_temp2.priority = priority_index
                priority_index = priority_index +1
                job_temp3.priority = priority_index
                release_file.append(job_temp)
                release_file.append(job_temp2)
                release_file.append(job_temp3)


        release_file = format(release_file)
        release_file = repeat(release_file , 3)
        # !!!!! we made an implicit modification there
        write_jobs(release_file)
        write_release(release_file)

        return release_file

    def generate_single_task_release(self, pict_size = 256):
        layer_release = self.generate_layers_release( pict_size)
        
        job = layer_release[0]
        # merge layer release
        for rel in layer_release:
            job.release_time = min(job.release_time, rel.release_time)
            job.processor_req = max (job.processor_req, rel.processor_req)
            job.execution_time += rel.execution_time # multiple processor???
            job.deadline = max (job.deadline, rel.deadline)
            job.latest_schedulable_time = max (job.latest_schedulable_time, rel.latest_schedulable_time)
            
        job.execution_time = job.execution_time
        return job



    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        '''
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
        model_output = self.fc_layers((self.cnn_layers(x)).reshape(x.shape[0], -1))


        return model_output


if __name__ == '__main__':
    net= MyAlexNet()
    net.generate_layers_release(pict_size = 256)
    #write_release([net.generate_single_task_release()])
