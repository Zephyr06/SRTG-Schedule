import torch
import torch.nn as nn

max_processor = 16

def write_release(release_file, path = 'myJob1.txt'):
    file=open(path,'w+')
    if(file.closed):
        print("File not exist!!!")
    for rel in release_file:
        file.write(str(int(rel.release_time))+','+str(int(rel.processor_req))+','+\
            "{:.6f}".format(rel.execution_time)+','+str(int(rel.deadline))+','+str(int(rel.latest_schedulable_time))+'\n')
    file.close()


class Job_attributes:
    release_time = 0
    processor_req = 0
    execution_time = 0
    deadline = 0
    latest_schedulable_time = 0

    def __init__(self, a,b,c,d, e):
        self.release_time=a
        self.processor_req=b
        self.execution_time=c
        self.deadline=d
        self.latest_schedulable_time=e

class SimpleNet(nn.Module):
    '''Simple Network with atleast 2 conv2d layers and two linear layers.'''
    def __init__(self):
        '''
        Init function to define the layers and loss function
        '''
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=500, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=15)
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


    def generate_layers_release(self, pict_size = 256, scale = 1e-4):
        total_layers=len(self.cnn_layers)
        release_file=[]

        curr_pict_size=pict_size

        # convolution layer
        for i in range(int(len(self.cnn_layers)/3)):
            processor_req=max_processor/pow(2,i)
            #  convolution layer
            in_channels=self.cnn_layers[i*3].in_channels
            out_channels = self.cnn_layers[i*3].out_channels
            kernel_size = self.cnn_layers[i*3].kernel_size[0]
            stride = self.cnn_layers[i*3].stride[0]
            padding=self.cnn_layers[i*3].padding[0]

            new_curr_pict_size = (curr_pict_size+padding*2-kernel_size+1)//stride

            cost = out_channels *new_curr_pict_size*new_curr_pict_size * (kernel_size*kernel_size*in_channels)

            release_file.append(Job_attributes(i*3+0, processor_req, cost * scale, 100000, 100000 ))

            pict_size = new_curr_pict_size

            #  max pool layer
            kernel_size=self.cnn_layers[i*3+1].kernel_size
            stride = self.cnn_layers[i*3+1].stride
            cost = (pict_size-(kernel_size-1) )//stride * kernel_size* kernel_size
            pict_size = (pict_size-(kernel_size-1) )//stride
            release_file.append(Job_attributes(i*3+1, processor_req, cost * scale, 100000, 100000 ))

            #  relu layer
            cost = pict_size*pict_size*2
            release_file.append(Job_attributes(i*3+2, processor_req, cost * scale, 100000, 100000 ))
        
        curr_release_time = len(self.cnn_layers)
        # fully connected layer
        for i in range(int(len(self.fc_layers)/2)):
            processor_req=1
            #  convolution layer
            in_features=self.fc_layers[i*2].in_features
            out_features = self.fc_layers[i*2].out_features

            new_curr_pict_size = out_features

            cost = in_features*out_features

            release_file.append(Job_attributes(curr_release_time+i*2+0, processor_req, cost * scale, 100000, 100000 ))

            #  relu layer
            cost = out_features*2
            release_file.append(Job_attributes(curr_release_time+i*2+1, processor_req, cost * scale, 100000, 100000 ))
        

        # !!!!! we made an implicit modification there
        write_release(release_file)

        return release_file

    def generate_single_task_release(self, pict_size = 256, scale = 1e-4):
        layer_release = self.generate_layers_release( pict_size, scale)
        
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

        model_output = self.fc_layers((self.cnn_layers(x)).reshape(x.shape[0], -1))

        return model_output

if __name__ == '__main__':
    net= SimpleNet()
    net.generate_layers_release(256)
    write_release([net.generate_single_task_release()])
