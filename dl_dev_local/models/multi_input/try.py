from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class fcDnnMi_1(BasicModule):
    '''
    Fully Connected Deep Neural Network
    '''
    def __init__(self,input_dim_1=482,input_dim_2=300,drop_rate=0.5):
        super(fcDnnMi_1,self).__init__()
        self.model_name = 'fcDnnMi_1'
        self.fc_dnn_layer = nn.Sequential(nn.Linear(input_dim_1+input_dim_2, 300),nn.ReLU(),nn.Dropout(drop_rate),
                                    nn.Linear(300,150),nn.ReLU(),nn.Dropout(drop_rate),
                                    nn.Linear(150,50),nn.ReLU(),nn.Dropout(drop_rate))

        self.end_layer = nn.Linear(50,1,bias=False)

    def forward(self,x):
        x1,x2 = x
        x = torch.cat([x1[:,-1,:],x2[:,-1,:]],dim=-1)
        x_signal = self.fc_dnn_layer(x)
        x = self.end_layer(x_signal)
        self.model_out['y_pred'] = x
        self.model_out['signals'] = x_signal
        return self.model_out
