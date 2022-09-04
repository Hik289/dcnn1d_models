import torch
import time
import torch.nn as nn

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) # 默认名字
        self.alpha_idx = None
        self.model_out = {'y_pred':None,'signals':None, 'multilayers':None}  ## 基本输出
        
    def load(self,path):
        self.load_state_dict(torch.load(path,map_location='cuda:0'))
        # self.load_state_dict(torch.load(path))
        

    def save(self,path,epoch=None,t=None,env_name = None, IC = None,):
        prefix = path + self.model_name + '-' + env_name + '_'
        name = prefix + t +  '_' + str(epoch) + '_' + str(round(IC,5)) +  '.pth'
        # torch.save(self.state_dict(),name) # 只保存神经网络的模型参数 
        torch.save(self,name) # 保留模型对象
        return name
