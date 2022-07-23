# import logging
# import logging.config

import logging
from os import path

log_file_path = path.join("/home/q123/Desktop/explo/logging.conf")
logging.config.fileConfig(log_file_path)
logger = logging.getLogger("ShapeLog."+__name__)

from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Identity, Linear, Sequential


class MLP(torch.nn.Module):
    
    """Implements an MLP that is differentiable w.r.t to parameters (when they are fed by BO)
    """

    def __init__(
                self,
                Ls: List[int],
                add_bias: bool = False,
                nonlinearity: Optional[Callable] = None,
                ):
        
        """Inits MLP with the provided weights 
        Note the MLP can support batches of weights """
        
        super(MLP, self).__init__()
            
        weight_sizes  = [(in_size,out_size)
                                for in_size, out_size in zip(Ls[:-1], Ls[1:])]
        n_layers = len(weight_sizes)
        
        len_params = sum(
            [
                (in_size + 1 * add_bias) * out_size
                for in_size, out_size in zip(Ls[:-1], Ls[1:])
            ]
        )
        
        n_actions = Ls[-1]
        
        if nonlinearity is None:
            #nonlinearity = torch.nn.ReLU()
            nonlinearity = torch.nn.Identity()
        
        
        self.__dict__.update(locals())
        
        self.register_parameter("default_weights",nn.Parameter(torch.zeros((1,self.len_params)))) ## maybe add device later
        
    def create_weights(self,params):
        
        weights,biases = [],[]
                
        start,end = (0,0)
        
        for (in_size,out_size) in self.weight_sizes:
            
            start = deepcopy(end)
            end   = deepcopy(start)  + (in_size * out_size)
            weight = params[...,start:end].reshape(*params.shape[:-1],out_size,in_size)
            
            if self.add_bias:
                
                bias = params[...,end:end+out_size].reshape(*params.shape[:-1],out_size)
                end = deepcopy(end) + out_size
            
            else :
                
                bias = torch.zeros(*params.shape[:-1],out_size)
                
            weights.append(weight) ## add transpose or dim error
            biases.append(bias)
                
            #print(f'weight.shape {weight.shape} bias.shape {bias.shape}')
        
        return weights,biases
        
        
    def forward(self,params,states):
        
        #logger.debug(f'params {params.shape} states {states.shape}')
        
        weights,biases = self.create_weights(params)
        outputs = states.T
        
        for i,(w,b) in enumerate(
                            zip(weights,biases)
                            ):
            
            w_tmp = w @ outputs
            b_tmp = b.unsqueeze(-1).expand_as(w_tmp)
            outputs =  w_tmp + b_tmp
        
        ## no nonlinearity for last layer
            if (i+1) < (self.n_layers) :
                outputs = self.nonlinearity(outputs)
                
        logger.debug(f'MLP : actions {outputs.shape}')
        return outputs
    
    def predict(self,states,*args,**kwargs):
        
        return self.forward(self.default_weights,states)

class ActorNetwork(nn.Module):
    
    def __init__(self, 
                Ls: List[int],
                add_bias: bool = False, 
                nonlinearity = None,
                **kwargs,
                ):
        super(ActorNetwork, self).__init__()

        self.weight_sizes  = [(in_size,out_size)
                                for in_size, out_size in zip(Ls[:-1], Ls[1:])]
        
        self.n_layers = len(self.weight_sizes)
        
        self.len_params = sum(
            [
                (in_size + 1 * add_bias) * out_size
                for in_size, out_size in zip(Ls[:-1], Ls[1:])
            ]
        )
        
        self.n_actions = Ls[-1]
        
        self.nonlinearity = torch.nn.Identity()  if nonlinearity is None else nonlinearity
        
        self.add_bias = add_bias
        
        self.layers = self.build_layers()
           
    
    def build_layers(self):
        
        dct = OrderedDict(
                            ('layer'+str(i),nn.Linear(sizes[0],sizes[1],bias=self.add_bias))
                            for i,sizes in enumerate(self.weight_sizes)                           
                        )
        
        layers = nn.Sequential(dct)   
        
        return layers    

    
    @property
    def n_params(self):
        
        n = 0
        for p in self.parameters():
            p_shape = torch.tensor(p.shape)
            n += torch.prod(p_shape,0)
        
        return n
    
    @property
    def device(self):
        device = next(self.parameters()).device
        return device
    
    def get_params(self):
        
        vector_params = []
        
        for p in self.parameters():
            
            vector_params.append(p.data.flatten())
        
        return torch.cat(vector_params)
    
    def set_params(self,new_params):
        
        with torch.no_grad():
            
            idx = 0
            for param in self.parameters():
                    weights = param.data
                    weights_shape = torch.tensor(weights.shape)
                    n_steps = torch.prod(weights_shape,0)
                    new_param = new_params[idx:idx+n_steps].reshape(*weights_shape)
                    param.data = new_param
                    idx += n_steps
                
        
    def add_noise(self,noise):
        
        
        with torch.no_grad():
            
            if not torch.is_tensor(noise):        
                noise = torch.tensor(noise,device=self.device)
            
            self.set_params(self.get_params()+noise)
            
        
    def forward(self, states):
        
        outputs = states
        
        for i,layer in enumerate(self.layers):
            
            outputs = layer(outputs)
            
            if (i+1) < (self.n_layers) :
                
                outputs = self.nonlinearity(outputs)
            
        return outputs
    
    def super_forward(self,params,states):
        
        self.set_params(params)
        a = self(states)
        
        return a
    
    