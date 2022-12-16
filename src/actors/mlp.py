import logging
from os import path
# import logging.config
# log_file_path = path.join("/home/q123/Desktop/explo/logging.conf")
# logging.config.fileConfig(log_file_path)
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
                input_shape,output_shape,
                Ls: List[int],
                add_bias: bool,
                nonlinearity: Optional[Callable] = None,
                **kwargs
                ):
        
        """Inits MLP with the provided weights 
        Note the MLP can support batches of weights """
        
        super(MLP, self).__init__()
        
        # if input_shape is not None:
        #     Ls = [input_shape,output_shape]
            
        
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
            nonlinearity = torch.nn.Tanh()
            #nonlinearity = torch.nn.functional.tanh()
            #nonlinearity = torch.nn.Identity()
        
        
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
        
        
    def forward(self,states,params=None):
        
        #logger.debug(f'params {params.shape} states {states.shape}')
        
        if params is None :
            
            params = self.default_weights.data    
            
        weights,biases = self.create_weights(params)
        outputs = states.T
        
        for i,(w,b) in enumerate(
                            zip(weights,biases)
                            ):
            
            w_tmp = w @ outputs
            b_tmp = b.unsqueeze(-1).expand_as(w_tmp)
            outputs =  w_tmp + b_tmp
        
        ## no nonlinearity for last layer
            # if (i+1) < (self.n_layers) :
            #     outputs = self.nonlinearity(outputs)

            outputs = self.nonlinearity(outputs) ###remove
                
        logger.debug(f'MLP : actions {outputs.shape}')
        return outputs
    
    def predict(self,states,*args,**kwargs):
        
        return self.forward(self.default_weights,states)