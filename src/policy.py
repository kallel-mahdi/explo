from torch.nn import Linear,Sequential,Identity
from typing import Tuple, List, Callable, Union, Optional
from copy import deepcopy
from itertools import chain
import torch

class MyMLP(torch.nn.Module):

    def __init__(
                self,
                Ls: List[int],
                params,
                add_bias: bool = False,
                nonlinearity: Optional[Callable] = None,
                ):
        
        """Inits MLP with the provided weights 
        Note the MLP can support batches of weights """
        
        super(MyMLP, self).__init__()
        
        print(f'MyMLP received params with shape',params.shape)
        self.params = params
        self.weight_sizes  = [(in_size,out_size)
                                for in_size, out_size in zip(Ls[:-1], Ls[1:])]
        self.len_params = sum(
            [
                (in_size + 1 * add_bias) * out_size
                for in_size, out_size in zip(Ls[:-1], Ls[1:])
            ]
        )
    
    def create_weights(self,params):
        
        weights = []
        start,end = (0,0)
        
        for (in_size,out_size) in self.weight_sizes:
            
            start = end
            end   = start  + (in_size * out_size)
            end   = start + in_size * out_size        
            
            
            #print("MyMLP params size",params[...,start:end].shape)
            weight = params[...,start:end].reshape(*params.shape[:-1],out_size,in_size)
            print("MyMLP weight size",weight.shape)
            #weight = params[...,start:end].reshape(out_size,in_size)
            weights.append(weight) ## add transpose or dim error
        
        return weights
        
        
    def forward(self,states):

        weights = self.create_weights(self.params)
        output = states
        
        for w in weights:
            print(f'forward: states.shape {output.shape} params_batch.shape {w.shape}')
            #output = output @ w
            output = w@ output.T
        
        print(f'forward: output.shape {output.shape}')
        return output

class MyMLP2(torch.nn.Module):
    
    def __init__(
                self,
                Ls: List[int],
                add_bias: bool = False,
                nonlinearity: Optional[Callable] = None,
                ):
        
        """Inits MLP with the provided weights 
        Note the MLP can support batches of weights """
        
        super(MyMLP2, self).__init__()
        
        self.weight_sizes  = [(in_size,out_size)
                                for in_size, out_size in zip(Ls[:-1], Ls[1:])]
        self.len_params = sum(
            [
                (in_size + 1 * add_bias) * out_size
                for in_size, out_size in zip(Ls[:-1], Ls[1:])
            ]
        )
    
    def create_weights(self,params):
        
        weights = []
        start,end = (0,0)
        
        for (in_size,out_size) in self.weight_sizes:
            
            start = end
            end   = start  + (in_size * out_size)
            end   = start + in_size * out_size        
            
            weight = params[...,start:end].reshape(*params.shape[:-1],out_size,in_size)
            weights.append(weight.T) ## add transpose or dim error
            
        return weights
        
        
    def forward(self,params,states):

        weights = self.create_weights(params)
        
        output = states
        for w in weights:
            output = output @ w
        
        return output


