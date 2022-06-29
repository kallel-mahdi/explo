import logging
import logging.config
from copy import deepcopy
from itertools import chain
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.nn import Identity, Linear, Sequential
import torch.nn as nn

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)


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
    
            
class MLPSequential(torch.nn.Module):
    
    def __init__(
                self,
                Ls: List[int],
                nonlinearity: Optional[Callable] = torch.nn.ReLU(),
                ):
        super().__init__()
        ops = []
        for in_size, out_size in zip(Ls[:-1], Ls[1:]):
            ops.append(torch.nn.Linear(in_size, out_size))
            ops.append(nonlinearity)
        self.f = torch.nn.Sequential(*ops[:-1])

    def forward(self, s):
        return self.f(s)

    def get_weights(self):
        weights = []
        for p in self.f.parameters():
            if p.requires_grad:
                weights.append(p.detach().clone().flatten())
        return torch.cat(weights)

    def set_weights(self, weights):
        idx = 0
        for p in self.f.parameters():
            if p.requires_grad:
                nb_par = 1
                for s in p.shape:
                    nb_par *= s
                p.data = weights[idx:idx + nb_par].view(p.shape)
                idx += nb_par


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a

if __name__ == '__main__':
    
    mlp = MLP([4,2],add_bias=True)
    params = torch.rand(10,mlp.len_params)
    states = torch.rand(1000,4)
    mlp(params,states).size()
