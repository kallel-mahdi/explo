from torch.nn import Linear,Sequential,Identity
from typing import Tuple, List, Callable, Union, Optional
from copy import deepcopy
from itertools import chain
import torch

class MLP:
    """Multilayer perceptrone.

    Consists of at least two layers of nodes: an input layer and an output
    layer. Optionally one can extend it with arbitrary many hidden layers.
    Except for the input nodes, each node is a neuron that can optionally use a
    nonlinear activation function.

    Attributes:
        L0: Number of input nodes. For a gym environment objective this
            corresponds to the states.
        Ls: List of numbers for nodes of optional hidden layers and the output
            layer. For a gym environment objective the last number of the list
            has to correspond to the actions.
        add_bias: If True every layer has one bias vector of the same dimension
            as the output dimension of the layer.
        nonlinearity: Opportunity to hand over a nonlinearity function.
    """

    def __init__(
                self,
                Ls: List[int],
                add_bias: bool = False,
                nonlinearity: Optional[Callable] = None,
                ):
        
        """Inits MLP."""

        self.Ls = Ls
        self.add_bias = add_bias
        self.weight_sizes  = [(in_size,out_size)
                                for in_size, out_size in zip(Ls[:-1], Ls[1:])]
        if self.add_bias :
            
            self.bias_sizes = [(out_size)
                                for  out_size in Ls[1:]  ]
        
        self.len_params = sum(
            [
                (in_size + 1 * add_bias) * out_size
                for in_size, out_size in zip(Ls[:-1], Ls[1:])
            ]
        )
        
        if nonlinearity is None: 
            self.nonlinearity = Identity
    
    def reset_weights(self,net,params):
        
        start,end = (0,0)
        
        for layer,(in_size,out_size) in zip(net,self.weight_sizes):
            
            start = end
            end   = start  + (in_size * out_size)
            end   = start + in_size * out_size        
            
            weight_params = params[start:end].reshape(out_size,in_size)
            layer.weight.data = weight_params
            
            if self.add_bias : 
                
                bias_params = params[end: end+ out_size].reshape(out_size)
                end = end + out_size
                layer.bias.data = bias_params
        
        return net 
    
    def build_net(self):
        
        ### initialize deep layers
        layer_list = [ [Linear(in_size,out_size,bias=self.add_bias),self.nonlinearity]
                        for (in_size,out_size) in self.weight_sizes[:-1]]
        
        ### last layer has no nonlinearity
        layer_list.append([Linear(*self.weight_sizes[-1],bias=self.add_bias)])
        
        ### initialize model
        net = Sequential(*chain(*layer_list))
        self.net = deepcopy(net)##tmp
        
        return net

        
    def __call__(self,states,params):
        
        
        ### we initialize network at each call (maybe reset network in future)
        net = self.build_net()
        #############
        net = self.reset_weights(net,params)
        self.updated_net = deepcopy(net)##tmp
        rslt = net(states)
        
        return rslt
    
    
    
if __name__ == "__main__":
    
    mlp = MLP([8,2],add_bias=True)
    params = torch.rand(mlp.len_params)
    states = torch.rand(10,5,8)
    mlp(states,params).size()

        
