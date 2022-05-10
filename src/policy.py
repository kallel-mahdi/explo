from typing import Tuple, List, Callable, Union, Optional

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
        L0: int,
        *Ls: List[int],
        add_bias: bool = False,
        nonlinearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Inits MLP."""
        self.L0 = L0
        self.Ls = Ls
        self.add_bias = add_bias
        self.len_params = sum(
            [
                (in_size + 1 * add_bias) * out_size
                for in_size, out_size in zip((L0,) + Ls[:-1], Ls)
            ]
        )

        if nonlinearity is None:
            nonlinearity = lambda x: x
        self.nonlinearity = nonlinearity

    def __call__(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Maps states and parameters of MLP to its actions.

        Args:
            state: The state tensor.
            params: Parameters of the MLP.

        Returns:
            Output of the MLP/actions.
        """
        with torch.no_grad():
            params = params.view(self.len_params)
            out = state
            start, end = (0, 0)
            in_size = self.L0
            for out_size in self.Ls:
                # Linear mapping.
                start, end = end, end + in_size * out_size
                out = out @ params[start:end].view(in_size, out_size)
                # Add bias.
                if self.add_bias:
                    start, end = end, end + out_size
                    out = out + params[start:end]
                # Apply nonlinearity.
                out = self.nonlinearity(out)
                in_size = out_size
        return out



if __name__ == "__main__":
    
    mlp = MLP(*[8,2])
    mlp.len_params