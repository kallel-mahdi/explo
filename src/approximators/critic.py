import torch
import torch.nn as nn
import torch.nn.functional as F



class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)
        self._ln1 = nn.LayerNorm(n_input)
        self._ln2 = nn.LayerNorm(n_features)
        self._ln3 = nn.LayerNorm(n_features)
        self._ln4 = nn.LayerNorm(n_features)
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)


        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):

        state_action = torch.cat((state.float(), action.float()), dim=1)
        state_action = self._ln1(state_action)

        features1 = F.relu(self._h1(state_action))
        features1 = self.dropout(features1)
        features1 = self._ln2(features1)
        
        features2 = F.relu(self._h2(features1))
        features2 = self.dropout(features2)
        features2 = self._ln3(features2)

        q = self._h3(features2)

        return torch.squeeze(q)
