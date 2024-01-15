from torch import nn

from typing import Optional, Callable

class Mlp(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features:Optional[int]=None,
                 out_features:Optional[int] = None,
                 act_layer:Optional[type[nn.Module]] = nn.GELU,
                 norm_layer:Optional[type[nn.Module]] = None,
                 bias:bool=True,
                 drop:float=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(self.norm(x)))
        return x
