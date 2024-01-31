import torch
import torch.nn as nn

from core.models.layers.mlp import Mlp
from core.models.layers.drop import DropPath

'''
reference : timm
'''

class Attention(nn.Module):
    def __init__(self,
                 dim:int,
                 num_heads:int,
                 qkv_bias:bool=False,
                 qk_norm:bool=False,
                 attn_drop:float=0.,
                 proj_drop:float=0.,
                 norm_layer:nn.Module=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim은 num_heads의 배수여야합니다.'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn() 

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x:torch.Tensor):
        *B, N, C = x.shape
        qkv = self.qkv(x).reshape(*B, N, 3, self.num_heads, self.head_dim).permute(-3, *[i for i in range(len(B))], -2, -4,-1)
        q, k, v = qkv.unbind(0)

        # qk norm
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(*B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self,
                 dim:int,
                 num_heads:int,
                 mlp_ratio:float=4.,
                 qkv_bias:bool=False,
                 qk_norm:bool=False,
                 proj_drop:float=0.,
                 attn_drop:float=0.,
                #  init_values=None, # for LayerScale
                 drop_path=0.,
                 act_layer:nn.Module=nn.GELU,
                 norm_layer:nn.Module=nn.LayerNorm,
                 mlp_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_norm=qk_norm,
                              attn_drop=attn_drop,
                              proj_drop=proj_drop,
                              norm_layer=norm_layer)
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer,
                             drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

        