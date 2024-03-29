import torch
import torch.nn as nn
from typing import Optional


class PatchEmbed(nn.Module):
    '''
    reference: timm.layers.patch_embed
    '''
    def __init__(self,
                 embed_dim:int,
                 img_size:int,
                 patch_size: int,
                 in_channels: int = 3,
                 norm_layer:Optional[nn.Module] = None,
                 bias:bool = True,
                 ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels = in_channels,
                              out_channels = embed_dim,
                              kernel_size= patch_size,
                              stride= patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape
        if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
        
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x



   

