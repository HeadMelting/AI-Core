# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim:int, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0) # [2, grid_size, grid_size]

    grid = grid.reshape([2, 1, grid_size, grid_size]) # [2, 1, grid_size, grid_size]
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # grid = [2, 1, grid_size, grid_size]
    assert embed_dim % 2 == 0, 'embed_dim 짝수여야한다네요'

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    # embed_dim: int (D)
    # pos: [1, 1, grid_size, grid_size]
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32) # [D/2]
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1) #[grid_size * grid_size]
    out = np.einsum('m,d->md', pos, omega) # outer product [grid_size^2, D/2]

    emb_sin = np.sin(out) #[grid_size^2, D/2]
    emb_cos = np.cos(out) #[grid_size^2, D/2]

    emb = np.concatenate([emb_sin, emb_cos], axis= 1)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    pos = np.arange(length, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
     
