from functools import reduce
from operator import mul

import numpy as np
import torch.nn as nn
import torch
from functools import lru_cache
import numpy as np
import torch
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C D H W
    """
    B, C, D, H, W = img.shape
    img_reshape = img.view(B, C, D, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 3, 5, 4, 6, 1).contiguous().reshape(-1, D * H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, D, H, W):
    """
    img_splits_hw: B' D H W C
    """

    B = int(img_splits_hw.shape[0] / (D * H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, D, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, D, H, W, -1)

    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        D = 32
        H = W = int(np.sqrt(new_HW // D))
        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x

    # Patch Embedding(4*4的patch，96的embeddim):5, 96, 32, 32, 32 # B, C, D, H, W
    # permute为: 5, 32, 32, 32, 96 # B, D, H, W, C
    # Window Partition(4*4patch的Window):

    # Shuffle mode
    # 5 (4 8) (4 8) (4 8) 96 # B (ws dd) (ws hh) () C
    # Not Shuffle mode
    # 5 (8 4) (8 4) (8 4) 96 # B (dd ws) (hh ws) (ww ws) C
    # 5 8 8 8 4 4 4 96  # B (dd hh ww) (ws ws ws) C

    # Window Reversion(4*4patch的Window):
    # Shuffle mode
    # 5 (4 8) (4 8) (4 8) 96 # B (ws dd) (ws hh) () C
    # Not Shuffle mode
    # 5 (8 4) (8 4) (8 4) 96 # B (dd ws) (hh ws) (ww ws) C

    # input_x = torch.arange(32).view(1, 4, 8)
    # print(input_x)
    # input_x1 = rearrange(input_x, 'b qkv (ws dd) -> qkv (b dd) ws', qkv=4, ws=2)
    # print(input_x1)
    # input_x2 = rearrange(input_x, 'b qkv (dd ws) -> qkv (b dd) ws', qkv=4, ws=2)
    # print(input_x2)


def shuffle_window_fusion(input_x, fuse_num):
    """
    Args:
        :param input_x: (B, D, H, W, C)
        :param fuse_num: (int)fuse_num
    Returns:
        windows: x (B, D, H, W, C)
    """
    B, D, H, W, C = input_x.shape
    total_num = fuse_num * fuse_num
    split = C // total_num  # 为8至少有64
    channel_fuse_x = torch.zeros_like(input_x)
    for i in range(total_num):
        j = i // fuse_num
        k = i % fuse_num
        channel_fuse_x[:, :, :, :, i * split:(i + 1) * split] = input_x[:, :, :, :, (fuse_num * k + j) * split:(fuse_num * k + j + 1) * split]
    return channel_fuse_x


def shuffle_window_partition(x, window_size, shuffle=False):
    """
    Args:
        :param x: (B, D, H, W, C)
        :param window_size (tuple[int]): window size
        :param shuffle: Doing Shuffle, If False, Regard window_num forward and window_size backward
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    # dd = D // window_size[0], ws1 = window_size[0]
    # hh = H // window_size[1], ws2 = window_size[1]
    # ww = W // window_size[2], ws3 = window_size[2]
    if not shuffle:
        x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
                   window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    else:
        x = x.view(B, window_size[0], D // window_size[0], window_size[1], H // window_size[1], window_size[2],
                   W // window_size[2], C)
        windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1, reduce(mul, window_size), C)
    # if not shuffle:  # C = (head dim) not shuffle是hh再前面
    #     windows = rearrange(x, 'b (dd ws1) (hh ws2) (ww ws3) C -> (b dd hh ww) (ws1 ws2 ws3) C',
    #                         C=C, ws1=window_size[0], ws2=window_size[1], ws3=window_size[2])
    # else:
    #     windows = rearrange(x, 'b (ws1 dd) (ws2 hh) (ws3 ww) C -> (b dd hh ww) (ws1 ws2 ws3) C',
    #                         C=C, ws1=window_size[0], ws2=window_size[1], ws3=window_size[2])
    return windows


def shuffle_window_reverse(windows, window_size, B, D, H, W, shuffle=False):
    """
    Args:
        :param windows: (B*num_windows, window_size, window_size, C)
        :param window_size: (tuple[int]) Window size
        :param B: (int) Batch size of image
        :param D: (int) Depth size of image
        :param H: (int) Height of image
        :param W: (int) Width of image
        :param shuffle: Doing Shuffle, If False, Regard window_num forward and window_size backward
    Returns:
        x: (B, D, H, W, C)
    """
    # dd = D // window_size[0], ws1 = window_size[0]
    # hh = H // window_size[1], ws2 = window_size[1]
    # ww = W // window_size[2], ws3 = window_size[2]
    if not shuffle:
        x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0],
                         window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    else:
        x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0],
                         window_size[1], window_size[2], -1)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, D, H, W, -1)
    # if not shuffle:
    #     x = rearrange(windows, '(b dd hh ww) ws1 ws2 ws3 C -> b (dd ws1) (hh ws2) (ww ws3) C',
    #                   b=B, dd=D // window_size[0], hh=H // window_size[1], ww=W // window_size[2],
    #                   ws1=window_size[0], ws2=window_size[1], ws3=window_size[2])
    # else:
    #     x = rearrange(windows, '(b dd hh ww) ws1 ws2 ws3 C -> b (ws1 dd) (ws2 hh) (ws3 ww) C',
    #                   b=B, dd=D // window_size[0], hh=H // window_size[1], ww=W // window_size[2],
    #                   ws1=window_size[0], ws2=window_size[1], ws3=window_size[2])
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device, shuffle):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = shuffle_window_partition(img_mask, window_size, shuffle)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
