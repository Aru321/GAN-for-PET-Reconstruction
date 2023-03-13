import torch.nn as nn
import torch.nn.functional as F
from .IntmdSequential import IntermediateSequential
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
#
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
#
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
#for encoder blk
class SelfAttention2(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super(SelfAttention2,self).__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C,H,W,D = x.shape
        #
        x=rearrange(x,' B C H W D -> B (H W D) C',H=H,W=W,D=D)
        N=H*W*D
        #
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #x=rearrange(x,'B (H W D) C ->B C H W D',H=H,W=W,D=D)
        #print('sa:',x.shape)
        return x
#
class CrossAttention(nn.Module):
    def __init__(
            self, dim, image_size=16, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = depthwise_separable_conv(dim, dim * 3)
        self.qkv_mri = depthwise_separable_conv(dim, dim * 3)  # mri modal
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x, mri):
        # print(x.shape)
        B, C, H, W, D = x.shape
        N = H * W * D
        qkv = (
            self.qkv(x)
                .reshape(B, 3, self.num_heads, C // self.num_heads, N)
                .permute(1, 0, 2, 4, 3)
        )
        qkv_mri = (
            self.qkv_mri(mri)
                .reshape(B, 3, self.num_heads, C // self.num_heads, N)
                .permute(1, 0, 2, 4, 3)
        )
        '''
        qkv_mri=self.qkv_mri(mri)
        qkv_mri=rearrange(qkv_mri,'B C H W D -> B (H W D) C')
        qkv_mri=rearrange(qkv_mri,'B N C ->  t B h N d',t=3,h=self.num_heads,d=C//self.num_heads)
        '''
        q, k1, v1 = (qkv[0], qkv[1], qkv[2])  # make torchscript happy (cannot use tensor as tuple)
        # print(q.shape)
        # print(q.shape)
        k, v = (qkv_mri[1],
                qkv_mri[2],)
        # print(k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print('ca:',x.shape)
        # reshape
        # x=rearrange(x,'B (H W D) C ->B C H W D',H=H,W=W,D=D)
        #
        return x
#
class CrossAttention2(nn.Module):
    def __init__(
        self, dim, image_size=16,heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = depthwise_separable_conv(dim, dim*3)
        self.qkv_mri = depthwise_separable_conv(dim, dim*3)#mri modal
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x,mri):
        #print(x.shape)
        B, C,H,W,D = x.shape
        N=H*W*D
        qkv = (
            self.qkv(x)
            .reshape(B, 3, self.num_heads, C // self.num_heads, N)
            .permute(1, 0,2, 4,3 )
        )
        qkv_mri = (
            self.qkv_mri(mri)
            .reshape(B, 3, self.num_heads, C // self.num_heads, N)
            .permute(1, 0,2, 4,3 )
        )
        '''
        qkv_mri=self.qkv_mri(mri)
        qkv_mri=rearrange(qkv_mri,'B C H W D -> B (H W D) C')
        qkv_mri=rearrange(qkv_mri,'B N C ->  t B h N d',t=3,h=self.num_heads,d=C//self.num_heads)
        '''
        q ,k1,v1= (qkv[0],qkv[1],qkv[2]) # make torchscript happy (cannot use tensor as tuple)
        #print(q.shape)
        #print(q.shape)
        k,v=(qkv_mri[1],
             qkv_mri[2],)
        #print(k.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print('ca:',x.shape)
        #reshape
        #x=rearrange(x,'B (H W D) C ->B C H W D',H=H,W=W,D=D)
        #
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
#
class CrossResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x,y):
        return self.fn(x) + x,y

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))
#
class CrossPreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x,y):
        return self.dropout(self.fn(self.norm(x))),self.dropout(self.fn(self.norm(y)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
class CrossTransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        image_size,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        self.ca=CrossAttention(dim,heads=heads,dropout_rate=attn_dropout_rate)
        self.image_size=image_size
        self.embedding_dim=dim
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)
    def reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.image_size),
            int(self.image_size),
            int(self.image_size),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    def forward(self, x,y):
        res=self.ca(x,y)
        res,_=self.net(res)
        res=self.reshape_output(res)
        return res

class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)
if __name__ == '__main__':
    x=torch.randn(1,128,8,8,8)
    y=torch.randn(1,128,8,8,8)
    model=CrossAttention(dim=128,image_size=8,heads=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    res=model(x,y)
    print(res.shape)
