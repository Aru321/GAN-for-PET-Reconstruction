import torch
from torch import nn
import torch.nn.functional as F
from .PositionalEncoding1 import LearnedPositionalEncoding,FixedPositionalEncoding
from .Transformer1 import CrossAttention
from .Transformer1 import TransformerModel,FeedForward
from einops import rearrange
class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels//8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels//8, in_channels//8, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Backbone(nn.Module):
    def __init__(self, n_channels=1, n_filters=8, normalization='batchnorm', has_dropout=False):
        super(Backbone, self).__init__()
        self.has_dropout = has_dropout
        #LPET编码
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization='none')
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization='none')

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        #T1编码
        self.Tblock_one = ConvBlock(1, n_channels, n_filters, normalization='none')
        self.Tblock_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization='none')

        self.Tblock_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.Tblock_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.Tblock_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.Tblock_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.Tblock_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)

        self.fusion1 = Fusion_Network(n_filters=1)
        self.fusion2 = Fusion_Network(n_filters=2)
        self.fusion3 = Fusion_Network(n_filters=3)
        self.fusion4 = Fusion_Network(n_filters=4)
        #self.fusion3 = Fusion_Network2(out_channels=n_filters*4)
        #self.fusion4 = Fusion_Network2(out_channels=n_filters*8)

        #self.__init_weight()

    #input1,LPET input2,T1
    def forward(self, input1, input2):
        Lpet_x1 = self.block_one(input1)
        T_x1 = self.Tblock_one(input2)

        # feature fusion
        FLpet_x1 = self.fusion1(Lpet_x1, T_x1)

        Lpet_x1_dw = self.block_one_dw(FLpet_x1)
        T_x1_dw = self.Tblock_one_dw(T_x1)

        Lpet_x2 = self.block_two(Lpet_x1_dw)
        T_x2 = self.Tblock_two(T_x1_dw)

        #feature fusion
        FLpet_x2 = self.fusion2(Lpet_x2, T_x2)

        Lpet_x2_dw = self.block_two_dw(FLpet_x2)
        T_x2_dw = self.Tblock_two_dw(T_x2)


        Lpet_x3 = self.block_three(Lpet_x2_dw)
        T_x3 = self.Tblock_three(T_x2_dw)

        # feature fusion
        FLpet_x3 = self.fusion3(Lpet_x3, T_x3)

        Lpet_x3_dw = self.block_three_dw(FLpet_x3)
        T_x3_dw = self.Tblock_three_dw(T_x3)

        Lpet_x4 = self.block_four(Lpet_x3_dw)
        T_x4 = self.Tblock_four(T_x3_dw)

        FLpet_x4 = self.fusion4(Lpet_x4,T_x4)

        # if self.has_dropout:
        #     x4 = self.dropout(x4)

        res = [Lpet_x1, Lpet_x2, Lpet_x3]

        return res, FLpet_x4
class Fusion_Network2(nn.Module):
    def __init__(self,out_channels):
        super(Fusion_Network2, self).__init__()
        self.CrossAttn = CrossAttention(dim=out_channels, heads=4)
        # ffn
        self.ffn = FeedForward(out_channels, out_channels, dropout_rate=0.1)
        # other
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout3d(p=0.1)
    def forward(self,x,mri):
        x2 = rearrange(x, 'B C H W D ->B (H W D) C', H=x.shape[2], W=x.shape[3], D=x.shape[4])
        z = x2 + self.norm(self.CrossAttn(x, mri))
        #
        z = z + self.dropout(self.norm(self.ffn(z)))
        z = rearrange(z, 'B (H W D) C ->B C H W D', H=x.shape[2], W=x.shape[3], D=x.shape[4])
        return z
class Fusion_Network(nn.Module):
    def __init__(self, n_filters=8):
        super(Fusion_Network, self).__init__()
        # self.conv1 = nn.Conv3d(n_filters * 8, n_filters * 8, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv3d(n_filters * 8, n_filters * 8, kernel_size=1, padding=0)
        #self.conv3 = nn.Conv3d(n_filters * 16, n_filters * 8, kernel_size=1, padding=0)
        self.conv3d1 = nn.Sequential(
            nn.Conv3d(n_filters * 8, n_filters * 4, kernel_size=1, padding=0),
            nn.BatchNorm3d(n_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_filters * 4, 1, kernel_size=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.conv3d2 = nn.Sequential(
            nn.Conv3d(n_filters * 8, n_filters * 4, kernel_size=1, padding=0),
            nn.BatchNorm3d(n_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_filters * 4, n_filters * 8, kernel_size=1, padding=0),
            nn.BatchNorm3d(n_filters * 8),
            nn.Sigmoid()
        )
        self.average_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, l, t):
        #x1:feature of LPET  x2: feature of T1
        # l = self.conv1(l)
        # t = self.conv2(t)#torch.Size([12, 128, 8, 8, 8])
        fusion_feature = l + t
        #fusion_feature = torch.cat((l,t),1)
        # local fusion 空间注意力 12*1*8*8*8
        # x1 = self.conv3d1(fusion_feature)
        #
        # #global fusion 通道注意力  12*128*1*1*1
        # x2 = self.average_pool(fusion_feature)
        # x2 = self.conv3d2(x2)
        #
        # #broadcasting addition
        # x1 = fusion_feature * x1
        # x2 = fusion_feature * x2
        #
        # output_feature = x1 + x2
        #output_feature = self.conv3(output_feature)

        # x1 = self.sigmoid(x)
        # x2 = self.sigmoid(1-x)
        ##print(x1.shape,x2.shape) torch.Size([12, 128, 8, 8, 8]) torch.Size([12, 128, 8, 8, 8])

        #output_feature = x1 * l + x2 * t

        return fusion_feature

class Decoder(nn.Module):
    def __init__(self, n_channels=1, n_filters=8, normalization='batchnorm', has_dropout=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_six = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_seven = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.last_conv = nn.Sequential(
            nn.Conv3d(n_filters, 1, 1, padding=0),
            nn.Tanh()
        )

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()
    def forward(self, x, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]

        x4_up = self.block_four_up(x)
        x4_up = x4_up + x3

        x5 = self.block_five(x4_up)
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x2

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x1

        x7 = self.block_seven(x6_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x7 = self.dropout(x7)
        out = self.last_conv(x7)
        return out

# def TransBTS(_conv_repr=True, _pe_type="learned"):
#
#     img_dim = 8
#     num_channels = 1
#     patch_dim = 1
#     model = TransformerBTS(
#         img_dim,
#         patch_dim,
#         num_channels,
#         embedding_dim=512,
#         num_heads=8,
#         num_layers=4,
#         hidden_dim=512,
#         dropout_rate=0.1,
#         attn_dropout_rate=0.1,
#         conv_patch_representation=_conv_repr,
#         positional_encoding_type=_pe_type,
#     )
#
#     print("model build successful")
#     return model
class TransformerBTS(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(TransformerBTS, self).__init__()

        # 嵌入维度可以整除多头注意力的头，即将多个头分别对不同的维度自注意力
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        # ？
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 512 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)
        self.decoder = Decoder(n_channels=1)
        self.backbone = Backbone(n_channels=1)
        self.fusion_net = Fusion_Network()
        self.Enblock= EnBlock1(in_channels=self.embedding_dim)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv3d(
                64,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, input1, input2):
        features, x = self.backbone(input1, input2)
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            #x = self.fusion_net(l, t)
            # print(x.shape) torch.Size([2, 512, 8, 8, 8])
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)  # increase the channel dimension from 128 to embedding_dim
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            # print(x.shape) torch.Size([2, 8, 8, 8, 512])
            x = x.view(x.size(0), -1, self.embedding_dim)  # embed: torch.Size([2, 512, 512])

        else:
            #x = self.fusion_net(l, t)
            # x = self.backbone(x)
            # x = self.bn(x)
            # x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                    .unfold(3, 2, 2)
                    .unfold(4, 2, 2)
                    .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)  # position torch.Size([2, 512, 512])
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)  # transformer torch.Size([2, 512, 512])
        x = self.pre_head_ln(x)
        x = self._reshape_output(x)  # reshape torch.Size([2, 512, 8, 8, 8])

        return x, features

    def forward(self, input1, input2):

        #feature, encoder_output, intmd_encoder_outputs = self.encode(x)
        encoder_output, intmd_encoder_outputs = self.encode(input1, input2)
        transformer_output = self.Enblock(encoder_output)

        decoder_output = self.decoder(transformer_output, intmd_encoder_outputs)

        return decoder_output

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x
if __name__ == '__main__':
    x=torch.randn(1,1,64,64,64)
    y=torch.randn(1,1,64,64,64)
    model=TransformerBTS(
                                            img_dim=8,
                                            patch_dim=1,
                                            num_channels=1,
                                            embedding_dim=512,
                                            num_heads=8,
                                            num_layers=4,
                                            hidden_dim=512,
                                            dropout_rate=0.1,
                                            attn_dropout_rate=0.1,
                                            conv_patch_representation=True,
                                            positional_encoding_type="learned")
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    x=model(x,y)
    print(x.shape)
    