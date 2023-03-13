import torch
import util
from torch.optim import lr_scheduler
import data.Preprocess.datautils3d as util3d
from image_pool import ImagePool
import model.Cross3D_ab2 as Cross3D_ab2
import model.Cross3D as Cross3D
import model.Cross3D2 as Cross3D2
import model.Cross3D3 as Cross3D3
from skimage.metrics import peak_signal_noise_ratio
import model.CVT3D_Model as CVT3D_Model
import model.CVT3D_Model_Multi_new as CVT3D_Model_Multi_new
import model.CVT3D_Model_Multi as CVT3D_Model_Multi
import model.multitransformerGAN_model2 as MultiTrans
import model.Cross3D_ab1 as ab1
import model.SwinTransformer_shuffle_channel_attn_unet as shuffle_channel
import model.hybrid as hybrid
import torch.nn as nn
from torch import optim
import os
import numpy as np
import argparse
import sys
from datetime import datetime
from skimage.metrics import normalized_root_mse
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import SimpleITK as sitk
from medpy.io import load

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, './model')


# file_txt_dir file_dir_str
# epochs int
# batch_size int
# devices device_str
# lr_G float
# lr_D float
# lamb float
# beta1 float
# resume - checkpoint boolean
# use_checkpoint boolean?
# checkpoint_dir file_dir_str
# val_epoch_inv int
# temp_val_img_cut file_dir_str
# res_model file_dir_str
# pretrained_model boolean
# pretrained_model_dir file_dir_str
# python train_multi.py --Ex_num 114514 --arc proposed --batch_size 5 --data_l_cut_path ./dataset/processed/Lphant_cut --data_s_cut_path ./dataset/processed/Sphant_cut --data_mri_cut_path ./dataset/processed/T1phant_cut --lr_G 2e-4 --lr_D 2e-4
# nvidia-smi
# 9,13,15
# ablation study
# python train_multi.py --Ex_num 2 --arc ab2 --batch_size 12 --epochs 70 --lr_G 2e-4 --lr_D 2e-4
# python train_multi.py --Ex_num 1 --arc shuffle --batch_size 10 --epochs 70 --lr_G 2e-4 --lr_D 2e-4 --use_checkpoint
# python train_multi.py --Ex_num 1 --arc Cross3 --batch_size 5 --epochs 70 --lr_G 2e-4 --lr_D 2e-4
# python train_multi.py --Ex_num 1 --arc Cross2 --batch_size 5 --epochs 70 --lr_G 2e-4 --lr_D 2e-4
# python train_multi.py --Ex_num 1 --arc Cross --batch_size 5 --epochs 70 --lr_G 2e-4 --lr_D 2e-4
# python train_multi.py --Ex_num 9 --arc proposed --epochs 70 --batch_size 5 --lr_G 2e-4 --lr_D 1e-4
# python train_multi.py --Ex_num 1 --arc sota --epochs 70 --batch_size 10 --lr_G 2e-4 --lr_D 2e-4
# python train_multi.py --Ex_num 1 --arc new --epochs 50 --batch_size 10 --lr_G 2e-4 --lr_D 2e-5
def parse_option():
    parser = argparse.ArgumentParser('CVT3D training and evaluation script', add_help=False)
    parser.add_argument('--file_txt_dir', default="./dataset/split/", type=str,
                        help='path split txt file dir to dataset')
    parser.add_argument('--Ex_num', required=True, type=int, help='path split txt file dir to dataset')
    parser.add_argument('--data_l_cut_path', default='./dataset/processed/LPET_cut', type=str,
                        help='LPET cut data path')
    parser.add_argument('--data_s_cut_path', default='./dataset/processed/SPET_cut', type=str,
                        help='SPET cut data path')
    parser.add_argument('--data_mri_cut_path', default='./dataset/processed/T1_cut', type=str,
                        help='T1 cut data path')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1, type=int, help="batch size for single GPU")
    parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES var from this string')
    parser.add_argument('--lr_G', '--learning-rate-G', default=2e-4, type=float, help='initial learning rate of G')
    parser.add_argument('--lr_D', '--learning-rate-D', default=2e-4, type=float, help='initial learning rate of D')
    parser.add_argument('--lamb', default=100, type=float, dest='lamb')
    parser.add_argument('--beta1', default=0.9, type=float, dest='beta1')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--arc', type=str, choices=['proposed', 'sota', 'new', 'Cross', 'Cross2', 'Cross3', 'shuffle','ab1','ab2'],
                        help='proposed: CVT3D_Model, '
                             'sota: MultiTrans,'
                             'new: lightened')
    parser.add_argument('--checkpoint_file_path', default="./check_point.pkl", type=str, help='pretrained weight path')
    parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint during training')
    parser.add_argument('--val_epoch_inv', default=1, type=int, help='validation interval epochs')
    parser.add_argument('--temp_val_img_cut_dir', default="./temp/", type=str, help='pretrained weight path')
    parser.add_argument('--res_model_dir', default="./res_model/", type=str, help='pretrained weight path')
    parser.add_argument('--pretrained_model', action='store_true', help='pretrained weight path')
    parser.add_argument('--pretrained_model_dir', default="./pretrained_model/", type=str,
                        help='pretrained weight path')

    # parser.add_argument('-c',action='store_true') # store_true 是指带触发action时为真，不触发则为假
    # # python test.py -c   => c是true（触发）
    # # python test.py    => c是false（无触发）
    args, unparsed = parser.parse_known_args()
    return args


def readTxtLineAsList(txt_path):
    fi = open(txt_path, 'r')
    txt = fi.readlines()
    res_list = []
    for w in txt:
        w = w.replace('\n', '')
        res_list.append(w)
    return res_list


def train(args):
    # 依靠文件设置数据集切分
    train_txt_path = args.file_txt_dir + r"Ex" + str(args.Ex_num) + r"/train.txt"
    val_txt_path = args.file_txt_dir + r"Ex" + str(args.Ex_num) + r"/val.txt"
    train_imgs = readTxtLineAsList(train_txt_path)
    val_imgs = readTxtLineAsList(val_txt_path)
    print(train_imgs)
    print(val_imgs)
    trainloader = util3d.loadMultiData(args.data_l_cut_path, '', args.data_s_cut_path, '', args.data_mri_cut_path, '',
                                       prefixs=train_imgs,
                                       batch_size=args.batch_size, shuffle=True)
    valloader = util3d.loadMultiData(args.data_l_cut_path, '', args.data_s_cut_path, '', args.data_mri_cut_path, '',
                                     prefixs=val_imgs,
                                     batch_size=args.batch_size, shuffle=False)
    print("len:", len(trainloader))
    print("len:", len(valloader))

    # 设置训练显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices  # 设置可见显卡
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 单卡训练

    # 设置超参数
    lr_G, lr_D = args.lr_G, args.lr_D  # 学习率
    beta1 = args.beta1  # 动量
    lamb = args.lamb  # L1
    epochs = args.epochs
    start_epoch = 0
    #
    imgpool = ImagePool(5)
    #
    G = None
    D = None
    # 初始化G和D & 目标函数 & 优化器
    if args.arc == "proposed" or args.arc is None:
        G = CVT3D_Model_Multi.Generator().to(device)
        D = CVT3D_Model_Multi.Discriminator().to(device)

    elif args.arc == "sota":
        G = MultiTrans.TransformerBTS(
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
            positional_encoding_type="learned").to(device)
        D = CVT3D_Model_Multi.NLayerDiscriminator(input_nc=2, use_sigmoid=True).to(device)
        # D = CVT3D_Model_Multi.Discriminator().to(device)
    elif args.arc == "new":
        G = CVT3D_Model_Multi_new.TransformerBTS(
            img_dim=4,
            patch_dim=1,
            num_channels=1,
            embedding_dim=512,
            num_heads=8,
            num_layers=4,
            hidden_dim=512,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            conv_patch_representation=True,
            positional_encoding_type="learned").to(device).to(device)
        D = CVT3D_Model_Multi.Discriminator().to(device)
    elif args.arc == "hybrid":
        G = hybrid.TransformerBTS(
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
            positional_encoding_type="learned").to(device)
        # D = CVT3D_Model_Multi.NLayerDiscriminator(input_nc=2,use_sigmoid=True).to(device)
        D = CVT3D_Model_Multi.Discriminator().to(device)
        # D = CVT3D_Model_Multi.NLayerDiscriminator(input_nc=2, use_sigmoid=True).to(device)
    elif args.arc == "Cross":
        G = Cross3D.Generator().to(device)
        D = CVT3D_Model_Multi.NLayerDiscriminator(input_nc=3, use_sigmoid=True).to(device)
        # D = CVT3D_Model_Multi.Discriminator().to(device)
    elif args.arc == "Cross2":
        G = Cross3D2.Generator().to(device)
        D = CVT3D_Model_Multi.Discriminator().to(device)
    elif args.arc == "Cross3":
        G = Cross3D3.Generator().to(device)
        # D = CVT3D_Model_Multi.NLayerDiscriminator(input_nc=3,use_sigmoid=True).to(device)
        D = CVT3D_Model_Multi.Discriminator().to(device)
    elif args.arc == "shuffle":
        G = shuffle_channel.SwinTransformerSys3D_shuffle_channel_attn_unet(
            img_size=(64, 64, 64),
            in_chans=2,
            embed_dim=144,
            patch_size=(2, 2, 2),
            num_classes=1,
            depths=[2, 2, 2, 1],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[6, 8, 12, 24],
            window_size=(4, 4, 4),
            final_upsample="expand_first",
            fuse_method_e="CatChannelAttn_Num2",
            fuse_method_d="CatChannelAttn_Num2",
        ).to(device)
        D = CVT3D_Model_Multi.NLayerDiscriminator(input_nc=3, use_sigmoid=True).to(device)
    elif args.arc == "ab1":
        G=ab1.Generator().to(device)
        D = CVT3D_Model_Multi.Discriminator().to(device)
    elif args.arc == "ab2":
        G = Cross3D_ab2.Generator().to(device)
        D = CVT3D_Model_Multi.Discriminator().to(device)

        '''
    elif args.arc == "ab_2":
        G = CVT3D_Model_PureCNN.Generator().to(device)
        D = CVT3D_Model_PureCNN.Discriminator().to(device)
    '''

    BCELoss = nn.BCELoss().to(device)
    L1 = nn.L1Loss().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, 0.999))
    # 学习率
    schedulerG = lr_scheduler.MultiStepLR(optimizer_G, milestones=[50, 55, 60, 65], gamma=0.5)
    schedulerD = lr_scheduler.MultiStepLR(optimizer_D, milestones=[50, 55, 60, 65], gamma=0.5)
    # 用于训练过程记录
    val_epoch_best = 0
    PSNR_val_best = 0.0
    result_dir = args.res_model_dir + r"Ex" + str(args.Ex_num) + "/" + "lr_G-" + '{:g}'.format(
        lr_G) + " lr_D-"'{:g}'.format(lr_D)
    if args.arc != "CVT3D" and args.arc is not None:
        result_dir += " " + args.arc
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_dir, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    full_path = log_dir + '/result_record.txt'  # 也可以创建一个.doc的word文档
    record_file = open(full_path, 'w', encoding='utf-8', buffering=1)
    print('--------args----------', file=record_file)  # msg也就是下面的Hello world!
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]), file=record_file)
    print('--------args----------', file=record_file)

    # 设置训练方式：1.设置是否继续训练（依照checkpoint）;2.设置（是否）预训练模型;3.都不设置，从头开始 -- 三者只能选其一
    if args.arc == "proposed" or args.arc is None:
        if args.resume:
            if os.path.isfile(args.checkpoint_file_path):
                checkpoint = torch.load(args.checkpoint_file_path)
                start_epoch = checkpoint['epoch'] + 1
                G.load_state_dict(checkpoint['G'])
                D.load_state_dict(checkpoint['D'])
                optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            else:
                assert "checkpoint path not exsited or not set"
        elif args.pretrained_model:
            print(args.pretrained_model_dir + 'pretrain_generator.pkl')
            G = CVT3D_Model_Multi.Generator()
            G.load_state_dict(torch.load(args.pretrained_model_dir + 'pretrain_generator.pkl'))
            G = G.to(device)
            # D = torch.load(args.pretrained_model_dir + 'discriminator.pkl').to(device)
            print("=> loaded pretrained model")
        else:
            print("=> No resume or pretrained")
    else:
        print("=> Ablation study: use " + args.arc)

    # 模型大小查看
    total_params = sum(p.numel() for p in G.parameters())
    print(total_params)

    # 训练
    D_Loss, G_Loss, Epochs = [], [], range(1, epochs + 1)  # 一次epoch的loss
    l_edge = 1
    ERROR_CNT = 0
    torch.cuda.empty_cache()
    # init
    G.apply(util.weights_init)
    D.apply(util.weights_init)
    #
    for epoch in range(start_epoch, epochs):
        G.train()
        # 用于存储每个图片的loss的列表：D_losses, G_losses
        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0
        for x, y, mri in trainloader:
            batch += 1
            # 归一化
            X = (x - x.min()) / (x.max() - x.min())
            Y = (y - y.min()) / (y.max() - y.min())
            MRI = (mri - mri.min()) / (mri.max() - mri.min())
            # 训练Discriminator
            d_loss, g_loss = CVT3D_Model_Multi.GD_Train_with_edge(D, G, X, Y, MRI, optimizer_D, L1, optimizer_G,
                                                                   device, imgpool, lamb=100, l_edge=l_edge)
            D_losses.append(d_loss)
            G_losses.append(g_loss)
            '''
            if g_loss>100:
                savImgX = sitk.GetImageFromArray(X)
                sitk.WriteImage(savImgX,'./error'+'/'+'errorx_'+str(ERROR_CNT)+'.img')
                savImgY = sitk.GetImageFromArray(Y)
                sitk.WriteImage(savImgY,'./error'+'/'+'errory_'+str(ERROR_CNT)+'.img')
                ERROR_CNT+=1
            '''
            # 平均loss
            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
            print('[%d / %d]: batch#%d loss_d= %.3f  loss_g= %.3f lr_g=%.6f lr_d=%.6f' %
                  (epoch + 1, epochs, batch, d_l, g_l, optimizer_G.state_dict()['param_groups'][0]['lr'],
                   optimizer_D.state_dict()['param_groups'][0]['lr']))
        # 保存每次epoch的loss
        D_Loss.append(d_l)
        G_Loss.append(g_l)
        print("Train => Epoch:{} Avg.D_Loss:{} Avg.G_Loss:{}".format(epoch, d_l, g_l), file=record_file)
        # 保存 the last model (Generator)
        torch.save(G, os.path.join(log_dir, 'last_model.pkl'))
        schedulerG.step()
        schedulerD.step()
        # 保存训练储存点，并不断更新check_point
        if args.use_checkpoint:
            checkpoint = {
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }
            torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pkl'))

        # 每几个epoch val一次
        if epoch % args.val_epoch_inv == 0:
            G.eval()  # 调整成val的模式
            PSNR_vals = list()
            # 预测和保存
            for image_l, image_s, mri in valloader:
                # 归一化
                testl = image_l
                image_l = (testl - testl.min()) / (testl.max() - testl.min())
                image_l = image_l.to(device)
                tests = image_s
                image_s = (tests - tests.min()) / (tests.max() - tests.min())
                testmri = mri
                testmri = (testmri - testmri.min()) / (testmri.max() - testmri.min())
                testmri = testmri.to(device)
                image_s = np.squeeze(image_s.detach().numpy())
                #cat_image = torch.cat([image_l, testmri], dim=1)
                res = G(image_l,testmri)
                # res=res*(l_max-l_min)+l_min
                res = res.cpu().detach().numpy()
                res = np.squeeze(res)
                #
                image_l = image_l.cpu().detach().numpy()
                image_l = np.squeeze(image_l)
                y = np.nonzero(image_s)  # 取非黑色部分
                image_s_1 = image_s[y]
                res_1 = res[y]
                # 计算PSNR
                cur_psnr = (peak_signal_noise_ratio(res_1, image_s_1, data_range=1))
                # ssim
                # cur_ssim = structural_similarity(res, image_s, multichannel=True)
                # cur_mrse = normalized_root_mse(image_s,res)**2
                PSNR_vals.append(cur_psnr)
            cur_mean_PSNR_val = np.mean(PSNR_vals)
            if cur_mean_PSNR_val > PSNR_val_best:
                PSNR_val_best = cur_mean_PSNR_val
                val_epoch_best = epoch
                torch.save(G.state_dict(), os.path.join(log_dir, "best_PSNR_model.pkl"))
                print('Val   => Epoch:{} Avg.Cur_PSNR:{} Avg.Best_PSNR(Epoch:{}):{}'.format(epoch, cur_mean_PSNR_val,
                                                                                            val_epoch_best,
                                                                                            PSNR_val_best
                                                                                            ), file=record_file)
                print(
                    "Model Was Saved ! Current Best Avg. PSNR: {} in {} ;Current Avg. PSNR: {}".format(
                        PSNR_val_best, val_epoch_best, cur_mean_PSNR_val
                    )
                )
                # print('ssim:',cur_ssim)
                # print('nmse:',cur_mrse)
            else:
                print('Val   => Epoch:{} Avg.Cur_PSNR:{} Avg.Best_PSNR(Epoch:{}): {}'.format(epoch, cur_mean_PSNR_val,
                                                                                             val_epoch_best,
                                                                                             PSNR_val_best
                                                                                             ), file=record_file)
                print(
                    "Model Not Saved ! Current Best Avg. PSNR: {} in {} ;Current Avg. PSNR: {}".format(
                        PSNR_val_best, val_epoch_best, cur_mean_PSNR_val
                    )
                )
                # print('ssim:', cur_ssim)
                # print('nmse:', cur_mrse)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    args = parse_option()
    print(args)
    train(args)
