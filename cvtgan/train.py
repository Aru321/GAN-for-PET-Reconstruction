import torch
import data.Preprocess.datautils3d as util3d
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import normalized_root_mse
import model.CVT3D_Model as CVT3D_Model
import model.CrossIR as CrossIR
import model.cGAN_Model as cGAN
import model.CVT3D_Full as full
import model.Cross3D as Cross3D
import model.CVT3D_Model_Multi as CVT3D_Model_Multi
import model.Cross3D_single as Cross_single
import torch.nn as nn
from torch import optim
import os
from image_pool import ImagePool
import numpy as np
import argparse
import sys
import torch
import util
from torch.optim import lr_scheduler
from datetime import datetime
import SimpleITK as sitk
from medpy.io import load

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

# nvidia-smi
# 9,13,15
# ablation study
# python train.py --Ex_num 103 --arc proposed --batch_size 10 --data_l_cut_path ./dataset/processed/Lphant_cut --data_s_cut_path ./dataset/processed/Sphant_cut  --lr_G 2e-4 --lr_D 2e-4
# python train.py --Ex_num 1 --arc full --batch_size 8 --lr_G 2e-4 --lr_D 2e-4
# python train.py --Ex_num 1 --arc CrossIR --batch_size 12 --lr_G 2e-4 --lr_D 2e-4
# python train.py --Ex_num 1 --arc Cross_single --batch_size 10 --lr_G 2e-4 --lr_D 2e-5
# python train.py --Ex_num 8 --arc cGAN --batch_size 12 --lr_G 2e-4 --lr_D 2e-5
# python train.py --Ex_num 9 --arc cGAN --batch_size 12 --lr_G 2e-4 --lr_D 2e-5
# python train.py --Ex_num 10 --arc cGAN --batch_size 12 --lr_G 2e-4 --lr_D 2e-5
# python train.py --Ex_num 1 --arc proposed --batch_size 10 --lr_G 2e-4 --lr_D 2e-5
# python train.py --Ex_num 1 --arc Cross --batch_size 10 --lr_G 2e-4 --lr_D 1e-5
# python train.py --Ex_num 9 --arc ab_2 --lr_G 2e-4 --lr_D 1e-5
# python train.py --Ex_num 13 --devices x --pretrained_model --lr_G 2e-4 --lr_D 1e-5
# python train.py --Ex_num 15 --devices x --pretrained_model --lr_G 2e-4 --lr_D 1e-5
# python train.py --Ex_num 9 --devices x --pretrained_model --lr_G 2e-4 --lr_D 2e-5
# python train.py --Ex_num 13 --devices x --pretrained_model --lr_G 2e-4 --lr_D 2e-5
# python train.py --Ex_num 15 --devices x --pretrained_model --lr_G 2e-4 --lr_D 2e-5
def parse_option():
    parser = argparse.ArgumentParser('CVT3D training and evaluation script', add_help=False)
    parser.add_argument('--file_txt_dir', default="./dataset/split/", type=str,
                        help='path split txt file dir to dataset')
    parser.add_argument('--Ex_num', required=True, type=int, help='path split txt file dir to dataset')
    parser.add_argument('--data_l_cut_path', default='./dataset/processed/LPET_cut', type=str,
                        help='LPET cut data path')
    parser.add_argument('--data_s_cut_path', default='./dataset/processed/SPET_cut', type=str,
                        help='SPET cut data path')
    parser.add_argument('--epochs', default=70, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1, type=int, help="batch size for single GPU")
    parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES var from this string')
    parser.add_argument('--lr_G', '--learning-rate-G', default=2e-4, type=float, help='initial learning rate of G')
    parser.add_argument('--lr_D', '--learning-rate-D', default=2e-4, type=float, help='initial learning rate of D')
    parser.add_argument('--lamb', default=100, type=float, dest='lamb')
    parser.add_argument('--beta1', default=0.9, type=float, dest='beta1')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--arc', type=str, choices=['proposed', 'Cross', 'cGAN','Cross_single','CrossIR','full'],
                        help='proposed: CVT3D_Model, '
                             'cross:Cross3D,'
                             'ab_2: CVT3D_Model_PureCNN')
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
    trainloader = util3d.loadData(args.data_l_cut_path, '', args.data_s_cut_path, '', prefixs=train_imgs,
                                  batch_size=args.batch_size, shuffle=True)
    valloader = util3d.loadData(args.data_l_cut_path, '', args.data_s_cut_path, '', prefixs=val_imgs,
                                batch_size=1, shuffle=False)
    print("len:", len(trainloader))
    print("len:", len(valloader))

    # 设置训练显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices  # 设置可见显卡
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 单卡训练
    imgpool = ImagePool(5)
    # 设置超参数
    lr_G, lr_D = args.lr_G, args.lr_D  # 学习率
    beta1 = args.beta1  # 动量
    lamb = args.lamb  # L1
    epochs = args.epochs
    start_epoch = 0

    G = None
    D = None
    # 初始化G和D & 目标函数 & 优化器
    if args.arc == "proposed" or args.arc is None:
        G = CVT3D_Model.Generator().to(device)
    elif args.arc == "Cross":
        G = Cross3D.Generator().to(device)
    elif args.arc == "Cross_single":
        G = Cross_single.Generator().to(device)
    elif args.arc == "cGAN":
        G = cGAN.Generator().to(device)
        #D = CVT3D_Model_Multi.Discriminator().to(device)
    elif args.arc == "CrossIR":
        G = CrossIR.Generator().to(device)
    elif args.arc == "full":
        G = full.Generator().to(device)
    #
    D = CVT3D_Model_Multi.NLayerDiscriminator(input_nc=3, use_sigmoid=True).to(device)
    #

    BCELoss = nn.BCELoss().to(device)
    L1 = nn.L1Loss().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, 0.999))
    # 学习率
    schedulerG = lr_scheduler.MultiStepLR(optimizer_G, milestones=[50, 60], gamma=0.4)
    schedulerD = lr_scheduler.MultiStepLR(optimizer_D, milestones=[50, 60], gamma=0.4)
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
    record_file = open(full_path, 'w',encoding='utf-8',buffering = 1)
    print('--------args----------', file=record_file)  # msg也就是下面的Hello world!
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]), file=record_file)
    print('--------args----------', file=record_file)
    # init
    G.apply(util.weights_init)
    D.apply(util.weights_init)
    # 设置训练方式：1.设置是否继续训练（依照checkpoint）;2.设置（是否）预训练模型;3.都不设置，从头开始 -- 三者只能选其一
    if args.arc == "CVT3D" or args.arc is None:
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
            G = torch.load(args.pretrained_model_dir + 'pretrain_generator.pkl').to(device)
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
    l_edge=1.
    ERROR_CNT = 0
    torch.cuda.empty_cache()

    for epoch in range(start_epoch, epochs):
        # 用于存储每个图片的loss的列表：D_losses, G_losses
        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0
        for x, y in trainloader:
            batch += 1
            # 归一化
            X = (x - x.min()) / (x.max() - x.min())
            Y = (y - y.min()) / (y.max() - y.min())
            # 训练Discriminator
            # 训练Discriminator
            d_loss, g_loss = CVT3D_Model_Multi.GD_Train_single(D, G, X, Y, optimizer_D, L1, optimizer_G, device, imgpool,
                                                        lamb=100,l_edge=l_edge)
            D_losses.append(d_loss)
            G_losses.append(g_loss)
            # 平均loss
            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
            print('[%d / %d]: batch#%d loss_d= %.3f  loss_g= %.3f' %
                  (epoch + 1, epochs, batch, d_l, g_l))
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
            NMSE_vals = list()
            # 预测和保存
            for image_l, image_s in valloader:
                # 归一化
                testl = image_l
                image_l = (testl - testl.min()) / (testl.max() - testl.min())
                image_l = image_l.to(device)
                tests = image_s
                image_s = (tests - tests.min()) / (tests.max() - tests.min())
                image_s = np.squeeze(image_s.detach().numpy())
                res = G(image_l)
                # res=res*(l_max-l_min)+l_min
                res = res.cpu().detach().numpy()
                res = np.squeeze(res)
                #
                image_l = image_l.cpu().detach().numpy()
                image_l = np.squeeze(image_l)
                y = np.nonzero(image_s)  # 取非黑色部分
                image_s_1 = image_s[y]
                res_1 = res[y]
                #nmse
                cur_nmse = (normalized_root_mse(image_s, res)) ** 2
                NMSE_vals.append(cur_nmse)
                # 计算PSNR
                cur_psnr = (peak_signal_noise_ratio(res_1, image_s_1, data_range=1))
                PSNR_vals.append(cur_psnr)
            cur_mean_NMSE_val = np.mean(NMSE_vals)
            cur_mean_PSNR_val = np.mean(PSNR_vals)
            if cur_mean_PSNR_val > PSNR_val_best:
                PSNR_val_best = cur_mean_PSNR_val
                nmse_best=cur_mean_NMSE_val
                val_epoch_best = epoch
                torch.save(G.state_dict(), os.path.join(log_dir, "best_PSNR_model.pkl"))
                print('Val   => Epoch:{} Avg.Cur_PSNR:{} Avg.Best_PSNR(Epoch:{}):{} ,nmse:{}'.format(epoch, cur_mean_PSNR_val,
                                                                                          val_epoch_best,
                                                                                          PSNR_val_best,
                                                                                            nmse_best
                                                                                          ), file=record_file)
                print('nmse')
                print(
                    "Model Was Saved ! Current Best Avg. PSNR: {} in {} ;Current Avg. PSNR: {}".format(
                        PSNR_val_best, val_epoch_best, cur_mean_PSNR_val
                    )
                )
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


if __name__ == '__main__':
    args = parse_option()
    print(args)
    train(args)
