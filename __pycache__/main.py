import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from gradient_edge_model import Gradient_Edge_Restruct_Network, Remove_Inpaint_Network, Restruct_Guidance_Network, \
    Discriminator
from network.gradient_net import Gradient_Restruct_Network,Gradient_Restruct_Network1,Restruct_Guidance_Network1
# from utils import PSNR, MatrixToImage
from loss.SSIMLoss import SSIMLoss
from loss.SILoss import SILoss
from loss.MMDLoss import MMDLoss
import torchvision
from tensorboardX import SummaryWriter
import datetime

from test import test_remove_inpaint,blind_test,test_restruct_guidance,test_restruct_guidance1
from train import remove_inpaint_train, edge_gradient_train,restruct_guidance_train,restruct_guidance1_train
# from config import Config



model_names = 'sasa'
dataset_names = ""

global args, save_path

def main():
    parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = load_config(parser)
    torch.cuda.set_device(args.device)
    save_path = '{}epochs{}_b{}_lr{}'.format(
        args.epochs,
        '_epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)

    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H")
        save_path = os.path.join(timestamp, save_path)

    save_path = os.path.join(args.dataset, save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        print(save_path)
        os.makedirs(save_path)

    # 初始化,加载预训练模型
    gen_model,gen_optimizer,dis_model,dis_optimizer = model_load(args)

    ####load the loss functions
    ssimLoss = SSIMLoss().cuda(args.device)
    L1Loss = nn.L1Loss().cuda(args.device)
    siLoss = SILoss().cuda(args.device)
    mmdLoss = MMDLoss(args.device).cuda(args.device)

    # 训练不同的模型
    # args.todo = "test"
    if args.todo == "train":
        # 0:边缘与梯度重构
        if args.model_type == 0:
            print("edge_gradient_train start\n")
            edge_gradient_train(gen_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, gen_model,args)
        # 1:使用边缘图与梯度图进行去反射与修复
        elif args.model_type == 1:
            print("remove_inpaint_train start\n")
            remove_inpaint_train( gen_model,gen_optimizer,dis_model,dis_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, args)
        # 2:一站式训练,使用GAN
        elif args.model_type == 2:
            print("restruct_guidance_train start\n")
            restruct_guidance1_train(gen_model,gen_optimizer,dis_model,dis_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, args)

    elif args.todo == "test":
        # 0:边缘与梯度重构
        if args.model_type == 0:
            edge_gradient_train(gen_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, gen_model,args)
        # 1:使用边缘图与梯度图进行去反射与修复
        elif args.model_type == 1:
            test_remove_inpaint(gen_model)
        # 2:一站式训练,使用GAN
        elif args.model_type == 2:
            # test_restruct_guidance(gen_model)
            test_restruct_guidance1(gen_model,gen_optimizer,dis_model,dis_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, args)
    else:
        blind_test(ssimLoss, L1Loss, siLoss, mmdLoss, gen_model)


def model_load(args):
    print("args.device:",args.device)
    vgg = torchvision.models.vgg16_bn(pretrained=True)  ##load the vgg model
    vgglist = list(vgg.features.children())
    if args.model_type == 0:
        # 卷积进行结构恢复
        # gen_model = Gradient_Edge_Restruct_Network(vgglist)  ##start the Gradient_Edge training
        # 注意力进行结构恢复
        # gen_model = Gradient_Restruct_Network(vgglist)  ##start the Gradient_Edge training
        # 一段式的网络
        gen_model = Gradient_Restruct_Network1(vgglist)  ##start the Gradient_Edge training
    elif args.model_type == 1 :
        gen_model = Remove_Inpaint_Network(vgglist)  ##start the  Remove_Inpaint training
    elif args.model_type == 2:
        # gen_model = Restruct_Guidance_Network(vgglist)  ##start the  Restruct_Guidance training
        # 加入注意力的一段式网络
        gen_model = Restruct_Guidance_Network1(vgglist)

    gen_optimizer = optim.Adam(gen_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer, milestones=args.milestones, gamma=0.5)

    gen_model = gen_model.cuda(args.device)

    dis_model = Discriminator(in_channels=3, use_sigmoid=True)
    dis_model = dis_model.cuda(args.device)
    dis_optimizer = optim.Adam(
        params=dis_model.parameters(),
        lr=args.lr
    )
    # 是否加载预训练模型
    # 默认不加载
    args.pretrained = True
    if args.pretrained:
        print("load pretrained model")
        if args.model_type == 0:
            gen_ckpt_data = torch.load(
                r'ckpt/checkpoint_edge_gradient1_150.pth')
            # dis_ckpt_data = torch.load(
            #     r'ckpt/checkpoint_dis_restruct_guidance_0.pth')
        elif args.model_type == 1:
            gen_ckpt_data = torch.load(
                r'ckpt/checkpoint_gen_remove_inpaint_120.pth')
            dis_ckpt_data = torch.load(
                r'ckpt/checkpoint_dis_remove_inpaint_120.pth')
        elif args.model_type == 2:
            gen_ckpt_data = torch.load(
                r'ckpt/checkpoint_gen_restruct_guidance1_330.pth')
            dis_ckpt_data = torch.load(
                r'ckpt/checkpoint_dis_restruct_guidance1_330.pth')
        gen_model.load_state_dict(gen_ckpt_data['state_dict'])
        dis_model.load_state_dict(dis_ckpt_data['state_dict'])

    return gen_model,gen_optimizer,dis_model,dis_optimizer

def load_config(parser):
    parser.add_argument('--data', metavar='DIR', default=r'datasets/train_set',
                        help='path to dataset')
    parser.add_argument('--dataset', metavar='DATASET', default=r'datasets',
                        choices=dataset_names,
                        help='dataset type : ' +
                             ' | '.join(dataset_names))
    parser.add_argument('-s', '--split', default=80,
                        help='test-val split file')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='FaceGeneration',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names))
    parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                        help='solver algorithms')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epoch-size', default=30000, type=int, metavar='N',
                        help='manual epoch size (will match dataset size if set to 0)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--bias-decay', default=0, type=float,
                        metavar='B', help='bias decay')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--no-date', action='store_true',
                        help='don\'t append date timestamp to folder')
    parser.add_argument('--milestones', default=[18, 60, 80], nargs='*',
                        help='epochs at which learning rate is divided by 2')

    ## 比较常用的参数
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--device', default=2, type=int, metavar='N',
                        help='set deviceid')
    parser.add_argument('--model_type', default=2, type=int, metavar='N',
                        help='set model_type\n0:"edge and gradient"\n1:"remove and inpaint"\n2:"restruct and guidance')
    parser.add_argument('--todo', default="train", type=str, metavar='N',
                        help='choice train or test')
    parser.add_argument('--pretrained', dest='pretrained', default=False,
                        help='path to pre-trained model')
    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
    parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
    # main("test")
    # main("blind_test")