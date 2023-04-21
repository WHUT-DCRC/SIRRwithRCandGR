import argparse, os
import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from gradient_edge_model import Gradient_Edge_Restruct_Network,Remove_Inpaint_Network
# from utils import PSNR, MatrixToImage
from torchvision.utils import save_image
from SSIMLoss import SSIMLoss
from SILoss import SILoss
from MMDLoss import MMDLoss
import torchvision
from tensorboardX import SummaryWriter
import flow_transforms
import scipy.io as sio
import datetime
import torchvision.transforms as transforms
# import datasets
import time
from torchvision.datasets import ImageFolder
from PIL import Image
from test import test,blind_test
from tqdm import tqdm


transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(self.opt.DATASET.MEAN, self.opt.DATASET.STD)
    ])
grayscale_transform = transforms.Compose([
   transforms.Grayscale(1), #这一句就是转为单通道灰度图像
   transforms.ToTensor(),
])

# 本地
# mixture_set = ImageFolder(root=r"datasets/train_set/mixture",
#                           transform=transform)
# gradient_set = ImageFolder(root=r"datasets/train_set/gradient",
#                            transform=grayscale_transform)
# edge_set = ImageFolder(root=r"datasets/train_set/edge",
#                            transform=grayscale_transform)
# mask_set = ImageFolder(root=r"datasets/train_set/mask",
#                            transform=grayscale_transform)

# 服务器
mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/mixture",
                          transform=transform)
reflection_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/reflection",
                             transform=transform)
background_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/background",
                             transform=transform)
gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/gradient",
                           transform=grayscale_transform)
edge_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/edge",
                       transform=grayscale_transform)
mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/mask",
                       transform=grayscale_transform)

def edge_gradient_train( optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, model , args):
    model.train()
    NUM_TOTAL_STEP = 1001

    mixture_loader = data.DataLoader(mixture_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    edge_loader = data.DataLoader(edge_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)

    for num_step in tqdm(range(NUM_TOTAL_STEP)):
        mixtureit = iter(mixture_loader)
        gradientit = iter(gradient_loader)
        edgeit = iter(edge_loader)
        maskit = iter(mask_loader)
        iter_step = 0
        totalLoss_count = 0
        el_count = 0
        gl_count = 0
        while iter_step < int(len(mixture_set)/args.batch_size):
            iter_step += 1
            mixture, _ = next(mixtureit)
            gradient, _ = next(gradientit)
            edge, _ = next(edgeit)
            mask, _ = next(maskit)

            inputs = mixture.float().cuda(args.device)
            gradients = gradient.float().cuda(args.device)
            edges = edge.float().cuda(args.device)
            masks = mask.float().cuda(args.device)

            output = model.forward(inputs,masks)

            outputE = output[0]
            outputG = output[1]
            el = ssimLoss(outputE,  edges) \
                 # + 0.1*L1Loss(outputE, edges) \
                 # + 0.5 * mmdLoss(outputE,edges)  ##loss functios for background

            gl = ssimLoss(outputG, gradients)\
                # + 0.1*L1Loss(outputG, gradients)\
                # + 0.5 * mmdLoss(outputG,gradients)##loss functios for gradient

            totalLoss = el+gl

            totalLoss_count += totalLoss
            el_count += el
            gl_count += gl


            optimizer.zero_grad()
            totalLoss.backward()  ##backward the loss fucntions
            optimizer.step()
            # print("after 1 num_step:totalLoss = ", totalLoss)
        if num_step % 10 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            print("totalLoss:", totalLoss_count/500, "\nloss_edge:", el_count/500, "\nloss_gradient:", gl_count/500)

            ## 原始输入
            im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            im_gradient = Image.fromarray((gradient[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            im_gradient_mask = Image.fromarray(((gradient[0][0]*(1.0-mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_edge = Image.fromarray((edge[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            im_edge_mask = Image.fromarray(((edge[0][0] * (1.0 - mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_mask = Image.fromarray((mask[0][0] * 255.0).cpu().numpy().astype(np.uint8))

            ## 生成输出
            im_outputE = Image.fromarray((outputE[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))
            im_outputG = Image.fromarray((outputG[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))

            ## 保存图片
            im_input.save("output/mixture{}.jpg".format(num_step))
            im_gradient.save("output/gradient{}.jpg".format(num_step))
            im_gradient_mask.save("output/gradient_mask{}.jpg".format(num_step))
            im_edge.save("output/edge{}.jpg".format(num_step))
            im_edge_mask.save("output/edge_mask{}.jpg".format(num_step))
            im_mask.save("output/mask{}.jpg".format(num_step))
            im_outputE.save("output/outputE{}.jpg".format(num_step))
            im_outputG.save("output/outputG{}.jpg".format(num_step))
        if num_step % 30 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            # save the model
            save_checkpoint({
                'num_step': num_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, num_step )

def remove_inpaint_train( optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, model, args):
    model.train()
    NUM_TOTAL_STEP = 1001

    mixture_loader = data.DataLoader(mixture_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    background_loader = data.DataLoader(background_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    reflection_loader = data.DataLoader(reflection_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    edge_loader = data.DataLoader(edge_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)

    for num_step in tqdm(range(NUM_TOTAL_STEP)):
        mixtureit = iter(mixture_loader)
        backgroundit = iter(background_loader)
        reflectionit = iter(reflection_loader)
        gradientit = iter(gradient_loader)
        edgeit = iter(edge_loader)
        maskit = iter(mask_loader)
        iter_step = 0
        totalLoss_count = 0
        rl_count = 0
        il_count = 0
        while iter_step < int(len(mixture_set)/args.batch_size):
            iter_step += 1
            mixture, _ = next(mixtureit)
            background, _ = next(backgroundit)
            reflection, _ = next(reflectionit)
            gradient, _ = next(gradientit)
            edge, _ = next(edgeit)
            mask, _ = next(maskit)

            inputs = mixture.float().cuda(args.device)
            backgrounds = background.float().cuda(args.device)
            reflections = reflection.float().cuda(args.device)
            gradients = gradient.float().cuda(args.device)
            edges = edge.float().cuda(args.device)
            masks = mask.float().cuda(args.device)

            output = model.forward(inputs,gradients,edges,masks)

            outputR = output[0]
            outputI = output[1]

            # print(outputR.shape,outputI.shape)
            removeloss = 0.5*ssimLoss(outputR*(1-masks),  backgrounds*(1-masks)) \
                 + 0.5*L1Loss(outputR*(1-masks), backgrounds*(1-masks)) \
                 # + 0.5 * mmdLoss(outputE,edges)  ##loss functios for background

            inpaintloss = 0.5*ssimLoss(outputI*masks, backgrounds*masks)\
                + 0.5*L1Loss(outputI*masks, backgrounds*masks)\
                # + 0.5 * mmdLoss(outputG,gradients)##loss functios for gradient


            totalLoss = removeloss + inpaintloss

            totalLoss_count += totalLoss
            rl_count += removeloss
            il_count += inpaintloss


            optimizer.zero_grad()
            totalLoss.backward()  ##backward the loss fucntions
            optimizer.step()
        if num_step % 10 == 0 and num_step != 0:
        # if num_step % 1 == 0 :

            print("totalLoss:", totalLoss_count/500, "\nloss_remove:", rl_count/500, "\nloss_inpaint:", il_count/500)

            # print("save img")
            ## 原始输入
            im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            im_background = Image.fromarray(
                np.transpose((backgrounds[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            im_reflection = Image.fromarray(
                np.transpose(((inputs[0]-backgrounds[0]) * 52.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            im_gradient = Image.fromarray((gradient[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            # im_gradient_mask = Image.fromarray(((gradient[0][0]*(1.0-mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_edge = Image.fromarray((edge[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            # im_edge_mask = Image.fromarray(((edge[0][0] * (1.0 - mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_mask = Image.fromarray((mask[0][0] * 255.0).cpu().numpy().astype(np.uint8))


            ## 生成输出
            im_outputR = Image.fromarray((outputR[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))
            im_outputI = Image.fromarray((outputI[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))

            im_input.save("output/mixture{}.jpg".format(num_step))
            im_background.save("output/background{}.jpg".format(num_step))
            im_gradient.save("output/gradient{}.jpg".format(num_step))
            # im_gradient_mask.save("output/gradient_mask{}.jpg".format(num_step))
            im_edge.save("output/edge{}.jpg".format(num_step))
            # im_edge_mask.save("output/edge_mask{}.jpg".format(num_step))
            im_mask.save("output/mask{}.jpg".format(num_step))
            im_outputR.save("output/outputR{}.jpg".format(num_step))
            im_outputI.save("output/outputI{}.jpg".format(num_step))
        if num_step % 30 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            # save the model
            save_checkpoint({
                'num_step': num_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, num_step,1 )

def restruct_guidance_train( optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, model, train_writer, args):
    model.train()
    NUM_TOTAL_STEP = 1001


    # reflection_set = ImageFolder(
    #     root=r"datasets/train_set/reflection",
    #     transform=transform)
    # mixture_set = ImageFolder(root=r"C:\Users\70432\Desktop\remove_dataset\opr\mixture",
    #                           transform=transform)
    # background_set = ImageFolder(
    #     root=r"C:\Users\70432\Desktop\remove_dataset\opr\background",
    #     transform=transform)


    mixture_loader = data.DataLoader(mixture_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    # background_loader = data.DataLoader(background_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    # reflection_loader = data.DataLoader(reflection_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    edge_loader = data.DataLoader(edge_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)

    for num_step in tqdm(range(NUM_TOTAL_STEP)):
    #     print(num_step, '/', NUM_TOTAL_STEP)
        mixtureit = iter(mixture_loader)
        # backgroundit = iter(background_loader)
        # reflectionit = iter(reflection_loader)
        gradientit = iter(gradient_loader)
        edgeit = iter(edge_loader)
        maskit = iter(mask_loader)
        iter_step = 0
        totalLoss_count = 0
        el_count = 0
        gl_count = 0
        while iter_step < int(len(mixture_set)/args.batch_size):
            iter_step += 1
            # print(iter_step, '/', int(len(mixture_set) / args.batch_size))
            mixture, _ = next(mixtureit)
            # background, _ = next(backgroundit)
            # reflection, _ = next(reflectionit)
            gradient, _ = next(gradientit)
            edge, _ = next(edgeit)
            mask, _ = next(maskit)

            inputs = mixture.float().cuda(args.device)
            # backgrounds = background.float().cuda(args.device)
            # reflections = reflection.float().cuda(args.device)
            gradients = gradient.float().cuda(args.device)
            edges = edge.float().cuda(args.device)
            masks = mask.float().cuda(args.device)

            output = model.forward(inputs,masks)

            outputE = output[0]
            outputG = output[1]
            el = ssimLoss(outputE,  edges) \
                 # + 0.1*L1Loss(outputE, edges) \
                 # + 0.5 * mmdLoss(outputE,edges)  ##loss functios for background

            gl = ssimLoss(outputG, gradients)\
                # + 0.1*L1Loss(outputG, gradients)\
                # + 0.5 * mmdLoss(outputG,gradients)##loss functios for gradient

            # + 0.5 * L1Loss(outputR,  reflections)
            # rl = ssimLoss(outputR, inputs-backgrounds) \
            #      + 0.5 * mmdLoss(
            #     outputR, inputs-backgrounds)  ##loss functios for reflection

            totalLoss = el+gl

            totalLoss_count += totalLoss
            el_count += el
            gl_count += gl


            optimizer.zero_grad()
            totalLoss.backward()  ##backward the loss fucntions
            optimizer.step()
            # print("after 1 num_step:totalLoss = ", totalLoss)
        if num_step % 10 == 0 and num_step != 0:
        # if num_step % 1 == 0 :

            print("totalLoss:", totalLoss_count/500, "\nloss_edge:", el_count/500, "\nloss_gradient:", gl_count/500)

            # print("save img")
            ## 原始输入
            im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            # im_background = Image.fromarray(
            #     np.transpose((backgrounds[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            # im_reflection = Image.fromarray(
            #     np.transpose(((inputs[0]-backgrounds[0]) * 52.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            im_gradient = Image.fromarray((gradient[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            im_gradient_mask = Image.fromarray(((gradient[0][0]*(1.0-mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_edge = Image.fromarray((edge[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            im_edge_mask = Image.fromarray(((edge[0][0] * (1.0 - mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_mask = Image.fromarray((mask[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            # print(np.transpose((gradient[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).shape)
            # print(gradient[0].shape, outputB[0].shape)
            # im_gradient = Image.fromarray(np.transpose((gradient[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))

            ## 生成输出
            im_outputE = Image.fromarray((outputE[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))
            # im_outputR = Image.fromarray(
            #     np.transpose((outputR[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            im_outputG = Image.fromarray((outputG[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))

            im_input.save("output/mixture{}.jpg".format(num_step))
            im_gradient.save("output/gradient{}.jpg".format(num_step))
            im_gradient_mask.save("output/gradient_mask{}.jpg".format(num_step))
            im_edge.save("output/edge{}.jpg".format(num_step))
            im_edge_mask.save("output/edge_mask{}.jpg".format(num_step))
            im_mask.save("output/mask{}.jpg".format(num_step))
            im_outputE.save("output/outputE{}.jpg".format(num_step))
            im_outputG.save("output/outputG{}.jpg".format(num_step))
        if num_step % 30 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            # save the model
            save_checkpoint({
                'num_step': num_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, num_step,2 )

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, epoch,type = 0, save_path = 'output'):
    if type == 0:
        filename = "checkpoint_edge_gradient_{}.pth".format(epoch)
    if type == 1:
        filename = "checkpoint_remove_inpaint_{}.pth".format(epoch)
    if type == 2:
        filename = "checkpoint_restruct_guidance_{}.pth".format(epoch)
    torch.save(state, os.path.join(save_path, filename))

def validate(val_loader, model, L1Loss, test_writer, output_writers, epoch):
    # switch to evaluate mode
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (mixture, background, reflection, gradient) in enumerate(val_loader):
        input = [j.cuda() for j in mixture]
        input = Variable(input[0], requires_grad=True)

        background = [j.cuda() for j in background]
        background = Variable(background[0], requires_grad=False)

        reflection = [j.cuda() for j in reflection]
        reflection = Variable(reflection[0], requires_grad=False)

        gradient = [j.cuda() for j in gradient]
        gradient = Variable(gradient[0], requires_grad=False)

        output = model(input)

        outputB = output[0]

        loss = L1Loss(outputB, background)
        losses.update(loss.item(), background.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        ###show the estimated results on tensorboard
        test_writer.add_scalar('evaluation_loss', loss.data[0], epoch)
        if i < len(output_writers):  # log first output of first batches
            output_writers[i].add_image('TGroundTruth', input[0].data.cpu(), epoch)
            output_writers[i].add_image('ToutputB', outputB[0].data.cpu(), epoch)
        # output_writers[i].add_image('targetB', background[0].data.cpu(), epoch)


