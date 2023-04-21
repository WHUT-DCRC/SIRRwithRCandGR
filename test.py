# from gradient_model import ReflectionNetwork
# from utils import PSNR, MatrixToImage
# from PIL import Imageimport argparse, os
import os

import numpy as np
import torch
from torch.utils import data
# from utils import PSNR, MatrixToImage
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.nn as nn
from loss.SSIMLoss import SSIMLoss
from loss.SILoss import SILoss
from loss.MMDLoss import MMDLoss
from loss.PSNRLoss import PSNRLoss

deviceId = 3
batch_size = 1
transform = transforms.Compose([transforms.ToTensor()])
grayscale_transform = transforms.Compose([
   transforms.Grayscale(1), #这一句就是转为单通道灰度图像
   transforms.ToTensor(),
])

# 本地
# mixture_set = ImageFolder(root=r"datasets/train_set/mixture",
#                           transform=transform)
# background_set = ImageFolder(root=r"datasets/train_set/background",
#                              transform=transform)
# gradient_set = ImageFolder(root=r"datasets/train_set/gradient",
#                            transform=grayscale_transform)
# edge_set = ImageFolder(root=r"datasets/train_set/edge",
#                            transform=grayscale_transform)
# mask_set = ImageFolder(root=r"datasets/train_set/mask",
#                            transform=grayscale_transform)

# 服务器数据集位置
mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/test_set/mixture",
                          transform=transform)
reflection_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/test_set/reflection",
                             transform=transform)
background_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/test_set/background",
                             transform=transform)
gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/test_set/gradient",
                           transform=grayscale_transform)
edge_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/test_set/edge",
                       transform=grayscale_transform)
mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/test_set/mask",
                       transform=grayscale_transform)

# 服务器官方数据集位置
mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/mixture",
                          transform=transform)
reflection_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/reflection",
                             transform=transform)
background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/background",
                             transform=transform)
gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/gradient",
                             transform=grayscale_transform)
mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/mask",
                       transform=grayscale_transform)

####load the loss functions
ssimLoss = SSIMLoss(deviceId).cuda(deviceId)
L1Loss = nn.L1Loss().cuda(deviceId)
siLoss = SILoss(deviceId).cuda(deviceId)
mmdLoss = MMDLoss(deviceId).cuda(deviceId)
PSNRLoss = PSNRLoss

# 输入mixture,mask
# 输出edge,gradient
def test_edge_gradient(model):
    # 加载模型
    print("load test_edge_gradient model")
    ckpt_data = torch.load(
        r'ckpt/checkpoint_800.pth',map_location='cuda:{}'.format(deviceId))
    model.load_state_dict(ckpt_data['state_dict'])
    model.test()

    mixture_loader = data.DataLoader(mixture_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    edge_loader = data.DataLoader(edge_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                  shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                  shuffle=True)
    mixtureit = iter(mixture_loader)
    gradientit = iter(gradient_loader)
    edgeit = iter(edge_loader)
    maskit = iter(mask_loader)

    NUM_TOTAL_STEP = int(len(mixture_set)/batch_size)
    num_step = 0
    while num_step < NUM_TOTAL_STEP:
    # while num_step < 1:
        num_step += 1
        print(num_step, '/', NUM_TOTAL_STEP)

        mixture, _ = next(mixtureit)
        gradient, _ = next(gradientit)
        edge, _ = next(edgeit)
        mask, _ = next(maskit)

        inputs = mixture.float().cuda(deviceId)
        gradients = gradient.float().cuda(deviceId)
        edges = edge.float().cuda(deviceId)
        masks = mask.float().cuda(deviceId)

        output = model.forward(inputs, masks)

        outputE = output[0]
        outputG = output[1]

        print("save imgs{}".format(num_step))
        ## 原始输入
        im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        im_gradient = Image.fromarray((gradient[0][0] * 255.0).cpu().numpy().astype(np.uint8))
        im_gradient_mask = Image.fromarray(((gradient[0][0] * (1.0 - mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
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

# 输入mixture,mask,edge,gradient
# 输出remove_img,inpaint_img
def test_remove_inpaint(model):
    print("load pretrained model")
    model = model_load(model,model_type=1,ckpt = '120')
    model.train()
    num_step = 0



    mixture_loader = data.DataLoader(mixture_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    background_loader = data.DataLoader(background_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    # reflection_loader = data.DataLoader(reflection_set, batch_size=batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    edge_loader = data.DataLoader(edge_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)
    mixtureit = iter(mixture_loader)
    backgroundit = iter(background_loader)
    # reflectionit = iter(reflection_loader)
    gradientit = iter(gradient_loader)
    edgeit = iter(edge_loader)
    maskit = iter(mask_loader)
    num_stepit = 0
    print(len(mixture_set),mixture_set)
    NUM_TOTAL_STEP = int(len(mixture_set)/batch_size)
    while num_step < NUM_TOTAL_STEP:
    # while num_step < 1:
        num_step += 1
        mixture, _ = next(mixtureit)
        background, _ = next(backgroundit)
        # reflection, _ = next(reflectionit)
        gradient, _ = next(gradientit)
        edge, _ = next(edgeit)
        mask, _ = next(maskit)
        # print("num now:",num)

        inputs = mixture.float().cuda(deviceId)
        backgrounds = background.float().cuda(deviceId)
        # reflections = reflection.float().cuda(deviceId)
        gradients = gradient.float().cuda(deviceId)
        edges = edge.float().cuda(deviceId)
        masks = mask.float().cuda(deviceId)

        output = model.forward(inputs, gradients, edges, masks)

        outputR = output[0]
        outputI = output[1]
        img_masked = output[2]
        removeloss = 0.5 * ssimLoss(outputR * (1 - masks), backgrounds * (1 - masks)) \
                     + 0.5 * L1Loss(outputR * (1 - masks), backgrounds * (1 - masks)) \
            # + 0.5 * mmdLoss(outputE,edges)  ##loss functios for background

        inpaintloss = 0.5 * ssimLoss(outputI, backgrounds) \
                      + 0.5 * L1Loss(outputI, backgrounds) \

        print("removeloss{}\ninpaintloss{}\n".format(removeloss,inpaintloss))
        print("save imgs{}".format(num_step))

        # 输入混合图,真实背景,遮罩
        im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        im_background = Image.fromarray(
            np.transpose((backgrounds[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        im_mask = Image.fromarray((mask[0][0] * 255.0).cpu().numpy().astype(np.uint8))

        ## 生成输出
        im_outputR = Image.fromarray(
            np.transpose((outputR[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        im_outputI = Image.fromarray(
            np.transpose((outputI[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        im_masked = Image.fromarray(
            np.transpose((img_masked[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        # 创建文件夹
        save_path = "output/{}".format(num_step)
        if not os.path.exists(save_path):
            print(save_path)
            os.makedirs(save_path)
        # im_input.save("output/mixture{}.jpg".format(num_step))
        im_background.save("output/{}/background{}.jpg".format(num_step,num_step))
        # im_mask.save("output/{}/mask{}.jpg".format(num_step,num_step))
        # im_outputR.save("output/{}/outputR{}.jpg".format(num_step,num_step))
        im_outputI.save("output/{}/outputI{}.jpg".format(num_step,num_step))
        im_masked.save("output/{}/im_masked{}.jpg".format(num_step,num_step))

def test_restruct_guidance(model):

    # model = model_load(model,model_type=2,ckpt = '330')
    model.train()

    num_step = 0


    mixture_loader = data.DataLoader(mixture_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    background_loader = data.DataLoader(background_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    # reflection_loader = data.DataLoader(reflection_set, batch_size=batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)
    mixtureit = iter(mixture_loader)
    backgroundit = iter(background_loader)
    # reflectionit = iter(reflection_loader)
    gradientit = iter(gradient_loader)
    # edgeit = iter(edge_loader)
    maskit = iter(mask_loader)

    num_stepit = 0
    print(len(mixture_set),mixture_set)
    NUM_TOTAL_STEP = int(len(mixture_set)/batch_size)
    while num_step < NUM_TOTAL_STEP:
    # while num_step < 1:
        num_step += 1
        print(num_step, '/', NUM_TOTAL_STEP)

        mixture, _ = next(mixtureit)
        background, _ = next(backgroundit)
        # reflection, _ = next(reflectionit)
        gradient, _ = next(gradientit)
        # edge, _ = next(edgeit)
        mask, _ = next(maskit)

        inputs = mixture.float().cuda(deviceId)
        backgrounds = background.float().cuda(deviceId)
        # reflections = reflection.float().cuda(args.device)
        gradients = gradient.float().cuda(deviceId)
        # edges = edge.float().cuda(args.device)
        masks = mask.float().cuda(deviceId)

        output = model.forward(inputs,masks)

        outputR = output[0]
        img_masked = output[1]
        outputG = output[2]

        # print(outputB.shape,outputB[0].type,backgrounds.shape,backgrounds[0].type)
        # bl = 0.8 * ssimLoss(outputB, 0.8 * backgrounds) + 0.1*L1Loss(outputB, 0.8 * backgrounds) + 0.5 * mmdLoss(outputB,
        #                                                                                                      0.8 * backgrounds)  ##loss functios for background
        #
        #
        # gl = siLoss(outputG, gradients)  ##loss functios for gradient
        # rl = ssimLoss(outputR, inputs - backgrounds) + 0.5 * mmdLoss(outputR, inputs - backgrounds)
        # totalLoss = bl + rl + gl  ##based on my experiments different weighting coefficients on rl may generate different results, if you make the coefficients smaller, maybe the results can be better
        # # totalLoss = bl+gl
        ssim = ssimLoss(backgrounds, outputR)
        psnr = PSNRLoss(backgrounds, outputR)
        print("ssimLoss:", ssim, "\npsnrLoss:", psnr)

        print("save img")
        # 创建文件夹
        save_path = "output/test/{}".format(num_step)
        if not os.path.exists(save_path):
            print(save_path)
            os.makedirs(save_path)
        ## 生成输出
        im_background = Image.fromarray(
            np.transpose((backgrounds[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        im_outputR = Image.fromarray(
            np.transpose((outputR[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        img_masked = Image.fromarray(
            np.transpose((img_masked[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        img_outputG = Image.fromarray((outputG[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))



        im_background.save("output/test/{}/im_background{}.jpg".format(num_step,num_step))
        img_masked.save("output/test/{}/img_masked{}.jpg".format(num_step,num_step))
        # im_gradient.save("output/gradient{}.jpg".format(num_step))
        # im_gradient_mask.save("output/gradient_mask{}.jpg".format(num_step))
        # im_edge.save("output/edge{}.jpg".format(num_step))
        # im_edge_mask.save("output/edge_mask{}.jpg".format(num_step))
        # im_mask.save("output/mask{}.jpg".format(num_step))
        im_outputR.save("output/test/{}/im_outputR{}.jpg".format(num_step,num_step))
        img_outputG.save("output/test/{}/img_outputG{}.jpg".format(num_step,num_step))

def blind_test(model):
    global n_iter
    # switch to train mode

    print("load pretrained model")
    ckpt_data = torch.load(
        r'C:\Users\70432\Desktop\CV_Project\MyWork\Gradient\ckpt\checkpoint_800.pth',map_location='cuda:{}'.format(deviceId))

    model.load_state_dict(ckpt_data['state_dict'])
    model.train()
    # end = time.time()
    num_step = 0
    mixture_set = ImageFolder(root=r"C:\Users\70432\Desktop\CV_Project\MyWork\Gradient\datasets\blind_set",
                              transform=transform)
    # background_set = ImageFolder(
    #     root=r"C:\Users\70432\Desktop\CV_Project\MyWork\Gradient\datasets\test_set\background",
    #     transform=transform)
    # reflection_set = ImageFolder(
    #     root=r"C:\Users\70432\Desktop\CV_Project\MyWork\Gradient\datasets\test_set\reflection",
    #     transform=transform)
    # gradient_set = ImageFolder(root=r"C:\Users\70432\Desktop\CV_Project\MyWork\Gradient\datasets\test_set\gradient",
    #                            transform=grayscale_transform)

    mixture_loader = data.DataLoader(mixture_set, batch_size=batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    # background_loader = data.DataLoader(background_set, batch_size=batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    # reflection_loader = data.DataLoader(reflection_set, batch_size=batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    # gradient_loader = data.DataLoader(gradient_set, batch_size=batch_size, num_workers=0, pin_memory=True,
    #                                   shuffle=False)
    mixtureit = iter(mixture_loader)
    # backgroundit = iter(background_loader)
    # reflectionit = iter(reflection_loader)
    # gradientit = iter(gradient_loader)
    num_stepit = 0
    print(len(mixture_set),mixture_set)
    NUM_TOTAL_STEP = int(len(mixture_set)/batch_size)
    while num_step < NUM_TOTAL_STEP:
        num_step += 1
        print(num_step, '/', NUM_TOTAL_STEP)
        mixture, _ = next(mixtureit)

        inputs = mixture.float().cuda(deviceId)

        output = model.forward(inputs)
        outputB = output[0]
        outputR = output[1]
        outputG = output[2]

        print("save img")
        im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        print( outputB.shape)
        im_outputB = Image.fromarray(
            np.transpose((outputB[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        # im_outputR = Image.fromarray(
        #     np.transpose((outputR[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        # im_outputG = Image.fromarray((outputG[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))

        im_input.save("output/mixture{}.jpg".format(num_step))
        im_outputB.save("output/outputB{}.jpg".format(num_step))
        # im_outputR.save("./outputR{}.jpg".format(num_step))
        # im_outputG.save("./outputG{}.jpg".format(num_step))


def test_restruct_guidance1( gen_model,gen_optimizer,dis_model,dis_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, args):
    gen_model.train()
    dis_model.train()
    NUM_TOTAL_STEP = 1001
    print("args.device",args.device)

    args.batch_size = 1
    mixture_loader = data.DataLoader(mixture_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    background_loader = data.DataLoader(background_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)

    mask_loader = data.DataLoader(mask_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)
    mixtureit = iter(mixture_loader)
    backgroundit = iter(background_loader)
    # reflectionit = iter(reflection_loader)
    gradientit = iter(gradient_loader)
    # edgeit = iter(edge_loader)
    maskit = iter(mask_loader)
    iter_step = 0
    # totalLoss_count = 0
    gen_count = 0

    while iter_step < int(len(mixture_set) / args.batch_size):
        iter_step += 1
        # print(iter_step, '/', int(len(mixture_set) / args.batch_size))
        mixture, _ = next(mixtureit)
        background, _ = next(backgroundit)
        # reflection, _ = next(reflectionit)
        gradient, _ = next(gradientit)
        # edge, _ = next(edgeit)
        mask, _ = next(maskit)

        inputs = mixture.float().cuda(args.device)
        backgrounds = background.float().cuda(args.device)
        # reflections = reflection.float().cuda(args.device)
        gradients = gradient.float().cuda(args.device)
        # edges = edge.float().cuda(args.device)
        masks = mask.float().cuda(args.device)

        # print(inputs.shape,masks.shape)
        output = gen_model.forward(inputs, masks)

        outputR = output[0]
        img_masked = output[1]
        outputG = output[2]
        gen_optimizer.zero_grad()
        # print(outputG.shape,gradients.shape)
        gradient_loss = 0.5 * ssimLoss(outputG, gradients) \
                        + 0.5 * L1Loss(outputG, gradients)
        remove_loss = 0.5 * ssimLoss(outputR, backgrounds) \
                      + 0.5 * L1Loss(outputR, backgrounds) \
                      + 0.5 * mmdLoss(outputR, backgrounds)  ##loss functios for background
        ssim_loss = ssimLoss(outputR, backgrounds)

        gen_loss = gradient_loss + remove_loss
        gen_count += gen_loss
        gen_loss.backward()  ##backward the loss fucntions
        gen_optimizer.step()

        # print("gen_loss:", gen_loss)
        print("save img")
        ## 原始输入
        # im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
        im_background = Image.fromarray(
            np.transpose((backgrounds[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))

        ## 生成输出
        im_outputR = Image.fromarray(
            np.transpose((outputR[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))

        img_masked = Image.fromarray(
            np.transpose((img_masked[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        img_outputG = Image.fromarray((outputG[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))

        ## 存图
        im_background.save("output/train/im_background{}.jpg".format(iter_step))
        img_masked.save("output/train/img_masked{}.jpg".format(iter_step))
        im_outputR.save("output/train/im_outputR{}.jpg".format(iter_step))
        img_outputG.save("output/train/img_outputG{}.jpg".format(iter_step))





# 加载三种模型中指定轮次的ckpt
def model_load(model,model_type = 0,ckpt = '60'):
    print("load pretrained model")
    if model_type == 0:
        ckpt_data = torch.load(
            r'ckpt/checkpoint_gradient_edge_{}.pth'.format(ckpt))
    elif model_type == 1:
        ckpt_data = torch.load(
            r'ckpt/checkpoint_gen_remove_inpaint_{}.pth'.format(ckpt))
    else:
        ckpt_data = torch.load(
            r'ckpt/checkpoint_gen_restruct_guidance1_{}.pth'.format(ckpt))
    model.load_state_dict(ckpt_data['state_dict'])

    return model