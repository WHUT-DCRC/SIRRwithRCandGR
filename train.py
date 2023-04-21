import os
import numpy as np
import torch
from torch.utils import data
from torch.autograd import Variable
# from utils import PSNR, MatrixToImage
import torchvision.transforms as transforms
# import datasets
import time
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
from loss.GANloss import   AdversarialLoss
from random import randint
from loss.PSNRLoss import PSNRLoss


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


# 服务器数据集位置
# mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/mixture",
#                           transform=transform)
# reflection_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/reflection",
#                              transform=transform)
# background_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/background",
#                              transform=transform)
# gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/gradient",
#                            transform=grayscale_transform)
# edge_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/edge",
#                        transform=grayscale_transform)
# mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/lab_Gradient/datasets/train_set/mask",
#                        transform=grayscale_transform)


# 服务器官方数据集位置
# mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/mixture",
#                           transform=transform)
# reflection_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/reflection",
#                              transform=transform)
# background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/background",
#                              transform=transform)
# gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/gradient",
#                              transform=grayscale_transform)
# mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/mask",
#                        transform=grayscale_transform)

# 服务器官方测试数据集位置
mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_test_set/mixture_test",
                          transform=transform)
reflection_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/reflection",
                             transform=transform)
background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_test_set/background_test",
                             transform=transform)
gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_test_set/gradient_test",
                             transform=grayscale_transform)
mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_test_set/mask",
                       transform=grayscale_transform)

# 服务器论文数据集位置
# mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/lunwen_set/I",
#                           transform=transform)
# reflection_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/sir2_train_set/reflection",
#                              transform=transform)
# background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/lunwen_set/T",
#                              transform=transform)
# gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/lunwen_set/G",
#                              transform=grayscale_transform)
# mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/lunwen_set/M",
#                        transform=grayscale_transform)

# 服务器唐卡数据集（提出）位置
# mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/TK_set/I2000",
#                           transform=transform)
#
# background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/TK_set/T2000",
#                              transform=transform)
# gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/TK_set/G2000",
#                              transform=grayscale_transform)
mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/TK_set/M2000",
                       transform=grayscale_transform)

# background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/lunwen_set/TK/tou_T",
#                           transform=transform)
# mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/lunwen_set/TK/tou_I",
#                           transform=transform)
# mask_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/lunwen_set/TK/tou_M",
#                           transform=grayscale_transform)

# real数据集
# train
background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/CEL/train/T",
                          transform=transform)
mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/CEL/train/I",
                          transform=transform)
gradient_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/CEL/train/G",
                             transform=grayscale_transform)
# test
background_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/CEL/test/T",
                          transform=transform)
mixture_set = ImageFolder(root=r"/home/zhanglf/MyWork/Restruct_Removal/datasets/CEL/test/I",
                          transform=transform)


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
            # edges = edge.float().cuda(args.device)
            masks = mask.float().cuda(args.device)

            output = model.forward(inputs,masks)

            # outputE = output[0]
            outputG = output
            # el = ssimLoss(outputE,  edges) \
            #      + 0.5*L1Loss(outputE, edges) \
                 # + 0.5 * mmdLoss(outputE,edges)  ##loss functios for background

            SsimLoss = ssimLoss(outputG, gradients)
            gl = SsimLoss\
                + 0.5*L1Loss(outputG, gradients)\
                + 0.5 * mmdLoss(outputG,gradients)##loss functios for gradient

            # totalLoss = el+gl
            totalLoss = gl

            totalLoss_count += totalLoss
            # el_count += el
            gl_count += gl


            optimizer.zero_grad()
            totalLoss.backward()  ##backward the loss fucntions
            optimizer.step()
        print("totalLoss = ", totalLoss)
        print("ssimLoss = ", SsimLoss)
        if num_step % 10 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
        #     print("totalLoss:", totalLoss_count/500, "\nloss_edge:", el_count/500, "\nloss_gradient:", gl_count/500)

            ## 原始输入
            im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            im_gradient = Image.fromarray((gradient[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            im_gradient_mask = Image.fromarray(((gradient[0][0]*(1.0-mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_edge = Image.fromarray((edge[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            im_edge_mask = Image.fromarray(((edge[0][0] * (1.0 - mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            im_mask = Image.fromarray((mask[0][0] * 255.0).cpu().numpy().astype(np.uint8))

            ## 生成输出
            # im_outputE = Image.fromarray((outputE[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))
            im_outputG = Image.fromarray((outputG[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))

            ## 保存图片
            # im_input.save("output/train/mixture{}.jpg".format(num_step))
            im_gradient.save("output/train/gradient{}.jpg".format(num_step))
            im_gradient_mask.save("output/train/gradient_mask{}.jpg".format(num_step))
            # im_edge.save("output/train/edge{}.jpg".format(num_step))
            # im_edge_mask.save("output/train/edge_mask{}.jpg".format(num_step))
            # im_mask.save("output/train/mask{}.jpg".format(num_step))
            # im_outputE.save("output/train/outputE{}.jpg".format(num_step))
            im_outputG.save("output/train/outputG{}.jpg".format(num_step))
        if num_step % 30 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            # save the model
            save_checkpoint({
                'num_step': num_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, num_step,0,'gen' )

def remove_inpaint_train( gen_model,gen_optimizer,dis_model,dis_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, args):
    gen_model.train()
    dis_model.train()
    NUM_TOTAL_STEP = 1001

    mixture_loader = data.DataLoader(mixture_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    background_loader = data.DataLoader(background_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    # reflection_loader = data.DataLoader(reflection_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    edge_loader = data.DataLoader(edge_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)

    for num_step in tqdm(range(NUM_TOTAL_STEP)):
        mixtureit = iter(mixture_loader)
        backgroundit = iter(background_loader)
        # reflectionit = iter(reflection_loader)
        gradientit = iter(gradient_loader)
        edgeit = iter(edge_loader)
        maskit = iter(mask_loader)
        iter_step = 0
        totalLoss_count = 0
        gen_count = 0
        dis_count = 0
        rand_dis = randint(2,7)
        print("rand_dis:",rand_dis)
        while iter_step < int(len(mixture_set)/args.batch_size):
            iter_step += 1
            # print(iter_step)
            mixture, _ = next(mixtureit)
            background, _ = next(backgroundit)
            # reflection, _ = next(reflectionit)
            gradient, _ = next(gradientit)
            edge, _ = next(edgeit)
            mask, _ = next(maskit)

            inputs = mixture.float().cuda(args.device)
            backgrounds = background.float().cuda(args.device)
            # reflections = reflection.float().cuda(args.device)
            gradients = gradient.float().cuda(args.device)
            edges = edge.float().cuda(args.device)
            masks = mask.float().cuda(args.device)


            output = gen_model.forward(inputs,gradients,edges,masks)

            outputR = output[0]
            outputI = output[1]
            img_masked = output[2]
            # if iter_step%3 == 0 and iter_step != 0:
            if iter_step % rand_dis == 0 :
                dis_optimizer.zero_grad()
                dis_real, _ = dis_model(backgrounds)  # in: [rgb(3)]
                dis_fake, _ = dis_model(outputI)
                adversarial_loss = AdversarialLoss(type="nsgan")
                dis_real_loss = adversarial_loss(dis_real, True, True)
                dis_fake_loss = adversarial_loss(dis_fake, False, True)
                dis_loss = (dis_real_loss + dis_fake_loss) / 2

                print("dis_loss",dis_loss)

                dis_count += dis_loss
                dis_loss.backward()  ##backward the loss fucntions
                dis_optimizer.step()
            else:
                gen_optimizer.zero_grad()
                # print(outputR.shape,outputI.shape)
                removeloss = 0.5 * ssimLoss(outputR * (1 - masks), backgrounds * (1 - masks)) \
                             + 0.5 * L1Loss(outputR * (1 - masks), backgrounds * (1 - masks)) \
                    # + 0.5 * mmdLoss(outputE,edges)  ##loss functios for background

                inpaintloss = 0.5 * ssimLoss(outputI , backgrounds ) \
                              + 0.5 * L1Loss(outputI , backgrounds ) \
                    # + 0.5 * mmdLoss(outputG,gradients)##loss functios for gradient

                gen_loss = removeloss + inpaintloss
                print("gen_loss",gen_loss)
                gen_count += gen_loss
                gen_loss.backward()  ##backward the loss fucntions
                gen_optimizer.step()

        if num_step % 10 == 0 and num_step != 0:
        # if num_step % 1 == 0 :

            print("\ngen_loss:", (gen_count*(rand_dis))/(500*(rand_dis-1)), "\ndis_loss:", dis_count*rand_dis/(500))

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
            im_outputR = Image.fromarray(
                np.transpose((outputR[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            im_outputI = Image.fromarray(
                np.transpose((outputI[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            im_masked = Image.fromarray(
                np.transpose((img_masked[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))

            im_input.save("output/train/mixture{}.jpg".format(num_step))
            im_background.save("output/train/background{}.jpg".format(num_step))
            # im_gradient.save("output/gradient{}.jpg".format(num_step))
            # im_gradient_mask.save("output/gradient_mask{}.jpg".format(num_step))
            # im_edge.save("output/edge{}.jpg".format(num_step))
            # im_edge_mask.save("output/edge_mask{}.jpg".format(num_step))
            im_mask.save("output/train/mask{}.jpg".format(num_step))
            im_outputR.save("output/train/outputR{}.jpg".format(num_step))
            im_outputI.save("output/train/outputI{}.jpg".format(num_step))
            im_masked.save("output/train/im_masked{}.jpg".format(num_step))
        if num_step % 30 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            # save the model
            save_checkpoint({
                'num_step': num_step,
                'state_dict': gen_model.state_dict(),
                'optimizer': gen_optimizer.state_dict(),
            }, num_step,1,'gen' )
            save_checkpoint({
                'num_step': num_step,
                'state_dict': dis_model.state_dict(),
                'optimizer': dis_optimizer.state_dict(),
            }, num_step, 1,'dis')

def restruct_guidance_train( gen_model,gen_optimizer,dis_model,dis_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, args):
    gen_model.train()
    dis_model.train()
    NUM_TOTAL_STEP = 1001


    # reflection_set = ImageFolder(
    #     root=r"datasets/train_set/reflection",
    #     transform=transform)

    mixture_loader = data.DataLoader(mixture_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    background_loader = data.DataLoader(background_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    # reflection_loader = data.DataLoader(reflection_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
    #                                     shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    # edge_loader = data.DataLoader(edge_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
    #                                   shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)

    for num_step in tqdm(range(NUM_TOTAL_STEP)):
        mixtureit = iter(mixture_loader)
        backgroundit = iter(background_loader)
        # reflectionit = iter(reflection_loader)
        gradientit = iter(gradient_loader)
        # edgeit = iter(edge_loader)
        maskit = iter(mask_loader)
        iter_step = 0
        # totalLoss_count = 0
        gen_count = 0
        dis_count = 0
        rand_dis = randint(2, 7)
        print("\nrand_dis:", rand_dis)
        while iter_step < int(len(mixture_set)/args.batch_size):
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

            output = gen_model.forward(inputs,masks)

            outputR_masked = output[0]
            outputI = output[1]
            img_masked = output[2]
            outputG = output[3]


            # if iter_step % rand_dis == 0 :
            if False:
                dis_optimizer.zero_grad()
                # 梯度损失
                # dis_input_real = torch.cat((images, edges), dim=1)
                # dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
                dis_real_g,_ = dis_model(torch.cat((gradients, gradients,gradients),1))  # in: (grayscale(1) + edge(1))
                dis_fake_g,_ = dis_model(torch.cat((outputG, outputG,outputG),1))  # in: (grayscale(1) + edge(1))
                # 修复损失
                dis_real_i, _ = dis_model(backgrounds)
                dis_fake_i, _ = dis_model(outputI)
                # 去反射损失
                dis_real_r, _ = dis_model(backgrounds*(1-masks))
                dis_fake_r, _ = dis_model(outputR_masked)

                # 计算损失
                adversarial_loss = AdversarialLoss(type="nsgan")


                # 真假结构恢复损失
                dis_real_g_loss = adversarial_loss(dis_real_g, True, True)
                dis_fake_g_loss = adversarial_loss(dis_fake_g, False, True)
                # 真假修复损失
                dis_real_i_loss = adversarial_loss(dis_real_i, True, True)
                dis_fake_i_loss = adversarial_loss(dis_fake_i, False, True)
                # 真假去反射损失
                dis_real_r_loss = adversarial_loss(dis_real_r, True, True)
                dis_fake_r_loss = adversarial_loss(dis_fake_r, False, True)
                # 总损失
                dis_loss = (dis_real_i_loss + dis_fake_i_loss + dis_real_r_loss + dis_fake_r_loss + dis_real_g_loss + dis_fake_g_loss) / 6


                dis_count += dis_loss
                dis_loss.backward()  ##backward the loss fucntions
                dis_optimizer.step()
            else:
                gen_optimizer.zero_grad()
                gradient_loss  = 0.5 * ssimLoss(outputG,gradients) \
                            + 0.5 * L1Loss(outputG, gradients)
                remove_loss = 0.5 * ssimLoss(outputR_masked, backgrounds * (1 - masks)) \
                             + 0.5 * L1Loss(outputR_masked, backgrounds * (1 - masks)) \
                    # + 0.5 * mmdLoss(outputE,edges)  ##loss functios for background

                inpaint_loss = 0.5 * ssimLoss(outputI , backgrounds ) \
                              + 0.5 * L1Loss(outputI , backgrounds ) \
                    # + 0.5 * mmdLoss(outputG,gradients)##loss functios for gradient

                # 三种不同损失函数
                # gen_loss = gradient_loss + remove_loss + inpaint_loss
                # gen_loss = remove_loss + inpaint_loss
                gen_loss = inpaint_loss
                gen_count += gen_loss
                gen_loss.backward()  ##backward the loss fucntions
                gen_optimizer.step()
        print("num_step\ngen_loss:{}\n,gradient_loss:{}\n,remove_loss:{}\n,inpaint_loss:{}\n".format(gen_loss,
                                                                                                     gradient_loss,
                                                                                                     remove_loss,
                                                                                                     inpaint_loss))
        if num_step % 10 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            print("dis_loss:", dis_loss)
            print("gen_loss:", gen_loss)
            print("save img")
            ## 原始输入
            # im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            im_background = Image.fromarray(
                np.transpose((backgrounds[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
            # im_reflection = Image.fromarray(
            #     np.transpose(((inputs[0]-backgrounds[0]) * 52.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            # im_gradient = Image.fromarray((gradient[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            # im_gradient_mask = Image.fromarray(((gradient[0][0]*(1.0-mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            # im_edge = Image.fromarray((edge[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            # im_edge_mask = Image.fromarray(((edge[0][0] * (1.0 - mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
            # im_mask = Image.fromarray((mask[0][0] * 255.0).cpu().numpy().astype(np.uint8))
            # print(np.transpose((gradient[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).shape)
            # print(gradient[0].shape, outputB[0].shape)
            # im_gradient = Image.fromarray(np.transpose((gradient[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))

            ## 生成输出
            im_outputR = Image.fromarray(
                np.transpose((outputR_masked[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            im_outputI = Image.fromarray(
                np.transpose((outputI[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            img_masked = Image.fromarray(
                np.transpose((img_masked[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
            img_outputG = Image.fromarray((outputG[0][0] * 255.0).cpu().detach().numpy().astype(np.uint8))



            im_background.save("output/train/im_background{}.jpg".format(num_step))
            img_masked.save("output/train/img_masked{}.jpg".format(num_step))
            # im_gradient.save("output/gradient{}.jpg".format(num_step))
            # im_gradient_mask.save("output/gradient_mask{}.jpg".format(num_step))
            # im_edge.save("output/edge{}.jpg".format(num_step))
            # im_edge_mask.save("output/edge_mask{}.jpg".format(num_step))
            # im_mask.save("output/mask{}.jpg".format(num_step))
            im_outputR.save("output/train/im_outputR{}.jpg".format(num_step))
            im_outputI.save("output/train/im_outputI{}.jpg".format(num_step))
            img_outputG.save("output/train/img_outputG{}.jpg".format(num_step))

        if num_step % 30 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            # save the model
            save_checkpoint({
                'num_step': num_step,
                'state_dict': gen_model.state_dict(),
                'optimizer': gen_optimizer.state_dict(),
            }, num_step,2,'gen' )
            save_checkpoint({
                'num_step': num_step,
                'state_dict': dis_model.state_dict(),
                'optimizer': dis_optimizer.state_dict(),
            }, num_step, 2,'dis')

def restruct_guidance1_train( gen_model,gen_optimizer,dis_model,dis_optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, args):
    gen_model.train()
    dis_model.train()
    NUM_TOTAL_STEP = 1001

    mixture_loader = data.DataLoader(mixture_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False)
    background_loader = data.DataLoader(background_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                        shuffle=False)
    gradient_loader = data.DataLoader(gradient_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False)
    mask_loader = data.DataLoader(mask_set, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=True)
    img_id = 0
    for num_step in tqdm(range(NUM_TOTAL_STEP)):
        mixtureit = iter(mixture_loader)
        backgroundit = iter(background_loader)
        gradientit = iter(gradient_loader)
        maskit = iter(mask_loader)
        iter_step = 0
        gen_count = 0
        rand_dis = randint(2, 7)
        print("\nrand_dis:", rand_dis)

        while iter_step < int(len(mixture_set)/args.batch_size):
            mixture, _ = next(mixtureit)
            background, _ = next(backgroundit)
            gradient, _ = next(gradientit)
            mask, _ = next(maskit)

            inputs = mixture.float().cuda(args.device)
            backgrounds = background.float().cuda(args.device)
            gradients = gradient.float().cuda(args.device)
            masks = mask.float().cuda(args.device)

            # print(inputs.shape,masks.shape)
            output = gen_model.forward(inputs,masks)

            outputR = output[0]
            img_masked = output[1]
            outputG = output[2]

            # ssim = 1 - ssimLoss(backgrounds, outputR)
            # psnr = PSNRLoss(backgrounds, outputR)
            # print("ssimLoss:", ssim, "\npsnrLoss:", psnr)
            # print("img_id", img_id)

            # 测试，只出一遍图
            Test = False
            if Test:
                while img_id < 11:

                    im_background = Image.fromarray(
                        np.transpose((backgrounds[img_id] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
                    ## 生成输出
                    im_outputR = Image.fromarray(
                        np.transpose((outputR[img_id] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))

                    img_outputG = Image.fromarray(
                        (outputG[img_id][0] * 255.0).cpu().detach().numpy().astype(np.uint8))
                    im_background.save("output/train/im_background{}.jpg".format(num_step * 100+ iter_step*10 + img_id))
                    im_outputR.save("output/train/im_outputR{}.jpg".format(num_step * 100 + iter_step * 10 + img_id))
                    img_outputG.save("output/train/img_outputG{}.jpg".format(num_step * 100+ iter_step*10 + img_id))
                    img_id = img_id + 1
                img_id = 0

            else:
                # if iter_step % rand_dis == 0:
                if False:
                    dis_optimizer.zero_grad()
                    gen_optimizer.zero_grad()
                    # 梯度损失
                    # dis_input_real = torch.cat((images, edges), dim=1)
                    # dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
                    dis_real_g, _ = dis_model(
                        torch.cat((gradients, gradients, gradients), 1))  # in: (grayscale(1) + edge(1))
                    dis_fake_g, _ = dis_model(torch.cat((outputG, outputG, outputG), 1))  # in: (grayscale(1) + edge(1))
                    # 去反射损失
                    dis_real_r, _ = dis_model(backgrounds)
                    dis_fake_r, _ = dis_model(outputR)

                    # 计算损失
                    adversarial_loss = AdversarialLoss(type="nsgan")

                    # 真假结构恢复损失
                    dis_real_g_loss = adversarial_loss(dis_real_g, True, True)
                    dis_fake_g_loss = adversarial_loss(dis_fake_g, False, True)

                    # 真假去反射损失
                    dis_real_r_loss = adversarial_loss(dis_real_r, True, True)
                    dis_fake_r_loss = adversarial_loss(dis_fake_r, False, True)
                    # 总损失
                    dis_loss = (dis_real_r_loss + dis_fake_r_loss + dis_real_g_loss + dis_fake_g_loss) / 4
                    print("dis_loss:", dis_loss)

                    # dis_count += dis_loss
                    dis_loss.backward()  ##backward the loss fucntions
                    dis_optimizer.step()
                    gen_optimizer.step()
                else:
                    gen_optimizer.zero_grad()
                    # print(outputG.shape,gradients.shape)
                    gradient_loss = 0.5 * ssimLoss(outputG, gradients) \
                                    + 0.5 * L1Loss(outputG, gradients)
                    remove_loss = 0.5 * ssimLoss(outputR, backgrounds) \
                                  + 0.5 * L1Loss(outputR, backgrounds) \
                                  + 0.5 * mmdLoss(outputR, backgrounds)  ##loss functios for background
                    # ssim_loss = ssimLoss(outputR, backgrounds)
                    ssim = 1 - ssimLoss(backgrounds, outputR)
                    psnr = PSNRLoss(backgrounds, outputR)
                    print("ssimLoss:", ssim, "\npsnrLoss:", psnr,"\n")


                    inpaint_loss = 0.5 * ssimLoss(outputR* (1 - masks), backgrounds * (1 - masks)) \
                                  + 0.5 * L1Loss(outputR* (1 - masks), backgrounds * (1 - masks)) \

                    # 三种不同损失函数
                    gen_loss = gradient_loss + remove_loss
                    # gen_loss = remove_loss + gradient_loss
                    # gen_loss = 2*remove_loss
                    print("gradient_loss:",gradient_loss,"\nremove_loss:",remove_loss,"\ninpaint_loss:",inpaint_loss)


                    # gen_loss = gradient_loss + 2*remove_loss

                    gen_loss.backward()  ##backward the loss fucntions
                    # gen_optimizer.step()
                # if num_step % 500 == 0 and num_step != 0:
                if num_step  == 0 :
                    print("save img")
                    ## 原始输入
                    while img_id < 10:
                        print("img_id", img_id)
                        # im_input = Image.fromarray(np.transpose((inputs[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
                        im_background = Image.fromarray(
                            np.transpose((backgrounds[img_id] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
                        # im_reflection = Image.fromarray(
                        #     np.transpose(((inputs[0]-backgrounds[0]) * 52.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
                        # im_gradient = Image.fromarray((gradient[0][0] * 255.0).cpu().numpy().astype(np.uint8))
                        # im_gradient_mask = Image.fromarray(((gradient[0][0]*(1.0-mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
                        # im_edge = Image.fromarray((edge[0][0] * 255.0).cpu().numpy().astype(np.uint8))
                        # im_edge_mask = Image.fromarray(((edge[0][0] * (1.0 - mask[0][0])) * 255.0).cpu().numpy().astype(np.uint8))
                        im_mask = Image.fromarray((masks[img_id][0] * 255.0).cpu().numpy().astype(np.uint8))
                        # print(np.transpose((gradient[0] * 255.0).cpu().detach().numpy(), (1, 2, 0)).shape)
                        # print(gradient[0].shape, outputB[0].shape)
                        # im_gradient = Image.fromarray(np.transpose((gradient[0] * 255.0).cpu().numpy(), (1, 2, 0)).astype(np.uint8))

                        ## 生成输出
                        # print(outputR.shape, outputR.type)
                        im_outputR = Image.fromarray(
                            np.transpose((outputR[img_id] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))

                        # print(img_masked.shape,img_masked.type)
                        img_mask = Image.fromarray(
                            np.transpose((img_masked[img_id] * 255.0).cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
                        img_outputG = Image.fromarray(
                            (outputG[img_id][0] * 255.0).cpu().detach().numpy().astype(np.uint8))

                        # im_background.save("output/train/im_background{}.jpg".format(num_step * 100+ iter_step*10 + img_id))
                        img_mask.save("output/train/img_masked{}.jpg".format(str(num_step * 100+ iter_step*10 + img_id).zfill(5)))
                        # im_gradient.save("output/gradient{}.jpg".format(num_step))
                        # im_gradient_mask.save("output/gradient_mask{}.jpg".format(num_step))
                        # im_edge.save("output/edge{}.jpg".format(num_step))
                        # im_edge_mask.save("output/edge_mask{}.jpg".format(num_step))
                        im_mask.save("output/train/mask{}.jpg".format(str(num_step * 100+ iter_step*10 + img_id).zfill(5)))
                        im_outputR.save(
                            "output/train/im_outputR{}.jpg".format(str(num_step * 100+ iter_step*10 + img_id).zfill(5)))
                        img_outputG.save("output/train/img_outputG{}.jpg".format(str(num_step * 100+ iter_step*10 + img_id).zfill(5)))
                        img_id = img_id + 1
                    img_id = 0
                iter_step += 1
        # 存CKPT
        if num_step % 20 == 0 and num_step != 0:
        # if num_step % 1 == 0 :
            # save the model
            save_checkpoint({
                'num_step': num_step,
                'state_dict': gen_model.state_dict(),
                'optimizer': gen_optimizer.state_dict(),
            }, num_step,2,'gen' )
            save_checkpoint({
                'num_step': num_step,
                'state_dict': dis_model.state_dict(),
                'optimizer': dis_optimizer.state_dict(),
            }, num_step, 2,'dis')

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

def save_checkpoint(state, epoch,type = 0,GorD = 'gen', save_path = 'output'):
    if type == 0:
        filename = "checkpoint_edge_gradient_{}.pth".format(epoch)
    if type == 1:
        filename = "checkpoint_{}_remove_inpaint_{}.pth".format(GorD,epoch)
    if type == 2:
        filename = "checkpoint_{}_restruct_guidance_{}.pth".format(GorD,epoch)
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



