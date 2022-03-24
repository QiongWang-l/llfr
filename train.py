import uuid

from tqdm import tqdm
import network
import utils
import random
import argparse
import numpy as np
from torch.utils import data
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import sys
import time
import lowlight_model
import Myloss
from torchvision import transforms
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image, ImageStat
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import random
import shutil
import math
from PIL import ImageFile
# import tensorflow as tf
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nets.facenet import Facenet
from nets.facenet_training import LossHistory, triplet_loss
from utils.dataloader import FacenetDataset, dataset_collate

# ImageFile.LOAD_TRUNCATED_IMAGES = True

transf = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/cityscapes',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='lqf',
                        choices=['voc', 'cityscapes', 'lqf'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # FR facenet参数
    parser.add_argument("--input_shape", type=list, default=[160, 160, 3],
                        choices=[[160, 160, 3], [112, 112, 3]], help='输入图像大小')
    parser.add_argument("--annotation_path", type=str, default='cls_train_test.txt',
                        help='指向根目录下的cls_train.txt，读取人脸路径与标签')
    parser.add_argument("--num_val", type=int, default=0, help='验证集大小')
    parser.add_argument("--num_train", type=int, default=0, help='训练集大小')
    parser.add_argument("--fr_lr", type=float, default=0.0001, help="learning rate")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=50e3,
                        help="epoch number")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=160)

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='28333',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    parser.add_argument('--lowlight_lr', type=float, default=0.0001)
    parser.add_argument('--lowlight_weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--snapshots_folder', type=str, default="LLE/")
    return parser


def get_num_classes(annotation_path):
    """ 计算一共有多少个人，用于利用交叉熵辅助收敛 """
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


# 用于train
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'lqf':
        #   0.05用于验证，0.95用于训练
        val_split = 0.05
        with open(opts.annotation_path, "r") as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        opts.num_val = num_val
        opts.num_train = num_train

        num_classes = get_num_classes(opts.annotation_path)

        train_dst = FacenetDataset(opts.input_shape, lines[:num_train], num_train, num_classes)
        val_dst = FacenetDataset(opts.input_shape, lines[num_train:], num_val, num_classes)

    else:
        raise Exception("Invalid dataset!")
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    global lr_scheduler, optimizer
    opts = get_argparser().parse_args()
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    Cuda = True
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    # torch.cuda.manual_seed(opts.random_seed)  # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(opts.random_seed)  # 为所有GPU设置随机种子
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=True,collate_fn=dataset_collate)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=True,collate_fn=dataset_collate)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    epoch_step = opts.num_train // opts.batch_size
    epoch_step_val = opts.num_val // opts.val_batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    '''
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    '''

    # Set up model
    pretrained = False
    model_path = "checkpoints/facenet_mobilenet.pth"
    Freeze_Train = True

    num_classes = get_num_classes(opts.annotation_path)
    model = Facenet(backbone="mobilenet", num_classes=num_classes, pretrained=pretrained)
    if not pretrained:
        weights_init_(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    L_triplet = triplet_loss()
    loss_history = LossHistory("logs")

    utils.mkdir('checkpoints')
    # Restore
    '''
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
    '''

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    '''
    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return
    '''

    # lowlight
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DCE_net = lowlight_model.enhance_net_nopool().cuda()
    DCE_net.apply(weights_init_)
    L_color = Myloss.L_color()
    L_con = Myloss.L_con()
    L_TV = Myloss.L_TV()
    L_percept = Myloss.perception_loss()
    L_const = Myloss.PerceptualLoss()

    lowlight_optimizer = torch.optim.Adam(DCE_net.parameters(), lr=opts.lowlight_lr,
                                          weight_decay=opts.lowlight_weight_decay)

    pathL = "./datasets/data/Contrast/low/"  # negative samples
    pathDirL = os.listdir(pathL)
    pathGT = "./datasets/data/Contrast/GT/"  # positive samples
    pathDirGT = os.listdir(pathGT)
    picknumber = 2

    # facenet
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0
    total_LL_loss = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0
    val_total_LL_loss = 0

    Init_Epoch = 0
    Interval_Epoch = 20

    '''
    path = os.getcwd()
    contrast_path = os.path.join(path, 'normalize_pic')
    if not os.path.exists(contrast_path):
        os.makedirs(contrast_path)
    '''

    print("冻结训练开始************************************************")
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False
            # Set up optimizer
            optimizer = optim.Adam(model_train.parameters(), 0.001)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    for epoch in range(Init_Epoch, Interval_Epoch):
        # =====  Train  =====
        DCE_net.train()
        model_train.train()
        trans = transforms.ToTensor()
        print('Start Train')
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Interval_Epoch}', postfix=dict, mininterval=1) as pbar:
            for iteration, batch in enumerate(train_loader):
                if iteration >= epoch_step:
                    break
                images, labels = batch
                if len(images) <= 1:
                    continue

                # print(images.shape)
                # 对比学习
                sampleL = random.sample(pathDirL, picknumber)
                X_L = Image.open(pathL + sampleL[0])
                X_L = X_L.resize((160, 160), Image.ANTIALIAS)
                X_L = trans(X_L)
                Y_L = Image.open(pathL + sampleL[1])
                Y_L = Y_L.resize((160, 160), Image.ANTIALIAS)
                Y_L = trans(Y_L)
                L = torch.stack((X_L, Y_L), 0)
                L = L.cuda()

                sampleGT = random.sample(pathDirGT, picknumber)
                X_GT = Image.open(pathGT + sampleGT[0])
                X_GT = X_GT.resize((160, 160), Image.ANTIALIAS)
                X_GT = trans(X_GT)
                Y_GT = Image.open(pathGT + sampleGT[1])
                Y_GT = Y_GT.resize((160, 160), Image.ANTIALIAS)
                Y_GT = trans(Y_GT)
                GT = torch.stack((X_GT, Y_GT), 0)
                GT = GT.cuda()

                with torch.no_grad():
                    images = torch.from_numpy(images).to(device, dtype=torch.float32)
                    labels = torch.from_numpy(labels).to(device, dtype=torch.long)

                # img_lowlight = images.cuda()
                enhanced_image_1, enhanced_image, A = DCE_net(images)

                # print(images)  tensor
                # print(enhanced_image)
                '''
                with tf.Session() as sess:
                    print('images tensor:')
                    sess.run(tf.global_variables_initializer())
                    print(sess.run(images))

                with tf.Session():
                    print('enhanced_image tensor:')
                    print(enhanced_image.eval())
                '''

                bs3 = opts.batch_size * 3
                t, temp_loss_cont, temp_loss_cont2 = 0, 0, 0
                while t < bs3:
                    temp_images = enhanced_image[t:t + 2]
                    t += 2
                    temp_loss_cont += torch.mean(max(L_con(temp_images, GT) - L_con(temp_images, L) + 0.3,
                                                     L_con(L, temp_images) - L_con(L, temp_images)))
                    temp_loss_cont2 += torch.mean(max(L_const(temp_images, GT) - L_const(temp_images, L) + 0.04,
                                                      L_con(L, temp_images) - L_con(L, temp_images)))

                # print(GT.shape) (2,3,160,160)
                # print(L.shape)
                
                Loss_TV = 200 * L_TV(A)
                loss_col = 8 * torch.mean(L_color(enhanced_image))
                loss_percent = torch.mean(L_percept(images, enhanced_image))
                loss_cont = 20 * temp_loss_cont / bs3
                loss_cont2 = 10 * temp_loss_cont2 / bs3

                lowlight_loss = Loss_TV + loss_col + loss_cont + loss_cont2 + loss_percent

                enhanced_image = enhanced_image.detach()
                # enhanced_image = enhanced_image.to(device, dtype=torch.float32)

                lowlight_optimizer.zero_grad()
                optimizer.zero_grad()  # 初始化梯度值
                before_normalize, outputs1 = model.forward_feature(enhanced_image)
                outputs2 = model.forward_classifier(before_normalize)

                _triplet_loss = L_triplet(outputs1, opts.batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                fr_loss = _triplet_loss + _CE_loss

                total_loss = lowlight_loss + fr_loss
                fr_loss.backward()   # 反向求解梯度
                lowlight_loss.backward()
                torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), opts.grad_clip_norm)
                lowlight_optimizer.step()   # 更新参数
                optimizer.step()

                with torch.no_grad():
                    accuracy = torch.mean(
                        (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

                total_triple_loss += _triplet_loss.item()
                total_CE_loss += _CE_loss.item()
                total_accuracy += accuracy.item()
                total_LL_loss += lowlight_loss.item()

                total_Loss = total_triple_loss + total_CE_loss + total_LL_loss

                if iteration % 20 == 0:
                    pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                        'total_CE_loss': total_CE_loss / (iteration + 1),
                                        'lowlight_loss': total_LL_loss / (iteration + 1),
                                        'total_loss': total_Loss / (iteration + 1),
                                        'accuracy': total_accuracy / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(20)

        print('Finish Train')

        model_train.eval()
        print('Start Validation')
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Interval_Epoch}', postfix=dict,
                  mininterval=1) as pbar:
            for iteration, batch in enumerate(val_loader):
                if iteration >= epoch_step_val:
                    break
                images, labels = batch
                if len(images) <= 1:
                    continue

                # 对比学习
                sampleL = random.sample(pathDirL, picknumber)
                X_L = Image.open(pathL + sampleL[0])
                X_L = X_L.resize((160, 160), Image.ANTIALIAS)
                X_L = trans(X_L)
                Y_L = Image.open(pathL + sampleL[1])
                Y_L = Y_L.resize((160, 160), Image.ANTIALIAS)
                Y_L = trans(Y_L)
                L = torch.stack((X_L, Y_L), 0)
                L = L.cuda()

                sampleGT = random.sample(pathDirGT, picknumber)
                X_GT = Image.open(pathGT + sampleGT[0])
                X_GT = X_GT.resize((160, 160), Image.ANTIALIAS)
                X_GT = trans(X_GT)
                Y_GT = Image.open(pathGT + sampleGT[1])
                Y_GT = Y_GT.resize((160, 160), Image.ANTIALIAS)
                Y_GT = trans(Y_GT)
                GT = torch.stack((X_GT, Y_GT), 0)
                GT = GT.cuda()

                with torch.no_grad():
                    images = torch.from_numpy(images).to(device, dtype=torch.float32)
                    labels = torch.from_numpy(labels).to(device, dtype=torch.long)

                img_lowlight = images.cuda()
                enhanced_image_1, enhanced_image, A = DCE_net(images)

                bs3 = opts.val_batch_size * 3
                t, temp_loss_cont, temp_loss_cont2 = 0, 0, 0
                while t < bs3:
                    temp_images = enhanced_image[t:t + 2]
                    t += 2
                    temp_loss_cont += torch.mean(max(L_con(temp_images, GT) - L_con(temp_images, L) + 0.3,
                                                     L_con(L, temp_images) - L_con(L, temp_images)))
                    temp_loss_cont2 += torch.mean(max(L_const(temp_images, GT) - L_const(temp_images, L) + 0.04,
                                                      L_con(L, temp_images) - L_con(L, temp_images)))

                Loss_TV = 200 * L_TV(A)
                loss_col = 8 * torch.mean(L_color(enhanced_image))
                loss_percent = torch.mean(L_percept(images, enhanced_image))
                loss_cont = 20 * temp_loss_cont / bs3
                loss_cont2 = 10 * temp_loss_cont2 / bs3

                lowlight_loss = Loss_TV + loss_col + loss_cont + loss_cont2 + loss_percent

                enhanced_image = enhanced_image.detach()

                optimizer.zero_grad()
                lowlight_optimizer.zero_grad()
                before_normalize, outputs1 = model.forward_feature(enhanced_image)
                outputs2 = model.forward_classifier(before_normalize)

                _triplet_loss = L_triplet(outputs1, opts.batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)

                with torch.no_grad():
                    accuracy = torch.mean(
                        (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

                val_total_triple_loss += _triplet_loss.item()
                val_total_CE_loss += _CE_loss.item()
                val_total_accuracy += accuracy.item()
                val_total_LL_loss += lowlight_loss.item()

                val_total_loss = val_total_triple_loss + val_total_CE_loss + val_total_LL_loss

                if iteration % 2 == 0:
                    pbar.set_postfix(**{'val_total_triple_loss': val_total_triple_loss / (iteration + 1),
                                        'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                                        'val_lowlight_loss': val_total_LL_loss / (iteration + 1),
                                        'val_total_loss': val_total_loss / (iteration + 1),
                                        'val_accuracy': val_total_accuracy / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(2)

        print('Finish Validation')

        loss_history.append_loss(total_accuracy / epoch_step, (total_triple_loss + total_CE_loss) / epoch_step,
                                 (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)

        print('Epoch:' + str(epoch + 1) + '/' + str(Interval_Epoch))
        print('Total Loss: %.4f' % (total_Loss / epoch_step))
        torch.save(model.state_dict(), 'logs/Epoch%d-FR_Loss%.4f-Total_Loss%.4f-Val_Loss%.4f.pth' %
                   ((epoch + 1), (total_triple_loss + total_CE_loss) / epoch_step, total_loss / epoch_step,
                    (val_total_triple_loss + val_total_CE_loss) / epoch_step_val))
        torch.save(DCE_net.state_dict(), opts.snapshots_folder + "LL-Epoch" + str(epoch + 1) + '.pth')

        lr_scheduler.step()

    print("解冻训练开始************************************************")
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = True
            # Set up optimizer
            optimizer = optim.Adam(model_train.parameters(), 0.001)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    for epoch in range(Init_Epoch, Interval_Epoch):
        # =====  Train  =====
        DCE_net.train()
        model_train.train()
        trans = transforms.ToTensor()
        print('Start Train')
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Interval_Epoch}', postfix=dict, mininterval=1) as pbar:
            for iteration, batch in enumerate(train_loader):
                if iteration >= epoch_step:
                    break
                images, labels = batch
                if len(images) <= 1:
                    continue

                # print(images.shape)
                # 对比学习
                sampleL = random.sample(pathDirL, picknumber)
                X_L = Image.open(pathL + sampleL[0])
                X_L = X_L.resize((160, 160), Image.ANTIALIAS)
                X_L = trans(X_L)
                Y_L = Image.open(pathL + sampleL[1])
                Y_L = Y_L.resize((160, 160), Image.ANTIALIAS)
                Y_L = trans(Y_L)
                L = torch.stack((X_L, Y_L), 0)
                L = L.cuda()

                sampleGT = random.sample(pathDirGT, picknumber)
                X_GT = Image.open(pathGT + sampleGT[0])
                X_GT = X_GT.resize((160, 160), Image.ANTIALIAS)
                X_GT = trans(X_GT)
                Y_GT = Image.open(pathGT + sampleGT[1])
                Y_GT = Y_GT.resize((160, 160), Image.ANTIALIAS)
                Y_GT = trans(Y_GT)
                GT = torch.stack((X_GT, Y_GT), 0)
                GT = GT.cuda()

                with torch.no_grad():
                    images = torch.from_numpy(images).to(device, dtype=torch.float32)
                    labels = torch.from_numpy(labels).to(device, dtype=torch.long)

                enhanced_image_1, enhanced_image, A = DCE_net(images)

                bs3 = opts.batch_size * 3
                t, temp_loss_cont, temp_loss_cont2 = 0, 0, 0
                while t < bs3:
                    temp_images = enhanced_image[t:t + 2]
                    t += 2
                    temp_loss_cont += torch.mean(max(L_con(temp_images, GT) - L_con(temp_images, L) + 0.3,
                                                     L_con(L, temp_images) - L_con(L, temp_images)))
                    temp_loss_cont2 += torch.mean(max(L_const(temp_images, GT) - L_const(temp_images, L) + 0.04,
                                                      L_con(L, temp_images) - L_con(L, temp_images)))

                # print(GT.shape) (2,3,160,160)
                # print(L.shape)
                Loss_TV = 200 * L_TV(A)
                loss_col = 8 * torch.mean(L_color(enhanced_image))
                loss_percent = torch.mean(L_percept(images, enhanced_image))

                loss_cont = 20 * temp_loss_cont / bs3
                loss_cont2 = 10 * temp_loss_cont2 / bs3

                lowlight_loss = Loss_TV + loss_col + loss_cont + loss_cont2 + loss_percent

                enhanced_image = enhanced_image.detach()
                # enhanced_image = enhanced_image.to(device, dtype=torch.float32)

                lowlight_optimizer.zero_grad()
                optimizer.zero_grad()
                before_normalize, outputs1 = model.forward_feature(enhanced_image)
                outputs2 = model.forward_classifier(before_normalize)

                _triplet_loss = L_triplet(outputs1, opts.batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                fr_loss = _triplet_loss + _CE_loss

                total_loss = lowlight_loss + fr_loss
                fr_loss.backward()  # 反向求解梯度
                lowlight_loss.backward()
                torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), opts.grad_clip_norm)
                lowlight_optimizer.step()
                optimizer.step()

                with torch.no_grad():
                    accuracy = torch.mean(
                        (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

                total_triple_loss += _triplet_loss.item()
                total_CE_loss += _CE_loss.item()
                total_accuracy += accuracy.item()
                total_LL_loss += lowlight_loss.item()

                total_Loss = total_triple_loss + total_CE_loss + total_LL_loss

                if iteration % 20 == 0:
                    pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                        'total_CE_loss': total_CE_loss / (iteration + 1),
                                        'lowlight_loss': total_LL_loss / (iteration + 1),
                                        'total_loss': total_Loss / (iteration + 1),
                                        'accuracy': total_accuracy / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(20)

        print('Finish Train')

        model_train.eval()
        print('Start Validation')
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Interval_Epoch}', postfix=dict,
                  mininterval=1) as pbar:
            for iteration, batch in enumerate(val_loader):
                if iteration >= epoch_step_val:
                    break
                images, labels = batch
                if len(images) <= 1:
                    continue

                # 对比学习
                sampleL = random.sample(pathDirL, picknumber)
                X_L = Image.open(pathL + sampleL[0])
                X_L = X_L.resize((160, 160), Image.ANTIALIAS)
                X_L = trans(X_L)
                Y_L = Image.open(pathL + sampleL[1])
                Y_L = Y_L.resize((160, 160), Image.ANTIALIAS)
                Y_L = trans(Y_L)
                L = torch.stack((X_L, Y_L), 0)
                L = L.cuda()

                sampleGT = random.sample(pathDirGT, picknumber)
                X_GT = Image.open(pathGT + sampleGT[0])
                X_GT = X_GT.resize((160, 160), Image.ANTIALIAS)
                X_GT = trans(X_GT)
                Y_GT = Image.open(pathGT + sampleGT[1])
                Y_GT = Y_GT.resize((160, 160), Image.ANTIALIAS)
                Y_GT = trans(Y_GT)
                GT = torch.stack((X_GT, Y_GT), 0)
                GT = GT.cuda()

                with torch.no_grad():
                    images = torch.from_numpy(images).to(device, dtype=torch.float32)
                    labels = torch.from_numpy(labels).to(device, dtype=torch.long)

                enhanced_image_1, enhanced_image, A = DCE_net(images)

                bs3 = opts.val_batch_size * 3
                t, temp_loss_cont, temp_loss_cont2 = 0, 0, 0
                while t < bs3:
                    temp_images = enhanced_image[t:t + 2]
                    t += 2
                    temp_loss_cont += torch.mean(max(L_con(temp_images, GT) - L_con(temp_images, L) + 0.3,
                                                     L_con(L, temp_images) - L_con(L, temp_images)))
                    temp_loss_cont2 += torch.mean(max(L_const(temp_images, GT) - L_const(temp_images, L) + 0.04,
                                                      L_con(L, temp_images) - L_con(L, temp_images)))

                Loss_TV = 200 * L_TV(A)
                loss_col = 8 * torch.mean(L_color(enhanced_image))
                loss_percent = torch.mean(L_percept(images, enhanced_image))
                loss_cont = 20 * temp_loss_cont / bs3
                loss_cont2 = 10 * temp_loss_cont2 / bs3

                lowlight_loss = Loss_TV + loss_col + loss_cont + loss_cont2 + loss_percent
                # lowlight_loss = Loss_TV + loss_col + loss_percent

                enhanced_image = enhanced_image.detach()

                optimizer.zero_grad()
                lowlight_optimizer.zero_grad()
                before_normalize, outputs1 = model.forward_feature(enhanced_image)
                outputs2 = model.forward_classifier(before_normalize)

                _triplet_loss = L_triplet(outputs1, opts.batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)

                with torch.no_grad():
                    accuracy = torch.mean(
                        (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

                val_total_triple_loss += _triplet_loss.item()
                val_total_CE_loss += _CE_loss.item()
                val_total_accuracy += accuracy.item()
                val_total_LL_loss += lowlight_loss.item()

                val_total_loss = val_total_triple_loss + val_total_CE_loss + val_total_LL_loss

                if iteration % 2 == 0:
                    pbar.set_postfix(**{'val_total_triple_loss': val_total_triple_loss / (iteration + 1),
                                        'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                                        'val_lowlight_loss': val_total_LL_loss / (iteration + 1),
                                        'val_total_loss': val_total_loss / (iteration + 1),
                                        'val_accuracy': val_total_accuracy / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(2)

        print('Finish Validation')

        loss_history.append_loss(total_accuracy / epoch_step, (total_triple_loss + total_CE_loss) / epoch_step,
                                 (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)

        print('Epoch:' + str(epoch + 1) + '/' + str(Interval_Epoch))
        print('Total Loss: %.4f' % (total_Loss / epoch_step))
        torch.save(model.state_dict(), 'logs/Epoch%d-FR_Loss%.4f-Total_Loss%.4f-Val_Loss%.4f.pth' %
                   ((epoch + 1), (total_triple_loss + total_CE_loss) / epoch_step, total_loss / epoch_step,
                    (val_total_triple_loss + val_total_CE_loss) / epoch_step_val))
        torch.save(DCE_net.state_dict(), opts.snapshots_folder + "LL-Epoch" + str(epoch + 1) + '.pth')

        lr_scheduler.step()


if __name__ == '__main__':
    # print("默认路径："+os.getcwdu()+"*********")
    main()
