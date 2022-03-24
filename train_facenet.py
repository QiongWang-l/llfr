import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import LossHistory, triplet_loss, weights_init
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils_fit import fit_one_epoch

#------------------------------------------------#
#   计算一共有多少个人，用于利用交叉熵辅助收敛
#------------------------------------------------#
def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

if __name__ == "__main__":

    Cuda            = True
    annotation_path = "cls_train_test.txt"
    input_shape     = [160,160,3]
    backbone        = "mobilenet"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    model_path      = "checkpoints/facenet_mobilenet.pth"
    Freeze_Train    = True

    num_workers     = 4

    lfw_eval_flag   = False
    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "checkpoints/lfw_pair.txt"

    num_classes = get_num_classes(annotation_path)
    #---------------------------------#
    #   载入模型并加载预训练权重
    #---------------------------------#
    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss            = triplet_loss()
    loss_history    = LossHistory("logs")
    #---------------------------------#
    #   LFW估计
    #---------------------------------#
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False) if lfw_eval_flag else None

    #-------------------------------------------------------#
    #   0.05用于验证，0.95用于训练
    #-------------------------------------------------------#
    val_split = 0.05
    with open(annotation_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #---------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    #---------------------------------------------------------#
    #---------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #---------------------------------------------------------#
    if True:
        #----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        #----------------------------------------------------#
        lr              = 1e-3
        Batch_size      = 4
        Init_Epoch      = 0
        Interval_Epoch  = 50

        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
        optimizer       = optim.Adam(model_train.parameters(), lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes)

        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
            
        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Interval_Epoch, Cuda, LFW_loader, Batch_size, lfw_eval_flag)
            lr_scheduler.step()

    if True:
        #----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        #----------------------------------------------------#
        lr              = 1e-4
        Batch_size      = 32
        Interval_Epoch  = 50
        Epoch           = 100

        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer       = optim.Adam(model_train.parameters(), lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes)

        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        for epoch in range(Interval_Epoch, Epoch):
            fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, LFW_loader, Batch_size, lfw_eval_flag)
            lr_scheduler.step()
