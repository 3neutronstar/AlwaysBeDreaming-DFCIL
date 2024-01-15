import torch
import torchvision.transforms as TF
from iscf_module import ISCF_ResNet
from cl_lite.head.dynamic_simple import DynamicSimpleHead
import cl_lite.backbone as B
import os

from datamodule import DataModule

def main():
    dataset = "cifar100"
    num_classes = 100
    data_root = '/data'
    class_order = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80, 2, 47, 87, 78, 66, 19, 6, 24, 10, 59, 30, 22, 29, 83, 37, 93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 84, 70, 4, 20, 15, 21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0, 72, 35, 58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9, 82, 39, 18, 77, 68, 32, 79, 12, 85, 36, 17, 64, 27, 74, 45, 61, 38, 51, 62, 65, 33, 5, 53, 97, 49]
    num_tasks = 5
    current_task = 2
    prefix = './lightning_logs/version_71/task_1/checkpoints/'
    if dataset.startswith("imagenet"):
        backbone = B.resnet.resnet18()
    else:
        backbone = ISCF_ResNet()
    


    # model = torch.nn.Sequential(backbone,head)

    state_dict = torch.load(os.path.join(prefix,"best_acc.ckpt"))['state_dict']

    # dataload
    data_module = DataModule(root=data_root, 
                             dataset=dataset, 
                             batch_size=128, 
                             num_workers=4,
                             num_tasks=num_tasks,
                             class_order=class_order,
                             current_task=current_task,
                             )
    data_module.setup()
    head = DynamicSimpleHead(num_classes=data_module.num_classes, num_features=backbone.num_features, bias=True)

    for t in range(current_task-1):
        head.append(num_classes//num_tasks)

    backbone_state= {}
    head_state = {}
    for k,v in state_dict.items():
        if k.startswith('backbone'):
            backbone_state[k[9:]] = v
        elif k.startswith('head'):
            head_state[k[5:]] = v
    backbone.load_state_dict(backbone_state)
    backbone.eval()
    head.load_state_dict(head_state)


    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    backbone.cuda()
    head.cuda()

    for batch in train_dataloader:
        images, labels = batch
        
        images = images.cuda()
        labels = labels.cuda()

    pass

if __name__ == "__main__":
    main()