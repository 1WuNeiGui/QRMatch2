from torchvision import transforms

from load_data import MyDataset
from semilearn.datasets.augmentation.randaugment import RandAugment

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]  # 均值
std['imagenet'] = [0.229, 0.224, 0.225]  # 标准差

# 弱增强，用于有监督分支和无监督分支
transform_weak = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean['imagenet'], std['imagenet'])
])

# 强增强：每张无标签图像只采样一次，第二路由 MyDataset 复制同一视图。
transform_strong = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    RandAugment(2, 10),
    transforms.ToTensor(),
    transforms.Normalize(mean['imagenet'], std['imagenet'])
])

# 可能是用于val数据集的transformer
transform_val = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean['imagenet'], std['imagenet'])
])

lb_flist_path = "data/label.txt"  # 有标签数据
ulb_flist_path = "data/unlabel.txt"  # 无标签数据
test_flist_path = "data/val.txt"  # 测试数据 不过这里的test数据在每个epoch后都会进行一次，看起来更像是val验证数据

unlabeled_dataset = MyDataset(ulb_train=True, flist=ulb_flist_path, weak_stansform=transform_weak,
                           strong_stansform=transform_strong)  # 创建无标签数据
labeled_dataset = MyDataset(lb_train=True, flist=lb_flist_path, weak_stansform=transform_weak)  # 创建有标签数据集
val_dataset = MyDataset(test=True, flist=test_flist_path, transforms=transform_val)  # 创建val数据集
# 到这里结束，上面的东西都是新加的，参考了颖哥的代码