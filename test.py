import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from tqdm import tqdm

from load_data import MyDataset
from utils import AverageMeter, accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Model Testing')
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='Pca', type=str, choices=['cifar10', 'cifar100', 'Pca'], help='dataset name')
    parser.add_argument('--batch-size', default=8, type=int, help='test batch size') # 如果按照原图尺寸去测试，是不是得设置为1
    parser.add_argument('--load-path', type=str, default='result2/model_best.pth.tar', help='path to the model checkpoint')
    parser.add_argument('--num-classes', type=int, default=15, help='number of classes') # 15分类二维码
    parser.add_argument('--arch', default='Resnet18', type=str, choices=['wideresnet', 'resnext', 'resnet'], help='model architecture')
    parser.add_argument('--img-size', type=int, default=256, help='image size')
    parser.add_argument('--no-progress', action='store_true', help='disable progress bar')
    args = parser.parse_args()

    device = torch.device('cuda', args.gpu_id if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(args.load_path, map_location=device, weights_only=True)
    model = create_model(args)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # # Prepare dataset and dataloader
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_val = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_flist_path = "data/test.txt"
    test_data = MyDataset(test=True, flist=test_flist_path, transforms=transform_val)
    test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.batch_size, drop_last=False)

    # Test the model
    test_loss, test_acc, top5_acc, f1, cm = test(args, test_loader, model, device)
    logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%')
    logger.info(f'F1 Score: {f1:.4f}')
    logger.info(f'Confusion Matrix:\n{cm}')

def create_model(args):
    from semilearn.nets.resnet.resnet import resnet18
    model = resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.num_classes)

    logger.info("Total model params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    return model

def test(args, test_loader, model, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targets = []
    all_predictions = []
    class_num = [0.0] * args.num_classes
    class_acc_num = [0.0] * args.num_classes
    model.eval()
    end = time.time()

    # 使用tqdm包装DataLoader
    test_loader = tqdm(test_loader, desc="Test Iter", disable=args.no_progress)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)["logits"]
            loss = F.cross_entropy(outputs, targets.long())
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())

            for i in range(len(targets)):
                class_num[targets[i]] += 1.0
                if outputs.argmax(dim=1)[i] == targets[i]:
                    class_acc_num[targets[i]] += 1.0

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}.".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))

    f1 = f1_score(all_targets, all_predictions, average='macro')
    cm = confusion_matrix(all_targets, all_predictions)
    return losses.avg, top1.avg, top5.avg, f1, cm

if __name__ == '__main__':
    main()