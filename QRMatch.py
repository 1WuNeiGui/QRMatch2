import argparse
import logging
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# import moco
from moco.builder import MoCo
from FCR import FCR

from utils import AverageMeter, accuracy
from collections import Counter
from copy import deepcopy

# SAB
from aia import AIASelector
from SuperAugmentation import Super_Augmentation
from grad_conflict import GradConflictLogger, encoder_parameters, gradient_cosine, should_measure

logger = logging.getLogger(__name__)
best_acc = 0


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


# 设置seed
def set_seed(seed):
    logging.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8, # 15号机器用的8
                        help='number of workers')
    parser.add_argument('--num_classes', default=15, type=int, help='number of classes')
    parser.add_argument('--dataset', default='二维码', type=str,  # log输出用
                        choices=['cifar10', 'cifar100', '二维码'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=153,  # log输出用的
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='Resnet18', type=str,  # 有用
                        choices=['wideresnet', 'resnext', 'Resnet18', 'Resnet34'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=30 * 500, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=30, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--aia-warmup-epochs', default=100, type=int,
                        help='epochs to wait before enabling AIA re-augmentation')
    parser.add_argument('--aia-alpha', default=0.5, type=float,
                        help='EMA smoothing factor alpha of the AIA sample-loss history (Eq. 13)')
    parser.add_argument('--aia-noise-std', default=1.0, type=float,
                        help='std of the Gaussian noise in AIA re-augmentation')
    parser.add_argument('--aia-mask-prob', default=0.1, type=float,
                        help='pixel mask ratio in AIA re-augmentation')
    parser.add_argument('--aia-off', action='store_true',
                        help='disable AIA completely (the statistics are still recorded)')
    parser.add_argument('--aia-only', default='all', choices=['all', 'correct', 'incorrect'],
                        help='oracle: re-augment only the simple samples whose pseudo-label is '
                             'correct / incorrect (ground-truth labels are used; analysis only)')
    parser.add_argument('--aia-zero', default='none', choices=['none', 'correct', 'incorrect'],
                        help='oracle: set the consistency loss of the simple samples whose '
                             'pseudo-label is correct / incorrect to 0 (analysis only)')
    parser.add_argument('--aia-track', default=8, type=int,
                        help='number of random samples whose loss trajectory is recorded')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')  # 这里先设置为1吧
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,  # 伪标签置信度阈值
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    # Gradient conflict analysis between L_DDCL and L_cons (cosine similarity of gradients)
    parser.add_argument('--grad-cos', type=str2bool, default=True,
                        help='record the cosine similarity between DDCL and consistency gradients')
    parser.add_argument('--grad-cos-start', default=100, type=int,
                        help='first epoch (1-based) at which the gradient cosine is recorded')
    parser.add_argument('--grad-cos-interval', default=25, type=int,
                        help='record the gradient cosine every N epochs after --grad-cos-start')
    parser.add_argument('--grad-cos-tag', default='with_dtf', type=str,
                        help='tag used in the CSV file names, e.g. with_dtf or without_dtf')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    # Flexmatch
    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--use_DA', type=str2bool, default=False)
    parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=True)
    # Flexmatch

    # MOCO V2
    parser.add_argument("--moco-dim", default=512, type=int, help="feature dimension (default: 128)") # resnet34和18应该是512，resnet50应该是2048
    parser.add_argument(
    "--moco-k",
    default=1024,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
    parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
    parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)
    parser.add_argument("--mlp", action="store_true", help="use mlp head")
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == "Resnet18": # 用Resnet-18试试
            from semilearn.nets.resnet.resnet import resnet18
            model = resnet18()
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, args.num_classes)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args.seed)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        # args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        # args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    from data_preprocessing import labeled_dataset, unlabeled_dataset, val_dataset

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    # The main model is shared by the classifier, MoCo query branch, and FCR.
    # MoCo owns only an EMA copy of this model as its momentum/key encoder.
    moco = MoCo(
        encoder_q=model,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
    ).to(args.device)
    fcr = FCR().to(args.device)

    # Only the shared main model is updated by gradient descent.
    # The MoCo key encoder is frozen and updated by EMA, not by the optimizer.
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if 'moco_state_dict' in checkpoint:
            moco.load_state_dict(checkpoint['moco_state_dict'])
        else:
            # Legacy checkpoints have no saved MoCo state; initialize its EMA
            # key encoder from the restored main/query encoder.
            moco.initialize_key_encoder(model)
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, val_loader,
          model, optimizer, ema_model, scheduler, unlabeled_dataset, moco, fcr)
    
def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
    

def consistency_loss(logits_s, logits_w, class_acc, p_target, p_model, index_u, history_dict, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False, sample_weight=None):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        if use_DA:
            if p_model == None:
                p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex

        # 这里用SAB的进站代码
        # SAB超级增强模块,这个放select之前还是之后？
        update_dict = (F.cross_entropy(logits_s.clone().detach(), max_idx.clone().detach(), # logits_s对应logits_u_s,max_idx对应targets_u
                                           reduction='none'))
        update_dict = update_dict.tolist()
        flag = 0
        for index in index_u:
            history_dict[index].append(update_dict[flag])
            # flag += 1
            flag = flag + 1
        flag = 0
        # SAB超级增强模块结束

        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(p_cutoff).long()
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        if sample_weight is not None:  # analysis only: zero the loss of chosen samples
            masked_loss = masked_loss * sample_weight
        return masked_loss.mean(), mask.mean(), select, max_idx.long(), p_model

    else:
        assert Exception('Not Implemented consistency_loss')


def train(args, labeled_trainloader, unlabeled_trainloader, val_loader,
          model, optimizer, ema_model, scheduler, unlabeled_dataset, moco, fcr):
    if args.amp:
        from apex import amp
    global best_acc
    lambda_ddcl = 0.5
    grad_logger = None
    if args.grad_cos and args.local_rank in [-1, 0]:
        grad_logger = GradConflictLogger(args.out, tag=args.grad_cos_tag)
        grad_params = encoder_parameters(model)
    test_accs = []
    history_dict = {}
    # AIA selector (paper Sec. III-D): EMA of sample losses, class-wise Otsu thresholds.
    # Labels are used only to log statistics / run the oracle analysis, never for training.
    aia = AIASelector(len(unlabeled_dataset), args.num_classes, unlabeled_dataset.label,
                      alpha=args.aia_alpha, out_dir=args.out, track_n=args.aia_track,
                      seed=args.seed)
    end = time.time()

    # Flexmatch
    classnum = args.num_classes
    it = 0

    p_model = torch.ones(classnum)
    p_model = p_model.to(args.device)
    classwise_acc = torch.zeros((args.num_classes,)).cuda(args.device)
    selected_label = torch.ones((len(unlabeled_dataset),), dtype=torch.long, ) * -1
    selected_label = selected_label.cuda(args.device)
    # Flexmatch

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        t_fn = Get_Scalar(args.T)  # temperature params function
        p_fn = Get_Scalar(args.threshold)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_ddcl = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            optimizer.zero_grad(set_to_none=True)
            try:
                inputs_x, targets_x, index_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, index_x = next(labeled_iter)

            try:
                inputs_u_s, inputs_u_s2, inputs_u_w, ulb_path, index_u = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u_s, inputs_u_s2, inputs_u_w, ulb_path, index_u = next(unlabeled_iter)
            
            x_ulb_idx = index_u

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]

            # Keep integer sample indices and initialize histories during warm-up.
            # consistency_loss() appends each sample's current loss later in this iteration.
            index_u = index_u.tolist()
            for value in index_u:
                if value not in history_dict:
                    history_dict[value] = []

            # AIA: simple samples = EMA loss below the class-wise Otsu threshold (Eq. 16).
            # The EMA / threshold state is the one before this iteration. epoch is zero-based,
            # so default=100 disables re-augmentation for epochs 0-99.
            zero_w = None
            simple = aia.is_simple(index_u)
            right = aia.correct(index_u)
            if epoch >= args.aia_warmup_epochs and not args.aia_off:
                chosen = simple.copy()
                if args.aia_only == 'correct':
                    chosen &= right
                elif args.aia_only == 'incorrect':
                    chosen &= ~right
                for idx in np.nonzero(chosen)[0]:
                    inputs_u_s2[idx] = Super_Augmentation(
                        inputs_u_s2[idx], mask_prob=args.aia_mask_prob,
                        noise_std=args.aia_noise_std).to(inputs_u_s2.dtype)
                if args.aia_zero != 'none':
                    drop = simple & (right if args.aia_zero == 'correct' else ~right)
                    zero_w = torch.from_numpy((~drop).astype(np.float32))

            # Flexmatch的相关代码
            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(unlabeled_dataset):  # not all(5w) -1
                if args.thresh_warmup:
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
            # ---------------------------------------------------------------------------

            # The AIA/pseudo-label path uses u_s2, a copy of u_s; DDCL retains the original u_s.
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s2)), 2 * args.mu + 1).to(args.device)  # 交错排列
            targets_x = targets_x.to(args.device)
            logits = model(inputs)['logits']
            logits = de_interleave(logits, 2 * args.mu + 1)  # 反交错排列
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # Flexmatch----------------------------------------------------------------------
            p_target = targets_x
            T = t_fn(it)
            p_cutoff = p_fn(it)
            it = it + 1

            Lu, mask, select, pseudo_lb, p_model = consistency_loss(logits_u_s,
                                                                    logits_u_w,
                                                                    classwise_acc,
                                                                    p_target,
                                                                    p_model,
                                                                    index_u,
                                                                    history_dict,
                                                                    'ce', T, p_cutoff,
                                                                    use_hard_labels=args.hard_label,
                                                                    use_DA=args.use_DA,
                                                                    sample_weight=None if zero_w is None
                                                                    else zero_w.to(logits_u_s.device))
            # update the AIA state with this iteration's per-sample consistency loss
            with torch.no_grad():
                cur_loss = F.cross_entropy(logits_u_s.detach(), pseudo_lb, reduction='none')
            aia.update(index_u, cur_loss.cpu().numpy(), pseudo_lb.cpu().numpy())
            select = select.to(x_ulb_idx.device)
            x_ulb_idx = x_ulb_idx.to(select.device)
            if x_ulb_idx[select == 1].nelement() != 0:
                selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]
            # ---------------------------------------------------------------------------

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            # mask = max_probs.ge(args.threshold).float()

            # Lu = (F.cross_entropy(logits_u_s, targets_u,
            #                       reduction='none') * mask).mean()

            # DDCL directly uses u_s; AIA's in-place edits are confined to the u_s2 copy.
            inputs_q_ddcl = inputs_u_s.to(args.device)
            inputs_k_ddcl = inputs_u_w.to(args.device)
            moco_output, moco_target, negative_images = moco(
                encoder_q=model, im_q=inputs_q_ddcl, im_k=inputs_k_ddcl)
            moco_loss = F.cross_entropy(moco_output, moco_target)

            # FCR uses spatial feature maps from the trainable main encoder, not raw-image FFTs.
            # Keep the anchor path differentiable; positive and negative features are stop-gradient.
            # The FIFO image queue has the same capacity K=args.moco_k as the MoCo queue.
            # Encode every historical image in small chunks to limit peak GPU memory, while
            # averaging the loss over all queued negatives (not a random subset).
            fcr_anchor_features = model(inputs_q_ddcl, only_map=True)
            with torch.no_grad():
                fcr_positive_features = model(inputs_k_ddcl, only_map=True)

            if negative_images:
                negative_chunk_size = 32
                fcr_loss_value_sum = None
                fcr_anchor_grad_sum = torch.zeros_like(fcr_anchor_features)
                total_negatives = 0

                for start in range(0, len(negative_images), negative_chunk_size):
                    image_chunk = negative_images[start:start + negative_chunk_size]
                    with torch.no_grad():
                        negative_image_batch = torch.stack(image_chunk).to(
                            device=args.device, dtype=inputs_q_ddcl.dtype)
                        fcr_negative_features = model(
                            negative_image_batch, only_map=True)

                    chunk_negatives = fcr_negative_features.shape[0]
                    chunk_loss = fcr(
                        fcr_anchor_features,
                        fcr_positive_features,
                        fcr_negative_features,
                    )
                    # Backpropagate each chunk only as far as the anchor feature map.
                    # This releases that chunk's FFT intermediates immediately instead of
                    # retaining all 1024-negative comparison tensors until loss.backward().
                    weighted_chunk_loss = chunk_loss * chunk_negatives
                    chunk_anchor_grad = torch.autograd.grad(
                        weighted_chunk_loss,
                        fcr_anchor_features,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    fcr_anchor_grad_sum = fcr_anchor_grad_sum + chunk_anchor_grad.detach()

                    weighted_chunk_value = weighted_chunk_loss.detach()
                    if fcr_loss_value_sum is None:
                        fcr_loss_value_sum = weighted_chunk_value
                    else:
                        fcr_loss_value_sum = fcr_loss_value_sum + weighted_chunk_value
                    total_negatives += chunk_negatives

                fcr_loss_value = fcr_loss_value_sum / total_negatives
                fcr_anchor_grad = fcr_anchor_grad_sum / total_negatives
                # Preserve the exact scalar loss value while supplying the accumulated
                # gradient to the shared encoder during the main loss.backward() call.
                fcr_loss = fcr_loss_value + (
                    (fcr_anchor_features - fcr_anchor_features.detach()) * fcr_anchor_grad
                ).sum()
            else:
                fcr_loss = fcr(
                    fcr_anchor_features, fcr_positive_features, None)
            loss_ddcl = lambda_ddcl * moco_loss + (1.0 - lambda_ddcl) * fcr_loss

            loss = Lx + args.lambda_u * Lu + loss_ddcl

            # Gradient conflict analysis: cosine between grad(L_DDCL) and grad(L_cons)
            # w.r.t. the shared encoder, before the normal backward pass.
            if grad_logger is not None and should_measure(
                    epoch + 1, args.grad_cos_start, args.grad_cos_interval):
                cos_sim, norm_ddcl, norm_cons = gradient_cosine(loss_ddcl, Lu, grad_params)
                grad_logger.log_iter(epoch + 1, batch_idx + 1, cos_sim, norm_ddcl, norm_cons,
                                     loss_ddcl.item(), Lu.item())

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_ddcl.update(loss_ddcl.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_ddcl:{loss_ddcl:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        loss_ddcl=losses_ddcl.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            val_model = ema_model.ema
        else:
            val_model = model

        if args.local_rank in [-1, 0]:
            val_loss, val_acc = test(args, val_loader, val_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.ddcl_loss', losses_ddcl.avg, epoch)
            args.writer.add_scalar('train/5.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', val_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', val_loss, epoch)
            aia_row = aia.end_epoch(epoch)
            args.writer.add_scalar('aia/1.n_simple', aia_row['n_simple'], epoch)
            args.writer.add_scalar('aia/2.simple_precision', aia_row['simple_precision'], epoch)
            args.writer.add_scalar('aia/3.pl_acc_seen', aia_row['pl_acc_seen'], epoch)
            if grad_logger is not None:
                cos_mean = grad_logger.end_epoch(epoch + 1)
                if cos_mean is not None:
                    args.writer.add_scalar('train/6.grad_cos_ddcl_cons', cos_mean, epoch + 1)
                    logger.info('Gradient cosine (DDCL vs. consistency) at epoch {}: {:.4f}'.format(
                        epoch + 1, cos_mean))
                    # Redraw the curve after every recorded point, so the final figure is
                    # ready when training ends. A plotting error never stops training.
                    try:
                        from plot_grad_conflict import plot_runs
                        label = {'with_dtf': 'With DTF', 'without_dtf': 'Without DTF'}.get(
                            args.grad_cos_tag, args.grad_cos_tag)
                        plot_runs([(grad_logger.epoch_path, label)],
                                  os.path.join(args.out, 'grad_conflict_' + args.grad_cos_tag))
                    except Exception as e:
                        logger.warning('Gradient cosine plot skipped: {}'.format(e))

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'moco_state_dict': moco.state_dict(),
            }, is_best, args.out)

            test_accs.append(val_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        val_loader = tqdm(val_loader,
                          disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)['logits']
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                val_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            val_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
