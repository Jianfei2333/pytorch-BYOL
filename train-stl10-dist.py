import copy
import math
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from dataset import STL10, MultiView
from eval import DistEvaluation
from model import BYOL, MLP, Branch, Encoder, Encoder_small, LinearEvaluator
from utils import AverageMeter, collect, get_device_id


def regression_loss(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)

    loss = 2 - 2 * (x * y).sum(dim=-1)
    return loss

def linear_evaluation(device_id, encoder, train_dataloader, test_dataloader, cl_epoch_idx, epochs):
    device = f"cuda:{device_id}"
    scaler = torch.cuda.amp.GradScaler()

    evaluator = LinearEvaluator(copy.deepcopy(encoder), train_dataloader.dataset.class_num).to(device)
    evaluator = nn.parallel.DistributedDataParallel(evaluator, device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    # optimizer = optim.SGD(evaluator.parameters(), 1e-3, 0.9, weight_decay=0)
    optimizer = optim.Adam(evaluator.parameters(), lr=1e-3, weight_decay=0)
    scheduler = None

    eval_runner = DistEvaluation("LE", evaluator, optimizer, scheduler, scaler, cl_epoch_idx, device)

    if torch.distributed.get_rank() == 0:
        tqdm.write("-" * 20)
        tqdm.write("Linear evaluation start.")
    eval_runner.train(train_dataloader, test_dataloader, epochs, 20)

def main(batch_size, lr, epochs):
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    device_id = get_device_id()
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    scaler = torch.cuda.amp.GradScaler()

    aug_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop((96, 96)),
        T.RandomApply([
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([
            T.GaussianBlur((9, 9))
        ], p=1.0),
        # GaussianBlur(kernel_size=3),
        T.ToTensor()
    ])
    test_transform = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor()
    ])
    multiview_transform = MultiView(aug_transform)
    cl_dataset = STL10(os.environ['DATAROOT'], split='train+unlabeled', transform=multiview_transform)
    cl_sampler = torch.utils.data.distributed.DistributedSampler(cl_dataset)
    cl_dataloader = DataLoader(cl_dataset, batch_size=batch_size, sampler=cl_sampler, pin_memory=False, num_workers=24)

    eval_train_dataset = STL10(os.environ['DATAROOT'], transform=aug_transform, split='train')
    eval_train_sampler = torch.utils.data.distributed.DistributedSampler(eval_train_dataset)
    eval_dataloader = DataLoader(eval_train_dataset, batch_size=32, sampler=eval_train_sampler, pin_memory=False, num_workers=24)

    eval_test_dataset = STL10(os.environ['DATAROOT'], transform=test_transform, split='test')
    eval_test_sampler = torch.utils.data.distributed.DistributedSampler(eval_test_dataset)
    eval_test_dataloader = DataLoader(eval_test_dataset, batch_size=512, sampler=eval_test_sampler, pin_memory=False, num_workers=24)

    mean=[.5, .5, .5]
    std=[.5, .5, .5]
    # mean=[0., 0., 0.]
    # std=[1., 1., 1.]

    online_branch = Branch(Encoder(mean=mean, std=std, backbone='resnet18'), MLP(512, 512, 128)).to(device)
    target_branch = Branch(Encoder(mean=mean, std=std, backbone='resnet18'), MLP(512, 512, 128)).to(device)
    predictor = MLP(128, 512, 128).to(device)
    byol = BYOL(online_branch=online_branch, target_branch=target_branch, predictor=predictor).to(device)
    byol = nn.parallel.DistributedDataParallel(byol, device_ids=[device_id], output_device=device_id)


    linear_evaluation(device_id, byol.module.online.encoder, eval_dataloader, eval_test_dataloader, 0, 200)

    optimizer = optim.SGD(byol.parameters(), lr=lr, momentum=0.9, weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(cl_dataloader), eta_min=1e-3, last_epoch=-1)

    tau_base = 0.996
    # compute_tau = lambda step: 1-(1-tau_base) * (math.cos(math.pi * step/epochs/len(cl_dataloader)) + 1) / 2
    compute_tau = lambda step: tau_base

    for epoch_idx in range(epochs):
        pbar = tqdm(total=len(cl_dataloader), leave=False, desc="BYOL {}/{}".format(epoch_idx, epochs))
        loss_meter = AverageMeter()
        for batch_idx, ((x_view1, x_view2), _) in enumerate(cl_dataloader):
            x_view1, x_view2 = x_view1.to(device), x_view2.to(device)

            with torch.cuda.amp.autocast():
                q1, q2, t1, t2 = byol(x_view1, x_view2)
                loss = (regression_loss(q1, t1) + regression_loss(q2, t2)).mean()
            
            loss_meter.update(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()
            
            tau = compute_tau(epoch_idx*len(cl_dataloader)+batch_idx)
            byol.module.momentum_update(tau)

            pbar.set_postfix_str("L. {:.4f}".format(loss.item()))
            pbar.update(1)
        pbar.close()

        avg_loss = loss_meter.report()
        avg_loss = collect(avg_loss, device)

        if torch.distributed.get_rank() == 0:
            tqdm.write("Epoch {} (total {}) Avg training loss: {:.4f}".format(epoch_idx, epochs, avg_loss))

        if (epoch_idx % 20 == 19):
            linear_evaluation(device_id, byol.module.online.encoder, eval_dataloader, eval_test_dataloader, epoch_idx, 200)
        if (epoch_idx % 100 == 99) and torch.distributed.get_rank() == 0:
            torch.save(byol.module.online.encoder.state_dict(), "stl10-epoch{}.pt".format(epoch_idx))

    if torch.distributed.get_rank() == 0:
        torch.save(byol.module.online.encoder.state_dict(), "stl10-final.pt")

if __name__ == '__main__':
    main(batch_size=64, lr=0.03, epochs=40)
