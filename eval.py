import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils import AverageMeter

class Evaluation():
    def __init__(self, mode, evaluator, optimizer, scheduler, global_epoch_idx, device):
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_epoch_idx = global_epoch_idx
        self.mode = mode
        self.desc = lambda status, progress: f"{mode} {status} {global_epoch_idx}: {progress}"
        self.device = device

    def eval(self, loader, progress):
        tp = 0
        total = 0
        with torch.no_grad():
            self.evaluator.eval()
            pbar = tqdm(total=len(loader), leave=False, desc=self.desc("eval", progress))
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.evaluator(data)
                pred = output.argmax(dim=1)
                tp += (pred == target).sum().item()
                total += pred.shape[0]
                pbar.update(1)
            pbar.close()
        
        return (tp, total)

    def train_step(self, loader, progress):
        self.evaluator.train()
        pbar = tqdm(total=len(loader), leave=False, desc=self.desc("train", progress))
        for batch_idx, batch in enumerate(loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            output = self.evaluator(data)
            loss = nn.functional.cross_entropy(output, target)

            pbar.set_postfix_str("L. {:.4f}".format(loss.item()))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.update(1)
        pbar.close()

    def train(self, train_loader, test_loader, n_epochs, log_interval):
        tp, total = self.eval(test_loader, "Start")
        tqdm.write("Initial Prediction: {}/{}, {:.4f}".format(tp, total, tp/total))

        for epoch_idx in range(n_epochs):
            self.train_step(train_loader, "{}/{}".format(epoch_idx, n_epochs))
            self.scheduler.step()

            if epoch_idx % log_interval == (log_interval-1):
                tp, total = self.eval(test_loader, "{}/{}".format(epoch_idx, n_epochs))
                tqdm.write("Epoch {} (total {}), Prediction: {}/{}, {:.4f}".format(epoch_idx, n_epochs, tp, total, tp/total))

        tqdm.write("Finished evaluation!")
        tqdm.write("-" * 20)

class DistEvaluation():
    def __init__(self, mode, evaluator, optimizer, scheduler, scaler, global_epoch_idx, device):
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.global_epoch_idx = global_epoch_idx
        self.mode = mode
        self.desc = lambda status, progress: f"{mode} {status} {global_epoch_idx}: {progress}"
        self.device = device

    def collect(self, x, mode='mean'):
        xt = torch.tensor([x]).to(self.device)
        torch.distributed.all_reduce(xt, op=torch.distributed.ReduceOp.SUM)
        # print(xt.item())
        xt = xt.item()
        if mode == 'mean':
            xt /= torch.distributed.get_world_size()
        return xt

    def eval(self, loader, progress):
        pred_meter = AverageMeter()
        with torch.no_grad():
            self.evaluator.eval()
            pbar = tqdm(total=len(loader), leave=False, desc=self.desc("eval", progress))
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                with torch.cuda.amp.autocast():
                    output = self.evaluator(data)
                pred = output.argmax(dim=1)
                
                pred_meter.update((pred==target).sum().item(), pred.shape[0])
                
                pbar.update(1)
            pbar.close()
        
        return (int(pred_meter.sum), pred_meter.count)

    def train_step(self, loader, progress):
        self.evaluator.train()
        avg_loss_meter = AverageMeter()
        pbar = tqdm(total=len(loader), leave=False, desc=self.desc("train", progress))
        for batch_idx, batch in enumerate(loader):
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            # Auto mixed precision.
            with torch.cuda.amp.autocast():
                output = self.evaluator(data)
                loss = nn.functional.cross_entropy(output, target)

            pbar.set_postfix_str("L. {:.4f}".format(loss.item()))
            avg_loss_meter.update(loss.item())
            
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            pbar.update(1)
        pbar.close()

        return avg_loss_meter.report()

    def train(self, train_loader, test_loader, n_epochs, log_interval):
        tp, total = self.eval(test_loader, "Start")
        tp = self.collect(tp, mode='sum')
        total = self.collect(total, mode='sum')
        if torch.distributed.get_rank() == 0:
            tqdm.write("Initial Prediction: {}/{}, {:.4f}".format(tp, total, tp/total))

        for epoch_idx in range(n_epochs):
            _ = self.train_step(train_loader, "{}/{}".format(epoch_idx, n_epochs))
            if self.scheduler is not None:
                self.scheduler.step()

            if epoch_idx % log_interval == (log_interval-1):
                tp, total = self.eval(test_loader, "{}/{}".format(epoch_idx, n_epochs))
                tp = self.collect(tp, mode='sum')
                total = self.collect(total, mode='sum')
                if torch.distributed.get_rank() == 0:
                    tqdm.write("Epoch {} (total {}), Prediction: {}/{}, {:.4f}".format(epoch_idx, n_epochs, int(tp), total, tp/total))

        if torch.distributed.get_rank() == 0:
            tqdm.write("Finished evaluation!")
            tqdm.write("-" * 20)