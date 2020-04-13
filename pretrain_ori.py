import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    cuda = torch.device("cuda")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (video, audio) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        video = video.to(device=cuda)
        audio = audio.to(device=cuda)
        bs = video.size(0)
        device = video.device

        #feature_v, feature_a = model(video, audio)
        #target = torch.arange(bs).to(device=device)
        #cosv2a = torch.mm(feature_a, feature_v.t()).t()
        #cosa2v = torch.mm(feature_v, feature_a.t()).t()

        #loss1 = criterion(cosv2a, target)
        #loss2 = criterion(cosa2v, target)
        #loss = loss1 + loss2

        #tmp = cosv2a.argmax(dim=1)
        #acc = calculate_accuracy(cosv2a, target)

        fc_correct, fc_wrong = model(video, audio)
        fc = torch.cat([fc_correct, fc_wrong], dim=0)
        target = torch.cat([torch.ones((bs, ), dtype=torch.long), torch.zeros((bs, ), dtype=torch.long)], dim=0)
        target = target.to(device=device)
        loss = criterion(fc, target)
        acc = calculate_accuracy(fc, target)

        losses.update(loss.data.item(), video.size(0))
        accuracies.update(acc, video.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

