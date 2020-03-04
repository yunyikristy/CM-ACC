import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import sys

from kmeanspp import *
from utils import AverageMeter, calculate_accuracy

def momentum_update(model, model_ema, m):
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1[1].detach().data)


def init_dict(data_loader, model_clone, batch_num):
    feature_v_dict = []
    feature_a_dict = []
    cuda = torch.device("cuda")
    for idx, (video, audio) in enumerate(data_loader):
        print('init dict batch', idx)
        if idx == batch_num:
            break
        with torch.no_grad():
            video = video.to(device=cuda)
            audio = audio.to(device=cuda)
            feature_v, feature_a = model_clone(video, audio)
            feature_v_dict.append(feature_v)
            feature_a_dict.append(feature_a)
    feature_v_dict = torch.cat(feature_v_dict, dim=0)
    feature_a_dict = torch.cat(feature_a_dict, dim=0)
    return feature_v_dict, feature_a_dict

def train_epoch(epoch, data_loader, model, model_clone, feature_v_pool, feature_a_pool, nowidx_pool, feature_v_dict, feature_a_dict, nowidx_dict, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()
    model_clone.train()

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

        feature_v, feature_a = model(video, audio)


        # sampling hard example based on gradient from pool
        #target = torch.arange(bs).to(device=device)
        feature_a_pool_all = torch.cat([feature_a.detach(), feature_a_pool], dim=0)
        feature_v_pool_all = torch.cat([feature_v.detach(), feature_v_pool], dim=0)

        cosv2a = torch.mm(feature_a_all, feature_v.t()).t()
        cosa2v = torch.mm(feature_v_all, feature_a.t()).t()

        lossv2a = criterion(cosv2a, target)
        lossa2v = criterion(cosa2v, target)

        # have not tested

        lossv2a.backward()
        lossa2v.backward()

        a_pool_gradient = feature_a_pool_all.grad.data
        v_pool_gradient = feature_v_pool_all.grad.data


        # a fake function
        a_idx = kmeanspp(a_pool_gradient)
        v_idx = kmeanspp(v_pool_gradient)

        a_pool_sample = feature_a_pool_all[a_idx]
        v_pool_sample = feature_v_pool_all[v_idx]

        pool_shape = a_idx.shape[0]
        
        feature_v_dict[nowidx_dict:nowidx_dict+pool_shape] = v_pool_sample.detach()
        feature_a_dict[nowidx:nowidx+pool_shape] = a_pool_sample.detach()
        nowidx_dict += pool_shape
        if nowidx_dict + pool_shape > feature_v_dict.shape[0]:
            nowidx_dict = 0
        # -----


        # compute loss with dict
        feature_a_dict_all = torch.cat([feature_a.detach(), feature_a_dict], dim=0)
        feature_v_dict_all = torch.cat([feature_v.detach(), feature_v_dict], dim=0)

        cosv2a = torch.mm(feature_a_all, feature_v.t()).t()
        cosa2v = torch.mm(feature_v_all, feature_a.t()).t()

        lossv2a = criterion(cosv2a, target).mean()
        lossa2v = criterion(cosa2v, target).mean()

        loss = lossv2a + lossa2v


        acc = calculate_accuracy(cosv2a, target)

        losses.update(loss.data.item(), video.size(0))
        accuracies.update(acc, video.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        momentum_update(model, model_clone, 0.9)

        # update pool
        with torch.no_grad():
            feature_v, feature_a = model_clone(video, audio)
            feature_v_pool[nowidx_pool:nowidx_pool+bs] = feature_v.detach()
            feature_a_pool[nowidx_pool:nowidx_pool+bs] = feature_a.detach()
            nowidx_pool += bs
            if nowidx_pool + bs > feature_v_pool.shape[0]:
                nowidx_pool = 0

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

    return feature_v_dict, feature_a_dict, nowidx
