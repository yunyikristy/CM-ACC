from datasets.kinetics import Kinetics
from datasets.kinetics_va import Kinetics_va, Kinetics_va_tar
from datasets.lrw_va import Lrw_va_tar
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.ucf101_va import UCF101_va
from datasets.hmdb51 import HMDB51


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'kinetics_va', 'ucf101_va', 'kinetics_va_tar', 'lrw_va_tar']

    if opt.dataset == 'kinetics_va':
        training_data = Kinetics_va(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif opt.dataset == 'kinetics_va_tar':
        training_data = Kinetics_va_tar(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif opt.dataset == 'lrw_va_tar':
        training_data = Lrw_va_tar(
            opt.video_path,
            'train',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101_va':
        training_data = UCF101_va(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'validation',
            False,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return test_data
