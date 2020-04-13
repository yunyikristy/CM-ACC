import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import cv2
import numpy as np

from utils import load_value_file

import tarfile
from io import BytesIO
import time

class MyTar(tarfile.TarFile):

    def build_index(self):
        if not self._loaded:
            self._load()
        self.member_map = {}
        cc = []
        print(len(self.members))
        for member in self.members:
            name = member.name
            cc.append(name)
            self.member_map[name] = member
        print(cc[:10])

    def _getmember(self, name, tarinfo=None, normalize=False):
        return self.member_map[name]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):

    f = open(annotation_path)
    video_names = []
    cnt = 0
    class_to_idx = {}
    idx_to_class = {}
    nframe = {}
    for line in f:
        tmp = line.strip().split(',')
        c = tmp[0].split('/')[0]
        if not c in class_to_idx:
            class_to_idx[c] = cnt
            idx_to_class[cnt] = c
            cnt += 1
        video_names.append((tmp[0], int(tmp[1]), class_to_idx[c]))
        #nframe[tmp[0]] = int(tmp[1])
    f.close()


    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i][0])
        audio_path = video_path.replace('video', 'audio') + '_mel.npy'
        if not os.path.exists(video_path):
            #print(video_path, audio_path, 'fuck1')
            continue
        if not os.path.exists(audio_path):
            #print(video_path, audio_path, 'fuck2')
            continue

        #n_frames_file_path = os.path.join(video_path, 'n_frames')
        #n_frames = int(load_value_file(n_frames_file_path))
        #if n_frames <= 0:
        #    continue
        n_frames = video_names[i][1]

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'audio': audio_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i][:-14].split('/')[1]
        }
        sample['label'] = video_names[i][2]

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


def make_dataset_tar(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, video_paths, audio_paths):

    f = open(annotation_path)
    video_names = []
    cnt = 0
    class_to_idx = {}
    idx_to_class = {}
    nframe = {}
    for line in f:
        tmp = line.strip().split(',')
        c = tmp[0].split('/')[0]
        if not c in class_to_idx:
            class_to_idx[c] = cnt
            idx_to_class[cnt] = c
            cnt += 1
        video_names.append((tmp[0], int(tmp[1]), class_to_idx[c]))
        #nframe[tmp[0]] = int(tmp[1])
    f.close()

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        #video_path = os.path.join(root_path, video_names[i][0])
        video_path = video_names[i][0]
        #audio_path = video_path.replace('video', 'audio') + '_mel.npy'
        audio_path = video_path + '.npy'

        #if not os.path.exists(video_path):
        #    continue
        #if not os.path.exists(audio_path):
        #    continue
        if not video_path in video_paths:
            continue
        if not audio_path in audio_paths:
            continue

        n_frames = video_names[i][1]

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'audio': audio_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i][:-14].split('/')[1]
        }
        sample['label'] = video_names[i][2]

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class

class Kinetics_va(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        video_tot = len(frame_indices)

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            
        clip = self.loader(path, frame_indices)

        audio = np.load(self.data[index]['audio'])
        audio_tot = audio.shape[0]

        tmp_indices = np.array(frame_indices).astype(np.float32)
        tmp_indices = tmp_indices / video_tot * audio_tot
        audio_indices = tmp_indices.astype(np.int32)

        audio_indices = np.clip(audio_indices, 0, audio_tot - 1)
        audio_clip = audio[audio_indices]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        #target = self.data[index]
        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        audio_clip = torch.from_numpy(audio_clip)

        return clip, audio_clip

    def __len__(self):
        return len(self.data)

class Kinetics_va_tar(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        #self.image_tar = MyTar.gzopen(root_path + '/a.tar', 'r')
        self.image_tar = tarfile.open(root_path + '/tmp.tar', 'r')
        #self.image_tar.build_index()

        namelist = self.image_tar.getnames()

        self.image_name_map = {}
        self.video_paths = set()
        for n in namelist:
            if n.endswith('.jpg'):
                tmp = n.split('/')
                newn = tmp[-3] +'/' + tmp[-2] +'/' + tmp[-1]
                self.image_name_map[newn] = n
                self.video_paths.add(tmp[-3] + '/' + tmp[-2])

        self.audio_tar = tarfile.open(root_path.replace('/video', '') + '/audio/audio.tar')
        #self.audio_tar = MyTar.gzopen(root_path.replace('/video', '') + '/audio/audio.tar')
        #self.audio_tar.build_index()
        namelist = self.audio_tar.getnames()
        self.audio_name_map = {}
        self.audio_paths = set()
        for n in namelist:
            if n.endswith('.npy'):
                tmp = n.split('/')
                newn = tmp[-2] + '/' + tmp[-1]
                self.audio_name_map[newn] = n
                self.audio_paths.add(tmp[-2] + '/' + tmp[-1])

        #keys = list(self.audio_name_map.keys())[:10]
        #for k in keys:
        #    print(k, self.audio_name_map[k])

        self.data, self.class_names = make_dataset_tar(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, self.video_paths, self.audio_paths)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        #self.loader = get_loader()

    
    def loader(self, path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = path + '/image_{:05d}.jpg'.format(i)

            if image_path in self.image_name_map:
                im = self.image_tar.extractfile(self.image_name_map[image_path])
                im = Image.open(BytesIO(im.read()))

                video.append(im)
        return video

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        video_tot = len(frame_indices)

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            
        clip = self.loader(path, frame_indices)

        audio_path = self.data[index]['audio']
        audio = self.audio_tar.extractfile(self.audio_name_map[audio_path])
        audio = np.load(BytesIO(audio.read()))
        audio_tot = audio.shape[0]

        tmp_indices = np.array(frame_indices).astype(np.float32)
        tmp_indices = tmp_indices / video_tot * audio_tot
        audio_indices = tmp_indices.astype(np.int32)

        audio_indices = np.clip(audio_indices, 0, audio_tot - 1)
        audio_clip = audio[audio_indices]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        audio_clip = torch.from_numpy(audio_clip)

        return clip, audio_clip

    def __len__(self):
        return len(self.data)
