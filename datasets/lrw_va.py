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
import pickle

from utils import load_value_file

import tarfile
from io import BytesIO
import time

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

def make_dataset_tar(root_path, subset, n_samples_for_each_video,
                 sample_duration, video_paths, audio_paths):

    video_names = list(video_paths.keys())

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = video_names[i]
        audio_path = video_path + '.npy'

        if not audio_path in audio_paths:
            continue

        n_frames = video_paths[video_path]

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

    return dataset

class TarReader(object):
    def __init__(self):
        super(TarReader, self).__init__()
        self.id_context = dict()
        self.name2member = dict()

    def read(self, tar_file, image_name):
        if tar_file in self.id_context:
            im = self.id_context[tar_file].extractfile(self.name2member[image_name])
            return im.read()
        else:
            file_handle = tarfile.open(tar_file)
            self.id_context[tar_file] = file_handle
            im = self.id_context[tar_file].extractfile(self.name2member[image_name])
            return im.read()

    def getnames(self, tar_file):
        if tar_file in self.id_context:
            im = self.id_context[tar_file].getnames()
        else:
            file_handle = tarfile.open(tar_file)
            pkl_file = tar_file.replace('.tar', '.pkl')
            if not os.path.exists(pkl_file):
                members = file_handle.getmembers()
                pickle.dump(members, open(pkl_file, 'wb'))
            else:
                members = pickle.load(open(pkl_file, 'rb'))
                file_handle.members = members
                file_handle._loaded = True

            for m in members:
                self.name2member[m.name] = m

            self.id_context[tar_file] = file_handle
            im = self.id_context[tar_file].getnames()
        return im


class Lrw_va_tar(data.Dataset):
    def __init__(self,
                 root_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        tarlist = []
        a = os.walk(root_path)
        for b, c, d in a:
            for item in d:
                if item.endswith('.tar'):
                    tarlist.append(b + '/' + item)

        self.tarreader = TarReader()

        self.video_name_map = {}
        self.video_tar_map = {}
        self.video_paths = {}
        for i, tarname in enumerate(tarlist):
            ss = time.time()
            namelist = self.tarreader.getnames(tarname)
            ee = time.time()
            readtime = ee - ss
            ss = time.time()
            for n in namelist:
                #if ('/' + subset + '/') in n:
                if n.endswith('.jpg'):
                    tmp = n.split('/')
                    newn = '/'.join(tmp[1:])
                    video_name = '/'.join(tmp[1:-1])
                    
                    newn = newn.replace('/val/', '/train/')
                    video_name = video_name.replace('/val/', '/train/')

                    self.video_name_map[newn] = n
                    self.video_tar_map[n] = tarname
                    if not video_name in self.video_paths:
                        self.video_paths[video_name] = 0
                    self.video_paths[video_name] += 1
            ee = time.time()
            buildtime = ee - ss
            print('loading ', i, tarname, readtime, buildtime)

        self.audio_tarname = root_path.replace('/video', '') + '/audio/audio.tar'
        ss = time.time()
        namelist = self.tarreader.getnames(self.audio_tarname)
        ee = time.time()
        print('loading audio', ee - ss)
        self.audio_name_map = {}
        self.audio_paths = set()
        for n in namelist:
            if ('/' + subset + '/') in n:
                if n.endswith('.npy'):
                    tmp = n.split('/')
                    newn = '/'.join(tmp[1:])
                    audio_name = '/'.join(tmp[1:])
                    self.audio_name_map[newn] = n
                    self.audio_paths.add(audio_name)

        self.data = make_dataset_tar(
            root_path, subset, n_samples_for_each_video,
            sample_duration, self.video_paths, self.audio_paths)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
    
    def loader(self, path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = path + '/image_{:05d}.jpg'.format(i)

            if image_path in self.video_name_map:
                video_name = self.video_name_map[image_path]
                tar_name = self.video_tar_map[video_name]
                im = self.tarreader.read(tar_name, video_name)
                im = Image.open(BytesIO(im))
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
        audio = self.tarreader.read(self.audio_tarname, self.audio_name_map[audio_path])
        audio = np.load(BytesIO(audio))
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
