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
import threading
import multiprocessing
import zipfile
from io import BytesIO
import io
import time

class _FileInFile(object):
    """A thin wrapper around an existing file object that
       provides a part of its data as an individual file
       object.
    """

    def __init__(self, fileobj, offset, size, blockinfo=None):
        self.fileobj = fileobj
        self.offset = offset
        self.size = size
        self.position = 0
        self.name = getattr(fileobj, "name", None)
        self.closed = False

        if blockinfo is None:
            blockinfo = [(0, size)]

        # Construct a map with data and zero blocks.
        self.map_index = 0
        self.map = []
        lastpos = 0
        realpos = self.offset
        for offset, size in blockinfo:
            if offset > lastpos:
                self.map.append((False, lastpos, offset, None))
            self.map.append((True, offset, offset + size, realpos))
            realpos += size
            lastpos = offset + size
        if lastpos < self.size:
            self.map.append((False, lastpos, self.size, None))

    def flush(self):
        pass

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return self.fileobj.seekable()

    def tell(self):
        """Return the current file position.
        """
        return self.position

    def seek(self, position, whence=io.SEEK_SET):
        """Seek to a position in the file.
        """
        if whence == io.SEEK_SET:
            self.position = min(max(position, 0), self.size)
        elif whence == io.SEEK_CUR:
            if position < 0:
                self.position = max(self.position + position, 0)
            else:
                self.position = min(self.position + position, self.size)
        elif whence == io.SEEK_END:
            self.position = max(min(self.size + position, self.size), 0)
        else:
            raise ValueError("Invalid argument")
        return self.position

    def read(self, size=None):
        """Read data from the file.
        """
        if size is None:
            size = self.size - self.position
        else:
            size = min(size, self.size - self.position)

        buf = b""
        while size > 0:
            while True:
                data, start, stop, offset = self.map[self.map_index]
                if start <= self.position < stop:
                    break
                else:
                    self.map_index += 1
                    if self.map_index == len(self.map):
                        self.map_index = 0
            length = min(size, stop - self.position)
            if data:
                self.fileobj.seek(offset + (self.position - start))
                b = self.fileobj.read(length)
                if len(b) != length:
                    raise ReadError("unexpected end of data")
                buf += b
            else:
                buf += NUL * length
            size -= length
            self.position += length
        return buf

    def readinto(self, b):
        buf = self.read(len(b))
        b[:len(buf)] = buf
        return len(buf)

    def close(self):
        self.closed = True
#class _FileInFile

class ExFileObject(io.BufferedReader):

    def __init__(self, fileobj, tarinfo):
        fileobj = _FileInFile(fileobj, tarinfo.offset_data,
                tarinfo.size, tarinfo.sparse)
        super().__init__(fileobj)

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
        if i % 10000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = video_names[i]
        audio_path = video_path.replace('.jpg', '.npy')

        if not audio_path in audio_paths:
            continue

        #n_frames = video_paths[video_path]
        n_frames = 31

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
            #self.lock[tar_file] = threading.RLock()
            im = self.id_context[tar_file].getnames()
        return im

class ZipReader(object):
    def __init__(self):
        super(ZipReader, self).__init__()
        self.id_context = dict()
        self.lock = multiprocessing.Lock()

    def read(self, zip_file, image_name):
        if zip_file in self.id_context:
            with self.lock:
                im = self.id_context[zip_file].open(image_name)
                res = im.read()
            return res
        else:
            file_handle = zipfile.ZipFile(zip_file)
            self.id_context[zip_file] = file_handle
            im = self.id_context[zip_file].open(image_name)
            return im.read()

    def getnames(self, zip_file):
        if zip_file in self.id_context:
            im = self.id_context[zip_file].namelist()
        else:
            file_handle = zipfile.ZipFile(zip_file)
            self.id_context[zip_file] = file_handle
            im = self.id_context[zip_file].namelist()
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

        ziplist = []
        a = os.walk(root_path)
        for b, c, d in a:
            for item in d:
                if item.endswith('.zip'):
                    ziplist.append(b + '/' + item)

        self.zipreader = ZipReader()

        self.video_name_map = {}
        self.video_zip_map = {}
        self.video_paths = {}
        for i, zipname in enumerate(ziplist):
            ss = time.time()
            namelist = self.zipreader.getnames(zipname)
            ee = time.time()
            readtime = ee - ss
            ss = time.time()
            for n in namelist:
                #if ('/' + subset + '/') in n:
                if n.endswith('.jpg'):
                    tmp = n.split('/')
                    newn = '/'.join(tmp[1:])
                    video_name = newn
                    
                    self.video_name_map[newn] = n
                    self.video_zip_map[n] = zipname
                    if not video_name in self.video_paths:
                        self.video_paths[video_name] = 0
                    self.video_paths[video_name] += 1
            ee = time.time()
            buildtime = ee - ss
            print('loading ', i, zipname, readtime, buildtime)

        self.audio_zipname = root_path.replace('/video', '') + '/audio/audio.zip'
        ss = time.time()
        namelist = self.zipreader.getnames(self.audio_zipname)
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
        #for i in frame_indices:
            #image_path = path + '/image_{:05d}.jpg'.format(i)

            #if image_path in self.video_name_map:
            #    video_name = self.video_name_map[image_path]
            #    tar_name = self.video_tar_map[video_name]
            #    im = self.tarreader.read(tar_name, video_name)
            #    im = Image.open(BytesIO(im))
            #    video.append(im)


        image_path = path
        video_name = self.video_name_map[image_path]
        zip_name = self.video_zip_map[video_name]
        im = self.zipreader.read(zip_name, video_name)
        im = Image.open(BytesIO(im))
        im = np.asarray(im)
        video = [Image.fromarray(im[:, i*240:(i+1)*240, :]) for i in frame_indices]
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
        audio = self.zipreader.read(self.audio_zipname, self.audio_name_map[audio_path])
        audio = np.load(BytesIO(audio), allow_pickle=True)
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
