import gc
import json
import os
import subprocess
import time  # added import
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as tvF
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.io import VideoReader as TVVideoReader

import logging 
log = logging.getLogger(__name__)


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0 
    else:
        return dist.get_rank()

class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=json.loads(result.stdout),
                         error=result.stderr)

class BDD100KDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform,
                 delta_t=None,
                 meta_info_file=None,
                 repeat_sample=None,
                 backend='torchvision-videoreader',
                 subset=None,
                 ):
        self.root_dir = root_dir
        self.backend = backend
        self.repeat_sample = repeat_sample or 1
        self.delta_t = delta_t
        
        assert backend in ['decord', 'torchvision', 'torchvision-videoreader'], f"backend must be one of ['decord', 'torchvision', 'torchvision-videoreader'], got {backend}"
        if backend == 'torchvision-videoreader':
            torchvision.set_video_backend('video_reader')
        else:
            torchvision.set_video_backend('pyav')

        video_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)])
        if subset is not None:
            video_paths = video_paths[:subset]
        if meta_info_file is not None:
            meta_info = np.load(meta_info_file, allow_pickle=True)
            self.video_paths = []
            self.video_metadata = []
            for info in meta_info:
                if info is None:
                    continue
                length = info.get('length', 0)
                if length < repeat_sample or (delta_t is not None and length < (delta_t[1] + 1)):
                    log.warning(f"Skipping {info['video_path']} with length {length}")
                    continue
                self.video_paths.append(info['video_path'])
                self.video_metadata.append(info)
        else:
            self.video_paths = video_paths
            self.video_metadata = [None for _ in video_paths]
        self._dataset_len = len(self.video_paths)
        self.transform = transform
       
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
        if self.backend == 'decord':
            vr = VideoReader(self.video_paths[idx], num_threads=0, ctx=cpu(cpuid))
            vr_len = len(vr)
        elif 'torchvision' in self.backend:
            vr = TVVideoReader(self.video_paths[idx], "video", num_threads=0)
            # conda-base FFMPEG does not preserve rotations properly, must read manually
            if self.video_metadata[idx] is not None and 'rotation' in self.video_metadata[idx]:
                vr_rotation = int(self.video_metadata[idx]['rotation'])
            else:
                try:
                    vr_rotation = -int(ffprobe(self.video_paths[idx]).json['streams'][0]['side_data_list'][0].get('rotation', '0'))
                except:
                    vr_rotation = 0
            if self.video_metadata[idx] is not None:
                vr_md = self.video_metadata[idx]
                vr_len = vr_md['length']
            else:
                vr_md = vr.get_metadata()['video']
                vr_md = {'duration': float(vr_md['duration'][0]), 'fps': float(vr_md['fps'][0])}
                vr_len = int(vr_md['duration'] * vr_md['fps']) - 1

        if self.delta_t is not None:
            i_s = np.random.randint(0, vr_len - self.delta_t[1], size=self.repeat_sample)
            delta_ts = np.random.randint(self.delta_t[0], self.delta_t[1]+1, size=self.repeat_sample)
            i_s = np.array([index for i, delta_t in zip(i_s, delta_ts) for index in [i, i+delta_t]])
            sort_indexes = np.argsort(i_s).astype(np.int32)
            unsort_indexes = np.argsort(sort_indexes).astype(np.int32)
        else:
            i_s = np.random.randint(0, vr_len, size=self.repeat_sample)
            sort_indexes = np.argsort(i_s).astype(np.int32)
            unsort_indexes = np.argsort(sort_indexes).astype(np.int32)

        try:
            if self.backend == 'decord':
                imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]
            elif 'torchvision' in self.backend:
                res = []
                i_s_ = [x / vr_md['fps'] for x in i_s[sort_indexes]]
                for i_ in i_s_:
                    vr.seek(i_)
                    count = 0
                    while True:
                        try:
                            res.append(next(vr)['data'])
                            break
                        except StopIteration:
                            log.warning(f"StopIteration at {i_ + (count * 1/vr_md['fps'])} for {self.video_paths[idx]}")
                            count += 1
                            if count < 3:
                                vr.seek(i_ + count * 1/vr_md['fps'])
                            else:
                                log.warning(f"Failed to read frame for 3rd iteration, resorting to keyframe from {i_-(count-2)/vr_md['fps']}")
                                vr.seek(i_-(count-2)/vr_md['fps'], keyframes_only=True)
                imgs = torch.stack(res, axis=0)
                if vr_rotation != 0:
                    imgs = torch.rot90(imgs, k=-vr_rotation//90, dims=[2, 3])
                imgs = imgs.permute(0, 2, 3, 1).numpy()[unsort_indexes]
            del vr; gc.collect()

            ls = []
            for i in range(self.repeat_sample):
                if self.delta_t is not None:
                    i1, i2 = i*2, i*2+1
                    img1 = tvF.to_pil_image(imgs[i1])
                    img2 = tvF.to_pil_image(imgs[i2])
                else:
                    img1 = tvF.to_pil_image(imgs[i])
                    img2 = None
                aug_img = self.transform(img1, img2)
                if len(ls) == 0:
                    ls = [[] for _ in range(len(aug_img))]
                for j in range(len(aug_img)):
                    ls[j].append(aug_img[j])

            return [torch.stack(l, dim=0) for l in ls]
        except Exception as e:
            log.error(f"Error reading video {self.video_paths[idx]}: {e}")
            return self.__getitem__(np.random.randint(0, self._dataset_len))
