from bisect import bisect
import gc
import glob
import json
import os
import subprocess
from typing import Callable, List, NamedTuple

from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as tvF
from decord import VideoReader, cpu
from torch.utils import data as torchdata
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import VideoReader as TVVideoReader
from torchvision.transforms import InterpolationMode

# from line_profiler import profile
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

class WalkingToursDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform: Callable,
                 delta_t=None,
                 mode: str = "train",
                 repeat_sample: int = None,
                 chunk_ratio: int = 4,
                 backend='decord',
                 dataset_fraction=None,
                 ):
        assert mode in ["train", "val", "test"]
        
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.repeat_sample = repeat_sample
        self.delta_t = delta_t
        self.backend = backend
        self.dataset_fraction = dataset_fraction
        
        assert backend in ['decord', 'torchvision', 'torchvision-videoreader'], f"backend must be one of ['decord', 'torchvision', 'torchvision-videoreader'], got {backend}"
        if backend == 'torchvision-videoreader':
            torchvision.set_video_backend('video_reader')
        else:
            torchvision.set_video_backend('pyav')

        video_paths = glob.glob(os.path.join(data_dir, '**', "*.mp4"), recursive=True)

        self.video_paths = video_paths
        self.video_idx_start = []
        self.video_md = []
        self.video_rotations = []
        self.sample_indices = []
        idx = 0
        for video_path in video_paths:
            _, vr_len, vr_md, vr_rotation = self.create_video_reader(video_path, None, 0)

            # Calculate the potential number of samples for the *entire* video first
            full_vr_len = vr_len
            if self.delta_t is not None:
                full_vr_len = max(0, vr_len - self.delta_t[1])
            full_num_samples = (full_vr_len // (chunk_ratio * repeat_sample)) * chunk_ratio

            target_num_samples = full_num_samples
            if self.dataset_fraction is not None and 0.0 < self.dataset_fraction < 1.0:
                target_num_samples = int(full_num_samples * self.dataset_fraction)
                target_num_samples = (target_num_samples // (chunk_ratio)) * chunk_ratio

            all_sample_groups = []
            arange_limit = (full_num_samples // chunk_ratio) * chunk_ratio * repeat_sample
            all_indices = np.arange(arange_limit)
            all_chunks = np.split(all_indices, full_num_samples // chunk_ratio)
            for chunk in all_chunks:
                all_sample_groups.extend(np.split(np.random.permutation(chunk), chunk_ratio))

            selected_group_indices = np.random.choice(len(all_sample_groups), size=target_num_samples, replace=False)
            selected_groups = [all_sample_groups[i] for i in selected_group_indices]

            self.video_idx_start.append(idx)
            idx += target_num_samples
            self.video_md.append(vr_md)
            self.video_rotations.append(vr_rotation)
            self.sample_indices.extend(selected_groups)

        self._dataset_len = idx
    
    def create_video_reader(self, video_path, video_idx, cpuid):
        if video_idx is not None:
            video_path = self.video_paths[video_idx]
        if self.backend == 'decord':
            vr = VideoReader(video_path, num_threads=0, ctx=cpu(cpuid))
            vr_len = len(vr)
            vr_md = None
            vr_rotation = None
        elif 'torchvision' in self.backend:
            vr = TVVideoReader(video_path, "video")
            # conda-base FFMPEG does not preserve rotations properly, must read manually
            if video_idx is not None:
                vr_md = self.video_md[video_idx]
                vr_rotation = self.video_rotations[video_idx]
            else:
                try:
                    vr_rotation = -int(ffprobe(video_path).json['streams'][0]['side_data_list'][0].get('rotation', '0'))
                except:
                    vr_rotation = 0
                vr_md = vr.get_metadata()['video']
            vr_len = int(vr_md['duration'][0] * vr_md['fps'][0]) - 1
        return vr, vr_len, vr_md, vr_rotation

    def __len__(self):
        return self._dataset_len
    
    # @profile
    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
        video_idx = bisect(self.video_idx_start, idx) - 1
        vr, _, vr_md, vr_rotation = self.create_video_reader(None, video_idx, cpuid)
        i_s = self.sample_indices[idx]
        if self.delta_t is not None:
            delta_ts = np.random.randint(self.delta_t[0], self.delta_t[1]+1, size=len(i_s))
            i_s = np.array([index for i, delta_t in zip(i_s, delta_ts) for index in [i, i+delta_t]])
        sort_indexes = np.argsort(i_s).astype(np.int32)
        unsort_indexes = np.argsort(sort_indexes).astype(np.int32)
        if self.backend == 'decord':
            imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]
            vr.seek(0)
        elif 'torchvision' in self.backend:
            vr.seek(0)
            res = []
            i_s_ = [x / vr_md['fps'][0] for x in i_s[sort_indexes]]
            for i_ in i_s_:
                vr.seek(i_)
                count = 0
                while True:
                    try:
                        res.append(next(vr)['data'])
                        break
                    except StopIteration:
                        log.warning(f"StopIteration at {i_ + (count * 1/vr_md['fps'][0])} for {self.video_paths[idx]}")
                        count += 1
                        if count < 3:
                            vr.seek(i_ + count * 1/vr_md['fps'][0])
                        else:
                            log.warning(f"Failed to read frame for 3rd iteration, resorting to keyframe from {i_-1/vr_md['fps'][0]}")
                            vr.seek(i_-1/vr_md['fps'][0], keyframes_only=True)
                            decode_res = next(vr)
                            log.warn(f"Keyframe from {i_} read succesfully at {decode_res['pts']}")
                            res.append(decode_res['data'])
                            break
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

class WalkingToursDecodedDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform: Callable,
                 delta_t=None,
                 dataset_fraction=None,
                 ):
        
        self.data_dir = data_dir
        self.transform = transform
        self.delta_t = delta_t
        
        image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if dataset_fraction is not None:
            subset_size = int(len(image_paths) * dataset_fraction)
            image_paths = np.random.choice(image_paths, size=subset_size, replace=False)
        self.image_paths = image_paths

        num_pairs = len(image_paths)
        if delta_t is not None:
            num_pairs = num_pairs - delta_t[1]

        self._dataset_len = num_pairs

    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        try:
            img1 = Image.open(self.image_paths[idx])
            img2 = None
            if self.delta_t is not None:
                delta_t = np.random.randint(self.delta_t[0], self.delta_t[1]+1)
                img2 = Image.open(self.image_paths[idx + delta_t])
        
            return self.transform(img1, img2)
        except Exception as e:
            log.error(f"Error reading image pair {idx}: {e}")
            return self.__getitem__(np.random.randint(0, self._dataset_len))

class DoraWalkingToursDecodedDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform: Callable,
                 num_frames,
                 delta_t,
                 dataset_fraction=None,
                 ):
        
        self.data_dir = data_dir
        self.transform = transform
        self.num_frames = num_frames
        self.delta_t = delta_t
        
        image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if dataset_fraction is not None:
            subset_size = int(len(image_paths) * dataset_fraction)
            image_paths = np.random.choice(image_paths, size=subset_size, replace=False)
        self.image_paths = image_paths

        num_pairs = len(image_paths)
        num_pairs = num_pairs - (delta_t * num_frames)
        self._dataset_len = num_pairs

    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        try:
            imgs = []
            for i in range(self.num_frames):
                img = Image.open(self.image_paths[idx + i * self.delta_t])
                imgs.append(img)
            return self.transform(imgs)
        except Exception as e:
            log.error(f"Error reading image pair {idx}: {e}")
            return self.__getitem__(np.random.randint(0, self._dataset_len))
