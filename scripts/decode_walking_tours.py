import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import gc
import json
import subprocess
from typing import NamedTuple

from PIL import Image
import torch
import torchvision
from torchvision.io import VideoReader as TVVideoReader

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

def process_segment(video_path: str, out_dir: str, start_frame: int, end_frame: int, device_id: int) -> None:
    """
    Process a segment of the video by decoding frames from start_frame (inclusive)
    to end_frame (exclusive) and saving them as PNGs using torchvision VideoReader.
    
    Args:
        video_path (str): Path to the video file.
        out_dir (str): Directory where PNGs are saved.
        start_frame (int): Absolute frame index to start decoding.
        end_frame (int): Absolute frame index to stop decoding.
        device_id (int): The device ID (unused for torchvision but kept for compatibility).
    """
    try:
        torchvision.set_video_backend('video_reader')
        vr = TVVideoReader(video_path, "video")
        vr_md = vr.get_metadata()['video']
        fps = vr_md['fps'][0]
        
        for frame_idx in range(start_frame, end_frame):
            filename = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
            if os.path.exists(filename):
                continue
                
            timestamp = frame_idx / fps
            vr.seek(timestamp)
            
            try:
                frame_data = next(vr)['data']
                frame_np = frame_data.permute(1, 2, 0).numpy()
                image = Image.fromarray(frame_np)
                image.save(filename)
            except StopIteration:
                print(f"Warning: Could not read frame {frame_idx} at timestamp {timestamp}")
                continue
                
            if (frame_idx - start_frame) % 50 == 0:
                print(f'Device {device_id}: Processed {frame_idx - start_frame} frames')
                
    except Exception as e:
        print(f"Error in process_segment on device {device_id} for frames {start_frame}-{end_frame}: {e}")
        traceback.print_exc()
        raise

def process_video(video_path: str, out_dir: str, num_workers: int = 4) -> None:
    """
    Splits the video into segments based on the total frame count,
    then decodes and saves each segment in parallel using torchvision VideoReader.
    
    Args:
        video_path (str): Path to the video file.
        out_dir (str): Directory to save PNG frames.
        num_workers (int): Number of parallel workers (and segments).
    """
    os.makedirs(out_dir, exist_ok=True)
    
    torchvision.set_video_backend('video_reader')
    vr = TVVideoReader(video_path, "video")
    vr_md = vr.get_metadata()['video']
    total_frames = int(vr_md['duration'][0] * vr_md['fps'][0]) - 1
    
    segment_length = total_frames // num_workers
    segments = []
    for i in range(num_workers):
        start_frame = i * segment_length
        end_frame = total_frames if i == num_workers - 1 else (i + 1) * segment_length
        segments.append((start_frame, end_frame, i))
    
    print(f"Total frames: {total_frames}")
    print(f"Segments (frame ranges with device ids): {segments}")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for start_frame, end_frame, device_id in segments:
            futures.append(
                executor.submit(process_segment, video_path, out_dir, start_frame, end_frame, device_id)
            )
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode a video into PNG frames by splitting it into segments and processing in parallel."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("out_dir", type=str, help="Directory to save PNG frames.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel CPU workers.")
    
    args = parser.parse_args()
    process_video(args.video_path, args.out_dir, args.num_workers)
