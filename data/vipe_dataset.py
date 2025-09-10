
import torch
import numpy as np
import os
import json
import random
import torchvision.transforms as transforms
from PIL import Image
from .data_utils.vipe_data_parse import load_vipe_data

class VipeDatasets(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root=None,
        video_sample_n_frames=16,
        cache_index=None,
        repeat_num=100,
        read_num_thread=1,
        text_drop_ratio=0.1,
    ):  
        self.data_root = data_root
        json_file_path = os.path.join(data_root, "metadata.json")
        self.dataset = json.load(open(json_file_path))                           # 加载json文件

        print("Load Dataset From:", json_file_path)
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")                                      # 数据集大小

        self.text_drop_ratio = text_drop_ratio

        self.video_sample_n_frames  = video_sample_n_frames      # 期待的帧数
        self.cache_index = cache_index
        self.repeat = repeat_num

        self.read_num_thread = read_num_thread
        self.load_from_cache = False

    def get_batch(self, idx):
        json_data_info = self.dataset[idx % len(self.dataset)]
        dataset_dir = os.path.join(self.data_root, json_data_info['file_path'])

        front_dataset_dir = os.path.join(dataset_dir, "front")
        side_left_dataset_dir = os.path.join(dataset_dir, "side_left")
        side_right_dataset_dir = os.path.join(dataset_dir, "side_right")
        camera_entrinsics_cailbration_file = os.path.join(dataset_dir, "camera_extrinsics.npy")

        if not os.path.exists(front_dataset_dir) or \
           not os.path.exists(side_left_dataset_dir) or \
           not os.path.exists(side_right_dataset_dir) or \
           not os.path.exists(camera_entrinsics_cailbration_file):
            print("File Not Exist:", dataset_dir)
            raise ValueError(f"File Not Exist, Check you dataset {dataset_dir}")
        
        front_info_pack = load_vipe_data(front_dataset_dir, 
                                    starting_frame_idx=0,
                                    resize_hw=(480, 720),
                                    crop_hw=(480, 720),
                                    num_frames=self.video_sample_n_frames,
                                    read_mask=False,
                                    num_thread=self.read_num_thread,
                                    video_idx=0)
        front_rgb, front_depth, front_mask, front_w2c, front_c2w, front_intrinsics = front_info_pack
    
        frame_info = {}
        frame_info["rgb_frames"] = np.uint8((front_rgb * 255.0).permute(0, 2, 3, 1))
        frame_info["depths"] = front_depth[:, 0, :, :].numpy()                                   # (N, H, W)    float32
        frame_info["masks"] = front_mask.numpy()
        frame_info["w2cs"] = front_w2c.numpy()                                                   # (N, 4, 4)    float32
        frame_info["intrinsics"] = front_intrinsics.numpy()                                      # (N, 3, 3)    float32

        text = json_data_info.get('text', '')
        if random.random() < self.text_drop_ratio:
            text = ''

        valid_cache_index = []
        for cache_i in self.cache_index:
            if cache_i >= frame_info["rgb_frames"].shape[0]:
                valid_cache_index.append(frame_info["rgb_frames"].shape[0]-1)
            elif cache_i < 0:
                valid_cache_index.append(0)
            else:
                valid_cache_index.append(cache_i)
        frame_info["cache_index"] = valid_cache_index

        return frame_info, text, 'video', front_dataset_dir

    def __len__(self):
        return self.length * self.repeat

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                data_info, text, file_type, dataset_dir = self.get_batch(idx)
                sample = data_info
                sample["prompt"] = text
                sample["data_type"] = file_type
                sample["idx"] = idx
                sample["file_name"] = dataset_dir
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)
        return sample

    
class VipeDatasetsTestInfo():
    def __init__(
        self,
        data_root=None,
        cache_index=None,
        sample_frames_num=81,
        read_num_thread=4
    ):  
        self.data_root = data_root
        self.cache_index = cache_index
        self.sample_frames_num = sample_frames_num
        self.read_num_thread = read_num_thread


    def get_data_dict(self):

        front_dataset_dir = os.path.join(self.data_root, "front")
        front_info_pack = load_vipe_data(front_dataset_dir, 
                                         starting_frame_idx=0,
                                         resize_hw=(480, 720),
                                         crop_hw=(480, 720),
                                         num_frames=self.sample_frames_num,
                                         read_mask=False,
                                         num_thread=self.read_num_thread,
                                         video_idx=0)
        
        front_rgb, front_depth, front_mask, front_w2c, front_c2w, front_intrinsics = front_info_pack
    
        data_info = {} 
        data_info["rgb_frames"] = np.uint8((front_rgb * 255.0).permute(0, 2, 3, 1))             # (N, s)
        data_info["depths"] = front_depth[:, 0, :, :].numpy()                                   # (N, H, W)    float32
        data_info["masks"] = front_mask.numpy()
        data_info["w2cs"] = front_w2c.numpy()                                                   # (N, 4, 4)    float32
        data_info["intrinsics"] = front_intrinsics.numpy()                                      # (N, 3, 3)    float32

        # 填充信息
        frame_info = {}
        frame_info["target_w2c"] = data_info["w2cs"]
        frame_info["target_intrinsic"] = data_info["intrinsics"]

        cache_frames = [data_info["rgb_frames"][i] for i in self.cache_index]
        cache_w2cs = [data_info["w2cs"][i] for i in self.cache_index]
        cache_intrinsics = [data_info["intrinsics"][i] for i in self.cache_index]
        cache_depths = [data_info["depths"][i] for i in self.cache_index]

        frame_info["cache_frames"] = np.stack(cache_frames, axis=0)
        frame_info["cache_w2cs"] = np.stack(cache_w2cs, axis=0)
        frame_info["cache_intrinsics"] = np.stack(cache_intrinsics, axis=0)
        frame_info["cache_depths"] = np.stack(cache_depths, axis=0)
        frame_info["first_frame"] = data_info["rgb_frames"][0]

        N, H, W, _ = data_info["rgb_frames"].shape
        frame_info["num_frames"] = N
        frame_info["height"] = H
        frame_info["width"] = W

        frame_info["cache_index"] = self.cache_index
        return frame_info
