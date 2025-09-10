import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_3dcache_test_pipeline import WanVideo3DCacheTestPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from diffsynth.models.utils import load_state_dict_from_safetensors

from data.dataset_10k import Dataset10KTestInfo
from data.vipe_dataset import VipeDatasetsTestInfo

# 屏蔽，不加载原来的vace
pipe = WanVideo3DCacheTestPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
# 加载训练好的vace
vace_state_dict = load_state_dict_from_safetensors("/root/autodl-tmp/output/epoch-0.safetensors")
pipe.vace_from_pretrained(vace_state_dict)
pipe.enable_vram_management()

use_waymo_datasets = False    # 切换数据
if not use_waymo_datasets:
    test_path = "/root/autodl-tmp/10_K/10K_1"
    cache_index = [0, 20]     # 要指定用来创建3d cache的frame_index
    test_info = Dataset10KTestInfo(test_path, cache_index, sample_frames_num=21, max_frame_num=21).get_data_dict()
else:
    test_path = "/root/autodl-tmp/waymo_datasets_vipe_output/segment-10072231702153043603_5725_000_5745_000_with_camera_labels"
    cache_index = [0, 20]     # 要指定用来创建3d cache的frame_index
    test_info = VipeDatasetsTestInfo(data_root=test_path, cache_index=cache_index, sample_frames_num=21).get_data_dict()


video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    dataset_info=test_info,
    seed=1, tiled=True
)
save_video(video, "/root/wan2.1_vace_1.3b.mp4", fps=15, quality=5)
