import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_3dcache_train_pipeline import WanVideo3DCacheTrainPipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data.dataset_10k import Dataset10K

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideo3DCacheTrainPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["rgb_frames"],          # 21, 480, 640, 3
            "input_w2c": data["w2cs"],                  # 21, 4, 4
            "input_intrinsics": data["intrinsics"],     # 21, 3, 3
            "input_depth": data["depths"],              # 21, 480, 640
            "height": data["rgb_frames"][0].shape[0],
            "width": data["rgb_frames"][0].shape[1],
            "num_frames": len(data["rgb_frames"]),
            "cache_index": data["cache_index"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # 下面都不填充
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["rgb_frames"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["rgb_frames"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            # print(f"Processing unit: {unit.__class__.__name__}")
            # WanVideoUnit_ShapeChecker             480 640 21
            # WanVideoUnit_3DCacheRender            1, 64, 6, 60, 80
            # WanVideoUnit_NoiseInitializer         torch.Size([1, 16, 6, 60, 80])
            # WanVideoUnit_PromptEmbedder           # 文本 （1，512， 4096）
            # WanVideoUnit_S2V                      # 不进入
            # WanVideoUnit_InputVideoEmbedder       torch.Size([1, 16, 6, 60, 80]) torch.Size([1, 16, 6, 60, 80])
            # WanVideoUnit_ImageEmbedderVAE         # 不进入
            # WanVideoUnit_ImageEmbedderCLIP        # 不进入
            # WanVideoUnit_ImageEmbedderFused       # 不进入
            # WanVideoUnit_FunControl               # 不进入
            # WanVideoUnit_FunReference             # 不进入
            # WanVideoUnit_FunCameraControl         # 不进入
            # WanVideoUnit_SpeedControl             # 不进入
            # WanVideoUnit_UnifiedSequenceParallel  # 不用管
            # WanVideoUnit_TeaCache                 # 不用管
            # WanVideoUnit_CfgMerger                # 不用管
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    if args.use_waymo_datasets:
        raise NotImplementedError("Waymo dataset is not implemented yet.")
    else:
        dataset = Dataset10K(args.dataset_base_path,
                             args.num_frames,
                             args.cache_index,
                             args.dataset_repeat)
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
