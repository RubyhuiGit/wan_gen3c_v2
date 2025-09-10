# accelerate launch examples/wanvideo/model_training/train_gen3c.py \
#   --dataset_base_path /root/autodl-tmp/10_K \
#   --dataset_metadata_path /root/autodl-tmp/10_K/metadata.json \
#   --num_frames 21 \
#   --dataset_repeat 10 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-4 \
#   --num_epochs 2 \
#   --remove_prefix_in_ckpt "pipe.vace." \
#   --output_path "/root/autodl-tmp/output" \
#   --trainable_models "vace" \
#   --cache_index 0 20 \
#   --use_gradient_checkpointing_offload

# test
python3 examples/wanvideo/model_inference/test_gen3c.py 