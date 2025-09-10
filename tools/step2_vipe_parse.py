import subprocess
import os
import sys
import argparse
import shutil

def copy_files(source_dir, dest_dir, filenames):
    for filename in filenames:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)       
        if os.path.isfile(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                print(f"  Copied: {filename}")
            except Exception as e:
                print(f"  Error copying {filename}: {e}")
        else:
            print(f" Warning: File not found and was not copied: {source_path}")

def run_vipe_inference(video_path, output_dir):
    if not os.path.isfile(video_path):
        print(f"error video path {video_path}")
        sys.exit(1) # 退出程序，返回错误码 1

    try:
        subprocess.run(cmd_check, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("error: 'vipe' command not found.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"vipe results save in {os.path.abspath(output_dir)}")
    
    command = [
        "vipe", 
        "infer", 
        video_path, 
        "--output", 
        output_dir
    ]

    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='')
        
        if proc.returncode != 0:
            sys.exit(1)
            
        print("Done!")
    except Exception as e:
        print("Error during vipe inference:", e)
        sys.exit(1)


if __name__ == "__main__":

    root_dir = "/root/autodl-tmp/waymo_datasets_parse"
    save_dir = "/root/autodl-tmp/waymo_datasets_vipe_output1"
    os.makedirs(save_dir, exist_ok=True)

    all_datasetsdirs = os.listdir(root_dir)
    all_datasetsdirs.sort()

    proc_index = 0
    for dir_path in all_datasetsdirs:
        proc_index += 1
        print(f"processing {proc_index}/{len(all_datasetsdirs)}: {dir_path}")
        
        if not dir_path.startswith("segment"):
            continue

        output_dir = os.path.join(save_dir, dir_path)
        os.makedirs(output_dir, exist_ok=True)

        files_to_copy=["camera_extrinsics.npy"]
        copy_files(os.path.join(root_dir, dir_path), output_dir, files_to_copy)
        
        seg_name = dir_path
        front_video_path = os.path.join(root_dir, dir_path, "FRONT.mp4")
        side_left_video_path = os.path.join(root_dir, dir_path, "SIDE_LEFT.mp4")
        side_right_video_path = os.path.join(root_dir, dir_path, "SIDE_RIGHT.mp4")
          
        cmd_check = ["which", "vipe"]
        
        run_vipe_inference(front_video_path, os.path.join(output_dir, "front"))
        run_vipe_inference(side_left_video_path, os.path.join(output_dir, "side_left"))
        run_vipe_inference(side_right_video_path, os.path.join(output_dir, "side_right"))

    print("Done all!")