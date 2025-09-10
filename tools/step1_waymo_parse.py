import os
import argparse
import glob
import multiprocessing
import tensorflow as tf
import numpy as np
import cv2
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from waymo_open_dataset import dataset_pb2 as open_dataset


WAYMO_FRAME_RATE = 10

def save_camera_extrinsics_as_npy(calibrations, output_dir):
    """
    计算所有摄像头两两之间的外部参数，并将它们存储在一个字典中，
    最后将该字典保存为 .npy 文件。
    """
    extrinsics_path = os.path.join(output_dir, "camera_extrinsics.npy")
    
    # 1. 存储从车辆坐标系到各个摄像头坐标系的变换矩阵
    vehicle_to_cam_transforms = {}
    for calib in calibrations:
        T_vehicle_to_cam = np.array(calib.extrinsic.transform).reshape(4, 4)
        vehicle_to_cam_transforms[open_dataset.CameraName.Name.Name(calib.name)] = T_vehicle_to_cam

    # 2. 创建一个字典来存储摄像头之间的变换矩阵
    extrinsics_data = {}
    cam_names = list(vehicle_to_cam_transforms.keys())

    # 3. 遍历所有摄像头组合，计算变换矩阵
    for cam1_name, cam2_name in combinations(cam_names, 2):
        # 获取从车辆到各摄像头的变换
        T_v_c1 = vehicle_to_cam_transforms[cam1_name]
        T_v_c2 = vehicle_to_cam_transforms[cam2_name]
        
        # 计算从 c2 到 v 的变换，即 T_v_c2 的逆
        T_c2_v = np.linalg.inv(T_v_c2)
        
        # 计算从 c1 到 c2 的变换: T_c2_c1 = T_c2_v * T_v_c1
        # 这个矩阵可以将一个在 cam1 坐标系下的点，变换到 cam2 坐标系下
        T_c2_c1 = T_c2_v @ T_v_c1
        
        # 计算反向变换
        T_c1_c2 = np.linalg.inv(T_c2_c1)

        # 使用清晰的key存入字典
        extrinsics_data[f"{cam1_name}_to_{cam2_name}"] = T_c2_c1
        extrinsics_data[f"{cam2_name}_to_{cam1_name}"] = T_c1_c2
            
    # 4. 将包含所有变换矩阵的字典保存为 .npy 文件
    # allow_pickle=True 是保存包含非数值对象的字典所必需的
    np.save(extrinsics_path, extrinsics_data, allow_pickle=True)

def process_waymo_segment(tfrecord_path, output_dir_base):
    try:
        segment_name = os.path.splitext(os.path.basename(tfrecord_path))[0]
        output_dir = os.path.join(output_dir_base, segment_name)
        os.makedirs(output_dir, exist_ok=True)
        
        log_prefix = f"[{segment_name}]"

        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

        video_writers = {}
        is_first_frame = True
        frame_count = 0

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frame_count += 1

            if is_first_frame:
                camera_names = sorted([open_dataset.CameraName.Name.Name(c.name) for c in frame.images])
                print(f"{log_prefix} find camera: {', '.join(camera_names)}")

                save_camera_extrinsics_as_npy(frame.context.camera_calibrations, output_dir)
                print(f"{log_prefix} calibration params saved")

                for camera_image in frame.images:
                    cam_name_str = open_dataset.CameraName.Name.Name(camera_image.name)
                    image_tensor = tf.image.decode_jpeg(camera_image.image)
                    height, width, _ = image_tensor.shape
                    
                    video_path = os.path.join(output_dir, f"{cam_name_str}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writers[cam_name_str] = cv2.VideoWriter(video_path, fourcc, WAYMO_FRAME_RATE, (width, height))
                is_first_frame = False

            for camera_image in frame.images:
                cam_name_str = open_dataset.CameraName.Name.Name(camera_image.name)
                image_tensor = tf.image.decode_jpeg(camera_image.image)
                image_np = image_tensor.numpy()
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                if cam_name_str in video_writers:
                    video_writers[cam_name_str].write(image_bgr)

        for writer in video_writers.values():
            writer.release()

        return f"{log_prefix} (total {frame_count} frame) parse succ, save in {output_dir}"

    except Exception as e:
        return f"[{os.path.basename(tfrecord_path)}] parse failed {e}"

def main():
    input_dir = "/root/autodl-tmp/waymo_datasets_unzip"
    output_dir = "/root/autodl-tmp/waymo_datasets_parse"
    num_workers = 8

    if not os.path.exists(input_dir):
        print(f"Error Path -> {input_dir}")
        return

    tfrecord_files = []
    tfrecord_files = sorted(glob.glob(os.path.join(input_dir, '*/*.tfrecord')))
    print(f"Total {len(tfrecord_files)} files find!")


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_waymo_segment, path, output_dir): path for path in tfrecord_files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"[{i+1}/{len(tfrecord_files)}] {result}")

if __name__ == '__main__':
    main()
