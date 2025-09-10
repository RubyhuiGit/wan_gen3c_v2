import os
import tarfile
import argparse
import sys

def batch_unzip_tar(input_dir, output_dir=None):
    if not os.path.isdir(input_dir):
        print(f"error'{input_dir}' not exist")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    found_files = False
    for filename in os.listdir(input_dir):
        print(f"{filename} is proc........")
        if filename.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz')):
            found_files = True
            source_path = os.path.join(input_dir, filename)
            folder_name = filename.split('.')[0]
            dest_path = os.path.join(output_dir, folder_name)   
            os.makedirs(dest_path, exist_ok=True)
            try:
                with tarfile.open(source_path, 'r:*') as tar:
                    tar.extractall(path=dest_path)
                print(f"succ process {filename}")
            except Exception as e:
                print(f"error: {e}")

    if not found_files:
        print("\n no tar files")
    else:
        print("\n Done!")

if __name__ == '__main__':
    input_dir="/root/autodl-tmp/waymo_datasets"
    output_dir="/root/autodl-tmp/waymo_datasets_unzip"
    batch_unzip_tar(input_dir, output_dir)