import os
import json

def create_json_from_folders(root_folder, output_filename="output.json"):
    if not os.path.isdir(root_folder):
        print(f"'{root_folder}' not exists or is not a directory.")
        return
    data_list = []
    try:
        entries = os.listdir(root_folder)
        entries.sort()
    except OSError as e:
        print(f"错误：无法访问文件夹 '{root_folder}'. {e}")
        return

    for entry_name in entries:
        full_path = os.path.join(root_folder, entry_name)
        if os.path.isdir(full_path):
            record = {
                "file_path": entry_name,
                "text": "Normal video",
                "type": "video"
            }
            data_list.append(record)

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4)

if __name__ == "__main__":
    target_folder = '/root/autodl-tmp/waymo_datasets_vipe_output' 
    output_file = os.path.join(target_folder, 'metadata.json')
    create_json_from_folders(target_folder, output_file)