import os
import glob
from typing import List
import re
from tqdm import tqdm
from pathlib import Path
import pickle
import random

def get_file_paths(input_path):
    """
    Recursively fetches all .bin file paths and their corresponding .label file paths.
    
    Args:
        input_path (str): The root directory to search within.
        
    Returns:
        list of tuples: List containing tuples of (.bin file path, .label file path).
    """
    bin_files = glob.glob(os.path.join(input_path, '**', 'velodyne', '*.bin'), recursive=True)
    file_pairs = []

    for bin_file in tqdm(bin_files, desc="Getting Filepaths"):
        label_file = bin_file.replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            file_pairs.append((bin_file, label_file))

    return file_pairs

def split_data(file_pairs, split_ratio, shuffle=False):
    """
    Splits the list of file pairs into training and testing sets based on the specified ratio.
    
    Args:
        file_pairs (list of tuples): List containing tuples of (.bin file path, .label file path).
        split_ratio (float): Ratio to split the data into training and testing sets.
        
    Returns:
        tuple: Two lists - training set and testing set.
    """
    if shuffle:
        random.shuffle(file_pairs)
    split_index = int(len(file_pairs) * split_ratio)
    train_list = file_pairs[:split_index]
    test_list = file_pairs[split_index:]
    return train_list, test_list

def split_at_substring(s, substring):
    parts = s.split(substring, 1)
    
    # check if the substring is found and if there are two parts
    if len(parts) == 2:
        # return the second part
        return substring + parts[1]
    else:
        # raise an error if substring is not found
        raise ValueError("No substring found") 

def get_info(filepath_list) -> dict:
    data_infos = dict()
    data_infos['metainfo'] = dict(dataset='TargetAmbient')
    data_list = []
    for cloud_path, label_path in tqdm(filepath_list, desc="Adding Filepaths to Metainfo"):
        match = re.search(r'/(\d{2})/velodyne/cloud_(\d+)\.bin', cloud_path)
        if match:
            directory_number = match.group(1)
            file_number = match.group(2)
        else:
            raise ValueError("Path does not match the expected format")
        data_list.append({
            'lidar_points': {
                'lidar_path': split_at_substring(cloud_path, "sequences"),
                'num_pts_feats': 5
            },
            'pts_semantic_mask_path': split_at_substring(label_path, "sequences"),
            'sample_idx': str(directory_number).zfill(2) + str(file_number).zfill(4)
        })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_info_files(pkl_prefix: str, save_path: str, file_pairs, train_pairs, test_pairs) -> None:
    print('Generate info.')
    save_path = Path(save_path)

    print("Generating Training Infos")
    infos_train = get_info(train_pairs)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Target info train file is saved to {filename}')
    with open(filename, 'wb') as f:
            pickle.dump(infos_train, f)

    print("Generating Val Infos")
    infos_val = get_info(test_pairs)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Target info val file is saved to {filename}')
    with open(filename, 'wb') as f:
            pickle.dump(infos_val, f)

    print("Generating TrainVal Infos")
    infos_trainval = get_info(file_pairs)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Target info trainval file is saved to {filename}')
    with open(filename, 'wb') as f:
            pickle.dump(infos_trainval, f)
    
    # print("Generating Testing Infos")
    # infos_test = infos_val
    # filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    # print(f'Target info test file is saved to {filename}')
    # with open(filename, 'wb') as f:
    #         pickle.dump(infos_test, f)

def main(input_path, save_path, split_ratio=0.75, shuffle=False):
    """
    Main function to execute the script.
    
    Args:
        input_path (str): The root directory to search within.
        save_path (str): The directory to save the info files in.
        split_ratio (float, optional): Ratio to split the data into training and testing sets. Default is 0.75.
    """
    # Get all file paths
    file_pairs = get_file_paths(input_path)
    
    # Split the data based on index
    train_list, test_list = split_data(file_pairs, split_ratio)
    
    print(f"Total files: {len(file_pairs)}")
    print(f"Train files: {len(train_list)}")
    print(f"Test files: {len(test_list)}")
    
    # For debugging purposes
    # print(train_list[:5])  # Print first 5 train pairs
    # print(test_list[:5])   # Print first 5 test pairs

    create_info_files("target", save_path, file_pairs, train_list, test_list)

if __name__ == "__main__":
    # input_path = '/mnt/d/all_pcd/kitti_format/sequences'
    # save_path = '/mnt/d/all_pcd/kitti_format'
    # shuffle = False

    input_path = '/mnt/d/target/kitti_format/sequences'
    save_path = '/mnt/d/target/kitti_format'
    shuffle = True

    main(input_path, save_path, split_ratio=0.75, shuffle=shuffle)