import os
import numpy as np
import pypcd4
from pypcd4 import PointCloud, MetaData
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_directory(root, files, input_root, output_root):
    for file in files:
        if file.endswith(".pcd"):
            input_pcd_path = os.path.join(root, file)
            
            # Load PCD file
            pcd = PointCloud.from_path(input_pcd_path)
            
            # Extracting the points as numpy array
            cloud = pcd.numpy()[:,[0,1,2,3,7]].astype(np.float32)

            # Split into a x, y, z, intensity, 0 cloud and ambient labels
            ambient = cloud[:, -1]
            cloud[:, -1] = 0.0

            # Define the output path with the 'velodyne' subfolder
            relative_path = os.path.relpath(root, input_root)
            velodyne_subfolder = os.path.join(output_root, relative_path, 'velodyne')
            labels_subfolder = os.path.join(output_root, relative_path, 'labels')
            os.makedirs(velodyne_subfolder, exist_ok=True)
            os.makedirs(labels_subfolder, exist_ok=True)
            output_bin_path = os.path.join(velodyne_subfolder, file.replace(".pcd", ".bin"))
            output_label_path = os.path.join(labels_subfolder, file.replace(".pcd", ".label"))

            # Save as .bin file
            cloud = cloud.reshape((cloud.shape[0] * cloud.shape[1]))
            cloud.tofile(output_bin_path)
            ambient.tofile(output_label_path)
            # print(f"Converted and saved: {output_bin_path}")

def convert_pcd_to_bin(input_root, output_root):
    directories = []
    for root, _, files in os.walk(input_root):
        directories.append((root, files, input_root, output_root))
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_directory, *args) for args in directories]

        # Initialize the tqdm progress bar
        with tqdm(total=len(futures), desc="Converting PCD to BIN") as pbar:
            for future in as_completed(futures):
                future.result()  # Ensure the task is complete
                pbar.update(1)  # Update the progress bar

if __name__ == "__main__":
    input_root = "/mnt/d/all_pcd/target_format/all_pcd"
    output_root = "/mnt/d/all_pcd/kitti_format/sequences"
    convert_pcd_to_bin(input_root, output_root)