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
            
            # Get the indices of the needed fields
            fields = ['x', 'y', 'z', 'intensity', 'ambient', 'label']
            indices = [pcd.fields.index(field) for field in fields]

            # Extracting the points as numpy array
            cloud = pcd.numpy()[:,indices].astype(np.float32)

            # Split into a [x, y, z, intensity, ambient], and labels
            points = cloud[:, :5]
            labels = cloud[:, 5]

            # Define the output path with the 'velodyne' subfolder
            relative_path = os.path.relpath(root, input_root)
            velodyne_subfolder = os.path.join(output_root, relative_path, 'velodyne')
            labels_subfolder = os.path.join(output_root, relative_path, 'labels')
            os.makedirs(velodyne_subfolder, exist_ok=True)
            os.makedirs(labels_subfolder, exist_ok=True)
            output_bin_path = os.path.join(velodyne_subfolder, file.replace(".pcd", ".bin"))
            output_label_path = os.path.join(labels_subfolder, file.replace(".pcd", ".label"))

            # Save points as .bin file
            cloud = points.reshape((points.shape[0] * points.shape[1]))
            cloud.tofile(output_bin_path)

            # Save labels as .label files
            labels = (labels.astype(np.uint32) << 16) | labels.astype(np.uint32)
            labels.tofile(output_label_path)

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
    input_root = "/mnt/d/target/target_format"
    output_root = "/mnt/d/target/kitti_format/sequences"
    convert_pcd_to_bin(input_root, output_root)