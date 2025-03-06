import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import numpy as np
import pickle
import argparse
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2
import multiprocessing as mp

class WaymoToSemanticKITTI(object):
    """Waymo to SemanticKITTI converter.
    This class serves as the converter to change the waymo raw data to SemanticKITTI format.
    Args:

    """

    def __init__(self, load_dir, save_dir, num_proc, keep_second_return=False):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.num_proc = int(num_proc)
        self.keep_second_return = keep_second_return

        # Print settings
        print("Loading data from: ", self.load_dir)
        print("Saving data to: ", self.save_dir)
        print("Num processesors used: ", self.num_proc)
        print("Keeping second return: ", self.keep_second_return)

    def convert_range_image_to_point_cloud_labels(self, frame, range_images, segmentation_labels, ri_index=0):
        """Convert segmentation labels from range images to point clouds.

        Args:
            frame: open dataset frame
            range_images: A dict of {laser_name, [range_image_first_return,
            range_image_second_return]}.
            segmentation_labels: A dict of {laser_name, [range_image_first_return,
            range_image_second_return]}.
            ri_index: 0 for the first return, 1 for the second return.

        Returns:
            point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
            points that are not labeled.
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        # print(calibrations)
        point_labels = []
        # extract just the top lidar
        c = calibrations[0]
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
        return point_labels        

    def create_lidar(self, frame):
        """Parse and save the lidar data in psd format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
        """
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        # Get 3d points in vehicle frame.
        # Get first returns 
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, keep_polar_features=True)
        # extract just the top lidar in body frame
        points_all = points[0]

        # Get second returns
        if (self.keep_second_return):
            points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True)
            # extract second returns from the top lidar in body frame
            points_all_ri2 = points_ri2[0]
            # Merge with 1st returns
            points_all = np.concatenate([points_all, points_all_ri2], axis=0)
        
        # get the lidar to body transformation
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        c = calibrations[0]
        trans = np.array(c.extrinsic.transform).reshape((4,4))
        # beam_inclinations = np.array(c.beam_inclinations)
        
        # convert points to lidar frame
        ones = np.ones((points_all.shape[0], 1))
        homogeneous_points = np.hstack([points_all[:, 3:6], ones])
        inverse_trans = np.linalg.inv(trans)
        transformed_points =  homogeneous_points @ inverse_trans.T

        # combine with intensities
        velodyne = np.c_[transformed_points[:,:3], points_all[:,1]]
        velodyne = velodyne.reshape((velodyne.shape[0] * velodyne.shape[1]))
        
        return velodyne

    
    def create_label(self, frame):
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        point_labels = self.convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels)
        point_labels_all = np.concatenate(point_labels, axis=0)

        # Get second returns
        if (self.keep_second_return):
            point_labels_ri2 = self.convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels, ri_index=1)
            point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
            point_labels_all = np.concatenate([point_labels_all, point_labels_all_ri2], axis=0)

        labels = point_labels_all

        # change -1s to 0s
        labels = np.where(labels < 0, 0, labels)

        # convert to semantic kitti format
        labels = (labels[:, 0].astype(np.uint32) << 16) | labels[:, 1].astype(np.uint32)

        return labels
    
    
    def process(self, start_idx, filepaths):

        file_name = self.files[start_idx]
        data_dir = self.data_dir_list[start_idx]
        dir_idx = self.dir_idx[start_idx]

        split = data_dir.split("/")[-1]
        waymo_file_path = os.path.join(data_dir, file_name)

        sub_dir = os.path.join(self.save_dir, "sequences", dir_idx)
        if not os.path.exists(sub_dir):
            os.makedirs(os.path.join(sub_dir, 'velodyne'))
            if split != 'testing':
                os.mkdir(os.path.join(sub_dir, 'labels'))

        print("Processing ", waymo_file_path)
        dataset = tf.data.TFRecordDataset(waymo_file_path, compression_type='')
        count = 0
        paths = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            has_label = bool(frame.lasers[0].ri_return1.segmentation_label_compressed)
            if has_label or split == 'testing':
                file_idx = "0" * (3 - len(str(count))) + str(count)
                print("processing frame (has_label=", has_label, "): ", dir_idx+file_idx)

                # process point cloud
                lidar_save_path = os.path.join(self.save_dir, "sequences", dir_idx, 'velodyne', dir_idx+file_idx+'.bin')
                point_cloud = self.create_lidar(frame)
                point_cloud.astype(np.float32).tofile(lidar_save_path)
                paths.append(dir_idx+file_idx+".bin")

                # process label
                if (split != 'testing'):
                    label_save_path = os.path.join(self.save_dir, "sequences", dir_idx, 'labels', dir_idx+file_idx+'.label')
                    label = self.create_label(frame)
                    label.tofile(label_save_path)
                count += 1
        filepaths[str(dir_idx)] = paths

    def convert_all(self):
        datasets = ['training', 'validation']

        self.files = []
        self.data_dir_list = []
        self.dir_idx = []
        for dataset_type in datasets:
            print("Converting " + dataset_type + " set") 
            data_dir = os.path.join(self.load_dir, dataset_type)
            
            files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.tfrecord')]
        
            self.files.extend(files)
            self.data_dir_list.extend([data_dir] * len(files))
            dataset_idx = str(datasets.index(dataset_type))
            self.dir_idx.extend([(dataset_idx + "0" * (3 - len(str(i))) + str(i)) for i in range(len(files))])
        
        # filepaths = {item: [] for item in self.dir_idx}
        # self.process(0, filepaths)

        # Create dictionary of empty lists for storing filepaths
        with mp.Manager() as manager:
            filepaths = manager.dict({item: [] for item in self.dir_idx})

            # Pass the shared dictionary to the processes
            with mp.Pool(processes=min(mp.cpu_count(), self.num_proc)) as p:
                p.starmap(self.process, [(i, filepaths) for i in range(len(self.files))])

            self.filepaths = dict(filepaths)
        self.create_info_file("waymo", self.save_dir)

        return True
    
    def get_meta_info(self, split: str) -> dict:
        data_infos = dict()
        data_infos['metainfo'] = dict(dataset='Waymo')
        data_list = []

        if (split == "train"):
            dataset_val = [0]
        elif (split == "val"):
            dataset_val = [1]
        elif (split == "trainval"):
            dataset_val = [0, 1]
        elif (split == "test"):
            dataset_val = [2]
        else:
            print("Invalid split!!!!!")
            return None
        
        for key in self.filepaths:
            if any(str(key).startswith(str(val)) for val in dataset_val):
                for filepath in self.filepaths[key]:
                    data_list.append({
                        'lidar_points': {
                            'lidar_path':
                            os.path.join('sequences', str(key), 'velodyne', filepath),
                            'num_pts_feats':
                            4
                        },
                        'pts_semantic_mask_path': 
                            os.path.join('sequences', key, 'labels', filepath[:-4] + '.label'),
                        'sample_idx': filepath[:-4]
                    })
        data_infos.update(dict(data_list=data_list))
        return data_infos

    def create_info_file(self, pkl_prefix: str, save_path: str) -> None:
        print('Generating infos.')

        infos_train = self.get_meta_info(split='train')
        filename = os.path.join(save_path, f'{pkl_prefix}_infos_train.pkl')
        print(f'Waymo info train file is saved to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(infos_train, f)

        infos_val = self.get_meta_info(split='val')
        filename = os.path.join(save_path, f'{pkl_prefix}_infos_val.pkl')
        print(f'Waymo info val file is saved to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(infos_val, f)

        infos_trainval = self.get_meta_info(split='trainval')
        filename = os.path.join(save_path, f'{pkl_prefix}_infos_trainval.pkl')
        print(f'Waymo info trainval file is saved to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(infos_trainval, f)

        # infos_test = self.get_meta_info(split='test')
        # filename = os.path.join(save_path, f'{pkl_prefix}_infos_test.pkl')
        # print(f'Waymo info test file is saved to {filename}')
        # with open(filename, 'wb') as f:
        #     pickle.dump(infos_test, f)
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir', help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('save_dir', help='Directory to save converted KITTI-format data')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    parser.add_argument('--keep_2nd_return', default=False, help='Keep second point returns in dataset')
    args = parser.parse_args()

    converter = WaymoToSemanticKITTI(args.load_dir, args.save_dir, args.num_proc, args.keep_2nd_return)
    converter.convert_all()