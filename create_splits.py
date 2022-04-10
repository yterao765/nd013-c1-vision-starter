import argparse
import glob
import os
import shutil
import random

import numpy as np

from utils import get_dataset
from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    tfrecords = glob.glob(os.path.join(source, "*.tfrecord"))
    stats = []
    for i in range(len(tfrecords)):
        record = tfrecords[i]
        dataset = get_dataset(record)
        dataset = dataset.shuffle(buffer_size=1000, seed=42).take(100)
        
        stats.append(get_stats(dataset, os.path.basename(record)))
    stats.sort(key=lambda x:x['N_object'])

    filename = [stat['filename'] for stat in stats]
    N_obj = [stat['N_object'] for stat in stats]
    N_vehicle = [stat['N_vehicle'] for stat in stats]
    N_pedestrian = [stat['N_pedestrian'] for stat in stats]
    N_cyclist = [stat['N_cyclist'] for stat in stats]

    # Focus on the tfrecord containing cyclist images first.
    nonzero_cyclist_idx = [i for i, N in enumerate(N_cyclist) if N > 0]
    nonzero_cyclist_N = [N for N in N_cyclist if N > 0]
    nonzero_cyclist_filename = [name for i, name in enumerate(filename) if i in nonzero_cyclist_idx]

    # Split the entire dataset into training, validation, test sets.
    # Target ratio is 0.75:0.15:0.10 between the three subsets with respect to the number of objects rather than the number of tfrecord files.
    # Assume that the number of images contained in each tfrecord is constant.
    train_files = []
    valid_files = []
    test_files = []
    fraction = {'train':0.75, 'valid':0.15, 'test':0.10}

    # Ratios of individual classes between the subsets should be similar to the entire ratio.
    # Count approximate total number of cyclists contained over all of the tfrecords, then calculate the number each subset should contain.
    total_cyclist = sum(nonzero_cyclist_N)
    train_cyclist_target = total_cyclist * fraction['train']
    valid_cyclist_target = total_cyclist * fraction['valid']
    test_cyclist_target = total_cyclist * fraction['test']

    # Assign tfrecords to the test set until the number of cyclists becomes larger than the target.
    # Then peform the same operation for the validation set. Remaining files are all assigned to the training set.
    train_cyclist = 0
    valid_cyclist = 0
    test_cyclist = 0
    idx_list = [i for i, _ in enumerate(nonzero_cyclist_filename)]
    while test_cyclist < test_cyclist_target:
        idx = random.choice(idx_list)
        test_files.append(nonzero_cyclist_filename[idx])
        test_cyclist += nonzero_cyclist_N[idx]
        idx_list.remove(idx)
    while valid_cyclist < valid_cyclist_target:
        idx = random.choice(idx_list)
        valid_files.append(nonzero_cyclist_filename[idx])
        valid_cyclist += nonzero_cyclist_N[idx]
        idx_list.remove(idx)
    for idx in idx_list:
        train_cyclist += nonzero_cyclist_N[idx]
        train_files.append(nonzero_cyclist_filename[idx])

    # Do the same thing above for the pedestrian class.
    # Don't consider pedestrians in the files already assigned to one of the subsets for simplicity.
    pedestrian_N = [N for i, N in enumerate(N_pedestrian) if i not in nonzero_cyclist_idx]
    pedestrian_filename = [name for i, name in enumerate(filename) if i not in nonzero_cyclist_idx]

    total_pedestrian = sum(pedestrian_N)
    train_pedestrian_target = total_pedestrian * fraction['train']
    valid_pedestrian_target = total_pedestrian * fraction['valid']
    test_pedestrian_target = total_pedestrian * fraction['test']

    train_pedestrian = 0
    valid_pedestrian = 0
    test_pedestrian = 0
    idx_list = [i for i, _ in enumerate(pedestrian_filename)]
    while test_pedestrian < test_pedestrian_target:
        idx = random.choice(idx_list)
        test_files.append(pedestrian_filename[idx])
        test_pedestrian += pedestrian_N[idx]
        idx_list.remove(idx)
    while valid_pedestrian < valid_pedestrian_target:
        idx = random.choice(idx_list)
        valid_files.append(pedestrian_filename[idx])
        valid_pedestrian += pedestrian_N[idx]
        idx_list.remove(idx)
    for idx in idx_list:
        train_pedestrian += pedestrian_N[idx]
        train_files.append(pedestrian_filename[idx])  

    # Copy the splits to suitable directories.
    for file in train_files:
        train_dir = os.path.join(destination, 'train')
        shutil.copy2(os.path.join(source, file), os.path.join(train_dir, file))
    for file in valid_files:
        valid_dir = os.path.join(destination, 'val')
        shutil.copy2(os.path.join(source, file), os.path.join(valid_dir, file))
    for file in test_files:
        test_dir = os.path.join(destination, 'test')
        shutil.copy2(os.path.join(source, file), os.path.join(test_dir, file))

def get_stats(dataset, record_name):
    stat = {'filename':record_name}
    num_vehicle = []
    num_pedestrian = []
    num_cyclist = []
    num_object = []

    for data in dataset:
        n_v, n_p, n_c = count_classes(data)
        num_vehicle.append(n_v)
        num_pedestrian.append(n_p)
        num_cyclist.append(n_c)
        num_object.append(n_v + n_p + n_c)

    stat['N_vehicle'] = np.mean(num_vehicle)
    stat['N_pedestrian'] = np.mean(num_pedestrian)
    stat['N_cyclist'] = np.mean(num_cyclist)
    stat['N_object'] = np.mean(num_object)

    return stat

def count_classes(data):
    classes = data['groundtruth_classes'].numpy()
    num_vehicle = np.count_nonzero(classes == 1)
    num_pedestrian = np.count_nonzero(classes == 2)
    num_cyclist = np.count_nonzero(classes == 4)

    return num_vehicle, num_pedestrian, num_cyclist



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)