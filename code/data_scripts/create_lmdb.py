import os
import os.path as osp
import cv2
import lmdb
import argparse
import sys
import glob
from tqdm import tqdm

def main(args):
    parser = argparse.ArgumentParser(description='Convert image folder to lmdb')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input folder path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output lmdb path')
    args = parser.parse_args(args)

    img_folder = args.input
    lmdb_save_path = args.output

    all_img_list = sorted(glob.glob(osp.join(img_folder, '*')))
    keys = [osp.splitext(osp.basename(img_path))[0] for img_path in all_img_list]

    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    with env.begin(write=True) as txn:
        resolutions = []
        for idx, (path, key) in tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list)):
            data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if data.ndim == 2:
                H, W = data.shape
                C = 1
            else:
                H, W, C = data.shape
            txn.put(key.encode('utf-8'), data)
            resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))

    env.close()
    print('Finish writing lmdb.')

if __name__ == '__main__':
    main(sys.argv[1:])
