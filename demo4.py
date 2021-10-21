#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Support Python 3.8
    @author: Lou Xiao(louxiao@i32n.com)
    @maintainer: Lou Xiao(louxiao@i32n.com)
    @copyright: Copyright 2018~2021
    @created time: 2021-10-17 19:26:32 CST
    @updated time: 2021-10-17 19:26:32 CST
"""

import os
import time
import json
import torch
import torch.utils.data as td
import h5py


def get_data_file(data_dir: str, index: int):
    index_l1 = index // 100
    index_l2 = index % 100
    index_name_l1 = 'sample_%010d' % index_l1
    index_name_l2 = 'sample_%03d' % index_l2

    file_name = '%s.hdf5' % index_name_l1
    data_file = os.path.join(data_dir, file_name)
    return data_file, index_name_l2


def preprocess(data_dir: str, num_samples: int):
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_samples):
        data_file, index_name_l2 = get_data_file(data_dir, i)
        with h5py.File(data_file, 'a') as f:
            xx = torch.rand(3, 128, 128, dtype=torch.float16)
            yy = torch.randint(0, 10, (1,), dtype=torch.long)

            key_xx = '%s/xx' % index_name_l2
            key_yy = '%s/yy' % index_name_l2
            # compression
            f.create_dataset(key_xx, data=xx, compression='gzip', compression_opts=9)
            f.create_dataset(key_yy, data=yy, compression='gzip', compression_opts=9)

    meta_file = os.path.join(data_dir, 'meta.json')
    with open(meta_file, 'w') as f:
        f.write(json.dumps({
            'count': num_samples,
        }))


class MyData(td.Dataset):

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_file = os.path.join(data_dir, 'meta.json')
        with open(meta_file, 'r') as f:
            dt = json.loads(f.read())
            self.sample_count = dt['count']

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index: int):
        # load file for sample index
        data_file, index_name = get_data_file(self.data_dir, index)
        with h5py.File(data_file, 'r') as f:
            dt = f[index_name]
            xx = dt['xx'][:]
            yy = dt['yy'][:]
        return xx, yy


def main():
    data_dir = './data/demo5'

    # preprocess(data_dir, 10000)

    batch_size = 32

    ds = MyData(data_dir)
    dl = td.DataLoader(ds, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=12)
    start = time.time()
    for i, (xx, yy) in enumerate(dl):
        pass
    total_time = time.time() - start
    print("total time: %5f" % total_time)


if __name__ == '__main__':
    main()
