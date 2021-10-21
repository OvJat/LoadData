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
import torch
import torch.utils.data as td
import h5py


def preprocess(data_dir: str, num_samples: int):
    data_file = os.path.join(data_dir, 'dataset.hdf5')
    os.makedirs(data_dir, exist_ok=True)
    with h5py.File(data_file, 'w') as f:
        for i in range(num_samples):
            xx = torch.rand(3, 128, 128, dtype=torch.float16)
            yy = torch.randint(0, 10, (1,), dtype=torch.long)

            index_name = 'sample_%010d' % i
            key_xx = '%s/xx' % index_name
            key_yy = '%s/yy' % index_name
            # compression
            # f.create_dataset(key_xx, data=xx, compression='gzip', compression_opts=9)
            # f.create_dataset(key_yy, data=yy, compression='gzip', compression_opts=9)
            f.create_dataset(key_xx, data=xx)
            f.create_dataset(key_yy, data=yy)


class MyData(td.Dataset):

    def __init__(self, data_file: str):
        self.data_file = data_file
        self.index_list = []
        # scan files
        with h5py.File(self.data_file, 'r') as f:
            for name in f.keys():
                if name.startswith('sample_'):
                    self.index_list.append(name)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index: int):
        # load file for sample index
        index_name = self.index_list[index]
        with h5py.File(self.data_file, 'r') as f:
            dt = f[index_name]
            xx = dt['xx'][:]
            yy = dt['yy'][:]
        return xx, yy


def main():
    data_dir = './data/demo3'
    data_file = os.path.join(data_dir, 'dataset.hdf5')

    # preprocess(data_dir, 10000)

    batch_size = 32

    ds = MyData(data_file)
    dl = td.DataLoader(ds, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=5)
    start = time.time()
    for i, (xx, yy) in enumerate(dl):
        pass
    total_time = time.time() - start
    print("total time: %5f" % total_time)


if __name__ == '__main__':
    main()
