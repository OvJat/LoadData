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


def preprocess(data_dir: str, num_samples: int):
    for i in range(num_samples):
        file_name = 'sample_%010d.pth' % i
        data_file = os.path.join(data_dir, file_name)
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        xx = torch.rand(3, 128, 128, dtype=torch.float16)
        torch.save({
            'image': xx,
        }, data_file)


class MyData(td.Dataset):

    def __init__(self, data_dir: str, sample_lag: int = 12):
        self.data_dir = data_dir
        self.sample_lag = sample_lag
        self.file_list = []
        # scan files
        for name in os.listdir(self.data_dir):
            if not (name.startswith('sample_') and name.endswith('.pth')):
                continue
            path = os.path.join(data_dir, name)
            self.file_list.append(path)

        total_length = len(self.file_list)
        self.sample_count = total_length - sample_lag + 1

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index: int):
        file_list = self.file_list[index:index + self.sample_lag]
        image_list = []
        for path in file_list:
            pth = torch.load(path)
            image = torch.unsqueeze(pth['image'], dim=0)
            image_list.append(image)
        images = torch.cat(image_list, dim=0)
        return images


def main():
    data_dir = './data/demo5'

    preprocess(data_dir, 10000)

    batch_size = 32

    ds = MyData(data_dir)
    dl = td.DataLoader(ds, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=2)
    start = time.time()
    for i, xx in enumerate(dl):
        print(xx.shape)
    total_time = time.time() - start
    print("total time: %5f" % total_time)


if __name__ == '__main__':
    main()
