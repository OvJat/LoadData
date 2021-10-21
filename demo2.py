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
        yy = torch.randint(0, 10, (1,), dtype=torch.long)
        torch.save({
            'image': xx,
            'label': yy,
        }, data_file)


class MyData(td.Dataset):

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_list = []
        # scan files
        for name in os.listdir(self.data_dir):
            if not (name.startswith('sample_') and name.endswith('.pth')):
                continue
            path = os.path.join(data_dir, name)
            self.file_list.append(path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        # load file for sample index
        path = self.file_list[index]
        pth = torch.load(path)
        xx = pth['image']
        yy = pth['label']
        return xx, yy


def main():
    data_dir = './data/demo2'

    preprocess(data_dir, 10000)

    batch_size = 32

    ds = MyData(data_dir)
    dl = td.DataLoader(ds, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=2)
    start = time.time()
    for i, (xx, yy) in enumerate(dl):
        pass
    total_time = time.time() - start
    print("total time: %5f" % total_time)


if __name__ == '__main__':
    main()
