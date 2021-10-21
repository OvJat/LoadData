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
    # shape [num, channels, height, width]
    images = torch.rand(num_samples, 3, 128, 128, dtype=torch.float16)
    labels = torch.randint(0, 10, (num_samples,), dtype=torch.long)
    data_file = os.path.join(data_dir, 'dataset.pth')
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    torch.save({
        'images': images,
        'labels': labels,
    }, data_file)
    print("save fake dataset file %s" % data_file)


class MyData(td.Dataset):

    def __init__(self, data_file: str):
        pth = torch.load(data_file)
        self.images = pth['images']
        self.labels = pth['labels']
        print(self.images.shape)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        xx = self.images[index]
        yy = self.labels[index]
        return xx, yy


def main():
    data_dir = './data/demo1'
    data_file = os.path.join(data_dir, 'dataset.pth')

    preprocess(data_dir, 10000)

    batch_size = 32

    ds = MyData(data_file)
    dl = td.DataLoader(ds, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0)
    start = time.time()
    for i, (xx, yy) in enumerate(dl):
        pass
    total_time = time.time() - start
    print("total time: %5f" % total_time)


if __name__ == '__main__':
    main()
