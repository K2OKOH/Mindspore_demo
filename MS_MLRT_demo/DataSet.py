import os
import cv2
import numpy as np

class MyDatasets():
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        # 数据集下的子文件夹(类别)
        data_floder = os.listdir(self.data_path)
        data = []
        for item in data_floder:
            img_list = os.listdir(os.path.join(data_path, item))
            now_label = data_floder.index(item)
            
            for img in img_list:
                # 添加数据路径和label
                data.append((os.path.join(data_path, item, img), now_label))

        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        """
        定义可迭代对象返回当前结果的逻辑
        """
        img_path = self.data[idx][0]
        img_label = self.data[idx][1]
        img = cv2.imread(img_path).astype(np.float32).transpose(2, 0, 1)
        if self.transform:
            img = self.transform(img)

        return img, img_label

    def __len__(self):
        """
        返回可迭代对象的长度
        :return: int
        """
        return len(self.data)

# PACS = MyDatasets('../DataSet/PACS', 'a')