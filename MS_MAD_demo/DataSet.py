import os
import cv2
import numpy as np

class MyDatasets():
    """
    文本CSV数据集加载器
    加载数据集并处理为一个Python迭代对象。
    """

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
        # self.review, self.label = [], []
        # self._load()

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

def FreCom(img):
    h,w = img.shape[:2]
    img_dct = np.zeros((h,w,3))
    #img_dct = np.fft.fft2(img, axes=(0, 1))
    for i in range(3):
        img_ = img[:, :, i] # 获取rgb通道中的一个
        img_ = np.float32(img_) # 将数值精度调整为32位浮点型
        img_dct[:,:,i] = cv2.dct(img_)  # 使用dct获得img的频域图像

    return img_dct

def Matching(img,reference,alpha=0.2,beta=1):
    #lam = np.random.uniform(alpha, beta)
    theta = np.random.uniform(alpha, beta)
    h, w = img.shape[:2]
    img_dct=FreCom(img)
    r = np.random.randint(1,5)
    # r = 1
    img_dct[r,r,:]=0
    ref_dct=FreCom(reference)
    # img_fc = img_dct * theta + ref_dct
    img_fc = img_dct + ref_dct * theta
    img_out = np.zeros((h, w, 3))
    #img_out = np.uint8(np.clip(np.fft.ifft2(img_fc, axes=(0, 1)),0,255))
    for i in range(3):
        img_ = img_fc[:, :, i]  # 获取rgb通道中的一个
        img_out[:, :, i] = cv2.idct(img_).clip(0,255)  # 使用dct获得img的频域图像

    return img_out

class EnhanceDatasets():
    """
    文本CSV数据集加载器
    加载数据集并处理为一个Python迭代对象。
    """

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
        # self.review, self.label = [], []
        # self._load()

    def __getitem__(self, idx):
        """
        定义可迭代对象返回当前结果的逻辑
        """
        img_path = self.data[idx][0]
        img_label = self.data[idx][1]
        img = cv2.imread(img_path).astype(np.float32).transpose(2, 0, 1)
        
        h1, w1 = img.shape[:2]
        if h1%2!=0 or w1%2!=0:
                img=cv2.resize(img,(w1-w1%2,h1-h1%2),interpolation=cv2.INTER_AREA)

        refrence=np.ones_like(img)
        # 参考图片随机色彩 (三个通道分别随机化)
        refrence[:,:,0] = refrence[:,:,0]*np.random.randint(0,255)
        refrence[:,:,1] = refrence[:,:,1]*np.random.randint(0,255)
        refrence[:,:,2] = refrence[:,:,2]*np.random.randint(0,255)

        img_matched = Matching(img,refrence)
            
        if self.transform:
            img_matched = self.transform(img_matched)
        
        return img_matched, img_label

    def __len__(self):
        """
        返回可迭代对象的长度
        :return: int
        """
        return len(self.data)

# PACS = MyDatasets('../DataSet/PACS', 'a')