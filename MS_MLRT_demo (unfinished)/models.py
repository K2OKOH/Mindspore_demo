import mindspore
import mindspore as ms
from mindspore import nn
import mindspore.ops as ops

class AlexNet(nn.Cell):
    def __init__(self, num_classes=7):
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, num_classes),
            
        )

    def construct(self, x):
        feat = self.features(x)
        feat = feat.view(feat.shape[0], -1)
        class_out = self.classifier(feat)
        return class_out
