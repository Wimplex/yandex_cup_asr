import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding


__all__ = ['ResNet18_Extractor', 'DeiT_Extractor', 'EfficientNet_Extractor']


class AMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        """ AM Softmax Loss """
        
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.__init_weights()

    def __init_weights(self):
        pass

    def forward(self, x, labels):        
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters(): W = F.normalize(W, dim=1)
        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        l = numerator - torch.log(denominator)
        return -torch.mean(l)


class BaseExtractor(nn.Module):
    def __init__(self, input_shape, num_classes, embed_size):
        """
        :param tuple input_shape:   shape of input tensor
        :param int num_classes:     number of classes to classify
        :param int embed_size:      latent vector size
        """
        super(BaseExtractor, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.fc = nn.Linear(1000, embed_size)
        self.emb_extractor = AMSoftmaxLoss(in_features=embed_size, out_features=num_classes)

    def forward(self, x, labels=None, return_type='loss'):
        x = F.dropout(F.leaky_relu(self.model(x), 0.2, True), 0.3)
        x = self.fc(x)
        if return_type == 'emb': out = x
        elif return_type == 'loss': out = self.emb_extractor(x, labels)
        else: out = (x, self.emb_extractor(x, labels))
        return out


class AudioClassifier(nn.Module):
    def __init__(self, extractor):
        """
        :param BaseExtractor extractor: embedding extractor model
        """
        super(AudioClassifier, self).__init__()
        self.extractor = extractor
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.classification_pipe = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.extractor.embed_size, 256, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            nn.Linear(256, self.extractor.num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        emb = self.extractor(x, return_type='emb')
        out = self.classification_pipe(emb)
        return out


class DeiT_Extractor(BaseExtractor):
    """ ~5M params, 76.6 Top-1% on ImageNet """
    def __init__(self, input_shape, num_classes, embed_size):
        super(DeiT_Extractor, self).__init__(input_shape, num_classes, embed_size)
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.model.patch_embed.proj = nn.Conv2d(input_shape[0], 192, 16, 16)


class ResNet18_Extractor(BaseExtractor):
    """ ~12M params, ??? (not even in rank) """
    def __init__(self, input_shape, num_classes, embed_size):
        super(ResNet18_Extractor, self).__init__(input_shape, num_classes, embed_size)
        self.model = torchvision.models.resnet18(pretrained=True)
        # self.model.conv1 = nn.Conv2d(input_shape[0], 64, 7, 2, 3, bias=False)


class EfficientNet_Extractor(BaseExtractor):
    """ ~43M params, 86% Top-1% in ImageNet """
    def __init__(self, input_shape, num_classes, embed_size, effnet_type='b6'):
        super(EfficientNet_Extractor, self).__init__(input_shape, num_classes, embed_size)
        self.model = EfficientNet.from_pretrained('efficientnet-%s' % effnet_type)


if __name__ == '__main__':
    model = EfficientNet_Extractor(input_shape=[3, 172, 64], num_classes=31, embed_size=256, effnet_type='b6')
    model.to('cuda:0')
