import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet

from modules.losses import AMSoftmaxLoss


__all__ = ['ResNet18_Extractor', 'DeiT_Extractor', 'EfficientNet_Extractor', 'AudioClassifier']


class ChannelPool(nn.Module):
    """ 
        Pooling over channels.
        input_shape = [C, H, W], output_shape = [2, H, W]
    """
    def forward(self, x):
        max_p = torch.max(x, 1)[0].unsqueeze(1)
        mean_p = torch.mean(x, 1).unsqueeze(1)
        return torch.cat([max_p, mean_p], dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, out_channels, kernel_size, stride, padding, \
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        """ 
            Idea of Spatial Attention Module (SAM) from CBAM paper: 
            https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
        """
        super(SpatialAttention, self).__init__()
        self.pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels  = 2, 
                out_channels = out_channels, 
                kernel_size  = kernel_size, 
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias
            ),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else nn.Identity(),
            nn.ReLU() if relu else nn.Identity()
        )

    def forward(self, x):
        x_out = self.pool(x)
        x_out = self.conv(x_out)
        scale_mask = F.sigmoid(x_out)
        return scale_mask * x
        

class BaseExtractor(nn.Module):
    def __init__(self, input_shape, num_classes, embed_size):
        """
            Basic extractor class.
            :param tuple input_shape:   shape of input tensor
            :param int num_classes:     number of classes to classify
            :param int embed_size:      latent vector size
        """
        super(BaseExtractor, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.sa = SpatialAttention(out_channels=1, kernel_size=7, stride=1, padding=3, relu=False)
        self.fc = nn.Linear(1000, embed_size)
        self.emb_extractor = AMSoftmaxLoss(in_features=embed_size, out_features=num_classes)

    def __forward_emb(self, x):
        x_out = self.model(x)
        x_out = F.dropout(F.leaky_relu(x_out, 0.2, True), 0.3)
        out = self.fc(x_out)
        return out

    def forward(self, x, vad_mask, labels=None, output_type='loss'):
        # Apply attention masks
        x = x * vad_mask.reshape(vad_mask.shape[0], 1, -1, 1)
        x = self.sa(x)

        # Process pipeline over masked input
        if output_type == 'emb':
            out = self.__forward_emb(x)
        elif output_type == 'loss':
            x = self.__forward_emb(x)
            out = self.emb_extractor(x, labels)
        return out

    @property
    def device(self):
        return self.fc.weight.device


class AudioClassifier(nn.Module):
    def __init__(self, extractor):
        """
            Classifier-wrapper over BasicExtractor class.
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
        emb = self.extractor.forward_emb(x)
        out = self.classification_pipe(emb)
        return out

    @property
    def device(self):
        return self.extractor.fc.weight.device


class DeiT_Extractor(BaseExtractor):
    """ ~5M params, 76.6 Top-1% on ImageNet """
    # Doesn't fit in my ill memory
    def __init__(self, **kwargs):
        super(DeiT_Extractor, self).__init__(**kwargs)
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.model.patch_embed.proj = nn.Conv2d(kwargs['input_shape'], 192, 16, 16)


class ResNet18_Extractor(BaseExtractor):
    """ ~12M params, ??? (not even in rank) """
    def __init__(self, **kwargs):
        super(ResNet18_Extractor, self).__init__(**kwargs)
        self.model = torchvision.models.resnet18(pretrained=True)


class EfficientNet_Extractor(BaseExtractor):
    """ ~43M params, 86% Top-1% in ImageNet (b0) """
    # Doesn't fit in memory even with b1 configuration
    def __init__(self, effnet_type='b6', **kwargs):
        super(EfficientNet_Extractor, self).__init__(**kwargs)
        self.model = EfficientNet.from_pretrained('efficientnet-%s' % effnet_type)
