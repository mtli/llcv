import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

import torchvision.datasets as tv_datasets
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    # old versions of torchvision
    from torchvision.models.utils import load_state_dict_from_url

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # pytorch style as implemented in torchvision (as opposed to caffe style)
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes*(base_width/64.))*groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes*self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    depth_cfgs = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3])
    }

    pretrained_urls = {
        18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def __init__(self, args, dataset):
        super().__init__()
        opts = args.model_opts
        self.num_classes = len(dataset.classes)
        self.is_cifar = isinstance(dataset, (tv_datasets.CIFAR10, tv_datasets.CIFAR10))
        self.depth = opts.get('depth', 18)
        block, layer_blocks = self.depth_cfgs[self.depth]
        self.in_channels = opts.get('in_channels', 3)
        self.base_width = opts.get('base_width', 64)
        self.groups = opts.get('groups', 1)
        assert self.groups == 1, 'groups are not supported yet'
        self.norm_type = opts.get('norm_type', 'BatchNorm2d')
        self.norm_layer = vars(nn)[self.norm_type]
        self.zero_init_residual = opts.get('zero_init_residual', True)
        # Deformable convs are not yet supported
        self.dcn = opts.get('dcn', None)
        self.stage_with_dcn = (False, False, False, False)

        # these three options are for finetuning
        self.norm_eval = opts.get('norm_eval', False)
        self.frozen_stages = opts.get('frozen_stages', -1)
        self.pretrained = opts.get('pretrained', None)
        if self.pretrained == 'pytorch':
            self.pretrained = self.pretrained_urls[self.depth]

        strides = opts.get('strides', None)
        if strides is None:
            if self.is_cifar:
                # conv1, maxpool, stage2, stage3, stage4, stage5
                self.strides = [1, 1, 1, 2, 2, 2]
            else:
                # by default use ImageNet strides
                self.strides = [2, 2, 1, 2, 2, 2]
        else:
            self.strides = strides


        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.base_width,
            kernel_size=7 if self.strides[0] > 1 else 3,
            stride=self.strides[0],
            padding=3 if self.strides[0] > 1 else 1,
            bias=False,
        )
        self.bn1 = self.norm_layer(self.base_width)
        self.relu = nn.ReLU(inplace=True)
        if self.strides[1] > 1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        else:
            self.maxpool = nn.Identity()

        self.inplanes = self.base_width
        self.dilation = 1

        self.layer1 = self._make_layer(block, self.base_width, layer_blocks[0], stride=self.strides[2])
        self.layer2 = self._make_layer(block, 2*self.base_width, layer_blocks[1], stride=self.strides[3])
        self.layer3 = self._make_layer(block, 4*self.base_width, layer_blocks[2], stride=self.strides[4])
        self.layer4 = self._make_layer(block, 8*self.base_width, layer_blocks[3], stride=self.strides[5])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8*self.base_width*block.expansion, self.num_classes)

        self.init_weights(self.pretrained)
        self._freeze_stages()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        outplanes = planes*block.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, 1, stride=stride, bias=False),
                self.norm_layer(outplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for p in m.parameters():
                    p.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def init_weights(self, pretrained=None, device='cpu'):
        if pretrained:
            if pretrained.startswith('https://') or pretrained.startswith('http://'):
                state_dict = load_state_dict_from_url(pretrained, map_location=device)
            else:
                state_dict = torch.load(pretrained, map_location=device)
            self.load_state_dict(state_dict)
        elif not self.is_cifar:
            # the CIFAR version works fine with layer default init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        co = m.conv2.conv_offset
                        nn.init.constant_(co.weight, 0)
                        if hasattr(co, 'bias') and co.bias is not None:
                            nn.init.constant_(co.bias, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 1)
                        nn.init.constant_(m.bn3.bias, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 1)
                        nn.init.constant_(m.bn2.bias, 0)

    def train(self, mode=True):
        # If self.norm_eval is True, the running mean/variance in
        # the normalization layers will not be updated
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
