from torchvision.models import detection as tv_det_models
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    # old versions of torchvision
    from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'mobilenet_v3_large_320': 'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
    'mobilenet_v3_large': 'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
}

class TVFasterRCNN(tv_det_models.FasterRCNN):
    def __init__(self, args, dataset):
        opts = args.model_opts
        self.num_classes = len(dataset.classes)
        self.backbone_name = opts.get('backbone', 'resnet50')
        backbone = tv_det_models.backbone_utils.resnet_fpn_backbone(self.backbone_name, False)
        super().__init__(backbone, self.num_classes)
        state_dict = load_state_dict_from_url(model_urls[self.backbone_name], progress=True)
        self.load_state_dict(state_dict)
