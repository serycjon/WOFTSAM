import einops
import torch
import torch.nn.functional as F
import cv2
from torchvision.transforms.functional import to_tensor, normalize
from typing import Optional, Union

class DINOFeatureExtractor:
    def __init__(self, model_name='dinov2_vits14_reg'):
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        model.eval()
        model.cuda()
        model.requires_grad_(False)
        self.model = model

    def extract(self, img_bgr: 'np.ndarray', layer: int, normalized=True) -> torch.Tensor:
        # Preprocess
        assert img_bgr.shape == (224, 224, 3)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = to_tensor(img_rgb).unsqueeze(0)
        img_tensor = normalize(img_tensor,
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

        feat_maps = self.model.get_intermediate_layers(img_tensor.cuda(), n=layer) # n=(3, 7, 11) -> returns tuple
        feat_map = feat_maps[0]  # 1, HW, D
        full_shape = einops.rearrange(feat_map, '1 (H W) D -> H W D', H=16, W=16, D=384) # 16 = 224 / 14 patch size
        full_shape = full_shape.cpu()
        if normalized:
            return F.normalize(full_shape, p=2, dim=2)
        else:
            return full_shape
        
        # if feature_point not in self.hooks:
        #     raise ValueError(f"Invalid feature_point: {feature_point}")

        # _ = self.model(img_tensor)
        # tokens = self.feature_maps[feature_point]  # [1, num_tokens, dim]

        # assert tokens.shape[0] == 1
        # # if tokens.shape[1] == 197:  # includes CLS token
        # #     patch_tokens = tokens[:, 1:, :]
        # #     feature_map = patch_tokens.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        # #     return feature_map  # [1, C, 14, 14]
        # return tokens
