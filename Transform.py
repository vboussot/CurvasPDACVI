from konfai.data.transform import Transform, TransformInverse 
from konfai.utils.dataset import Attribute
from scipy.ndimage import label
import torch
import numpy as np

class KeepLargestComponent(Transform):
    
    def __init__(self, target_label : int = 7):
        self.target_label = target_label
    
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        binary = (tensor.numpy() == self.target_label).astype(np.int32)
        labeled_array, num_features = label(binary)

        if num_features == 0:
            return torch.zeros_like(tensor)

        counts = np.bincount(labeled_array.ravel())
        counts[0] = 0 
        largest_component = counts.argmax()

        cleaned_mask = torch.zeros_like(tensor)
        cleaned_mask[labeled_array == largest_component] = 1
        return cleaned_mask

class CropImageByMask(TransformInverse):
   
    def __init__(self, mask: str, margin_mm : list[int] = [15, 50, 100]):
        super().__init__(True)
        self.margin_mm = margin_mm
        self.mask = mask
        self.state = {}

    def _get_box(self, name: str, shape: list[int], cache_attribute: Attribute):
        mask = None
        for dataset in self.datasets:
            if dataset.is_dataset_exist(self.mask, name):
                mask,_ = dataset.read_data(self.mask, name)
                break
        if mask is None:
            raise NameError(f"Mask : {name}/{self.mask} not found")
        mask = mask[0]
        spacing = cache_attribute.get_np_array("Spacing")
        indices = np.argwhere(mask == 1)

        if indices.size == 0:
            return tensor
        
        # Bounding box min/max
        min_idx = indices.min(axis=0)
        max_idx = indices.max(axis=0)
        # Convert margin in mm to voxels (Z, Y, X)
        margin_vox = [
            int(round(self.margin_mm[0] / spacing[2])),  # Z
            int(round(self.margin_mm[1] / spacing[1])),  # Y
            int(round(self.margin_mm[2] / spacing[0]))   # X
        ]

        # Apply margins
        min_idx = min_idx - margin_vox
        max_idx = max_idx + margin_vox
        # Clamp to image bounds
        shape = mask.shape  # (Z, Y, X)
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, shape)
        return min_idx[0], max_idx[0], min_idx[1], max_idx[1], min_idx[2], max_idx[2]


    def transform_shape(self, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        box = self._get_box(name, shape, cache_attribute)
        self.state[name] = box
        cache_attribute["Shape"] = torch.tensor([1]+shape)
        spacing = cache_attribute.get_np_array("Spacing")
        origin = cache_attribute.get_np_array("Origin")
        new_origin = [
            origin[0] + box[4] * spacing[0],
            origin[1] + box[2] * spacing[1],
            origin[2] + box[0] * spacing[2]
        ]
        cache_attribute["Origin"] = torch.tensor(new_origin)
        return [box[1]-box[0], box[3]-box[2], box[5]-box[4]]
    
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        box = self.state[name]
        return tensor[:,box[0]:box[1], box[2]:box[3], box[4]:box[5]]
    
    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        initial_shape = cache_attribute.pop_np_array("Shape").astype(np.int64)
        initial_shape[0] = 5
        cache_attribute.pop_np_array("Origin")
        box = self.state[name]
        del self.state[name]
        result = torch.zeros(tuple(initial_shape))
        result[:, box[0]:box[1], box[2]:box[3], box[4]:box[5]] = tensor
        return result