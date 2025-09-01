from konfai.predictor import Reduction
from konfai.data.transform import Transform
from konfai.utils.dataset import Attribute, data_to_image, Dataset
import torch
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label

class Concat(Reduction):

    def __init__(self):
        pass

    def __call__(self, result: torch.Tensor) -> torch.Tensor:
        return result.view(result.shape[1], -1, *result.shape[3:])
    
class Uncertainty(Transform):

    def __init__(self, mask: str):
        super().__init__()
        self.mask = mask

    def refine_segmentation_with_mask(self, output_segmentation: torch.Tensor, mask: np.ndarray):
        """
        Garde uniquement les composantes connectées de output_segmentation
        qui intersectent avec au moins un voxel == target_label dans cropped_mask.
        """
        seg = (output_segmentation.numpy() > 0).astype(np.uint8)
        mask = (mask == 1).astype(np.uint8)


        labeled, num_features = label(seg)
        if num_features == 0:
            return torch.from_numpy(output_segmentation)

        keep_components = []
        for comp in range(1, num_features + 1):
            comp_mask = (labeled == comp)
            if np.any(mask & comp_mask):
                keep_components.append(comp)

        refined = np.isin(labeled, keep_components).astype(np.uint8)
        return torch.from_numpy(refined)

    
    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        mask = None
        for dataset in self.datasets:
            if dataset.is_dataset_exist(self.mask, name):
                mask,_ = dataset.read_data(self.mask, name)
                break
        if mask is None:
            raise NameError(f"Mask : {name}/{self.mask} not found")
        mask = mask[0]
        
        box = cache_attribute.get_np_array("Box").astype(np.int64)
        mask = mask[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
        tensors = tensors.view(5, 3, *tensors.shape[1:])
        tensors = torch.argmax(tensors, dim=1).to(torch.uint8)
        tensors[tensors>1] = 0

        for i, tensor in enumerate(tensors):
            tensors[i] = self.refine_segmentation_with_mask(tensor, mask)
        
        probability = tensors.sum(0)/tensors.shape[0]
        dataset = Dataset("./Predictions/Curvas_1/Dataset", "mha")
        dataset.write("Prob", name, data_to_image(probability.unsqueeze(0).numpy(), cache_attribute))
        list_segmentation_sitk = [data_to_image(tensor.unsqueeze(0).numpy(), cache_attribute) for tensor in tensors]

        foregroundValue = 1
        threshold = 0.5
        max_iter = 50
        staple_filter = sitk.STAPLEImageFilter()
        staple_filter.SetForegroundValue(foregroundValue)
        staple_filter.SetMaximumIterations(max_iter)  # limite dure
        reference_segmentation_STAPLE_probabilities = staple_filter.Execute(list_segmentation_sitk)
        output_segmentation = reference_segmentation_STAPLE_probabilities > threshold
        output_segmentation = torch.tensor(sitk.GetArrayFromImage(output_segmentation)).unsqueeze(0)
        return output_segmentation