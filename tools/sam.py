try:
    from mobile_sam import SamPredictor,sam_model_registry
    import torch
    import numpy as np
except ImportError:
    pass
    
from sam_registry import sam_registry
@sam_registry.register('SegmentAnything')
class SegmentAnything:
    """
    original facebook segment anything pytorch model
    """
    def __init__(self,model_type:str,encoder_model_path:str,decoder_onnx_path:str):
        self.sam_type = "SAM"
        self.model_type = model_type
        self.encoder_model_path = encoder_model_path
        self.decoder_onnx_path = decoder_onnx_path
        self.sam = sam_model_registry[self.model_type](checkpoint=self.encoder_model_path)
        self.model = SamPredictor(self.sam)
        if torch.backends.cuda.is_built() :
            self.sam.to("cuda")
        self.predictor = SamPredictor(self.sam)
    def encode_image(self, image:np.ndarray)->np.ndarray:
        """
        image as numpy array
        """
        if not isinstance(image,np.ndarray):
            raise ValueError("Input image must be a numpy array")
        self.predictor.set_image(image)
        return self.predictor.features.detach().cpu().numpy()