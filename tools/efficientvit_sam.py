
try:
    from enum import Enum
    import numpy as np
    import cv2
    import onnxruntime as ort
    from typing import Any, Tuple, Union
    from copy import deepcopy
except ImportError:
    pass
YELLOW = (255, 244, 79)
GREY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (135, 206, 235)
PINK = (239, 149, 186)
BLACK = (0, 0, 0)
def draw_binary_mask(raw_image, binary_mask, mask_color):
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas
def draw_point_masks(img, masks, coord_and_label):
    fine_grained_mask = masks[0][-1]

    oh, ow = fine_grained_mask.shape
    img = draw_binary_mask(img, fine_grained_mask, mask_color=YELLOW)

    point_radius = ow // 125
    border_thickness = point_radius // 3

    for x, y, l in coord_and_label:
        point_color = BLUE if l == 1 else PINK
        cv2.circle(img, (int(x), int(y)), point_radius + border_thickness, BLACK, -1)
        cv2.circle(img, (int(x), int(y)), point_radius, point_color, -1)

    return img
class EfficientVitSAMType(Enum):
    l0 = 0
    l1 = 1
    l2 = 2
    @staticmethod
    def from_str(label:str):
        return EfficientVitSAMType[label.lower()]
class EfficientVitSamEncoder:
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        opt = ort.SessionOptions()
        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, input: np.ndarray) -> np.ndarray:
        """
        the input should be normalized and in CxHxW format
        """
        feature = self.session.run(None, {self.input_name: input})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)
class EfficientVitSamDecoder:
    def __init__(
        self, model_path: str, device: str = "cpu", target_size: int = 1024, mask_threshold: float = 0.0, **kwargs
    ):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")
        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)
    
    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes
    
    def mask_postprocessing(self,masks: np.ndarray, orig_im_size: tuple) -> np.ndarray:
        """
        original size is (h, w), masks shape is (256, 256), return masks shape (h, w)
        """
        org_h = orig_im_size[0]
        org_w = orig_im_size[1]
        masks = cv2.resize(masks[0,0,:], (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        #prepadded_size = resize_longest_image_size(orig_im_size, self.target_size)
        scale = self.target_size / max(org_w, org_h)
        prepadded_h = int(org_h * scale + 0.5)
        prepadded_w = int(org_w * scale + 0.5)
        masks = masks[..., : prepadded_h, : prepadded_w]
        if masks.shape[0] != org_h or masks.shape[1] != org_w:
            masks = cv2.resize(masks, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
        masks = np.expand_dims(masks, axis=0)
        masks = np.expand_dims(masks, axis=0)
        return masks
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        return_logits: bool = False,
    ):
        input_size = self.get_preprocess_shape(*origin_image_size, long_side_length=self.target_size)

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if point_coords is not None:
            if isinstance(point_coords, list) or isinstance(point_coords, tuple):
                point_coords = np.array(point_coords).reshape((-1, 2))
            if isinstance(point_labels, list) or isinstance(point_labels, tuple):
                point_labels = np.array(point_labels)
            point_coords = np.expand_dims(point_coords, axis=0).astype(np.float32)
            point_labels = np.expand_dims(point_labels, axis=0).astype(np.float32)
            point_coords = self.apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)
            prompts, labels = point_coords, point_labels

        if boxes is not None:
            boxes = self.apply_boxes(boxes, origin_image_size, input_size).astype(np.float32)
            box_labels = np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))

            if point_coords is not None:
                prompts = np.concatenate([prompts, boxes], axis=1)
                labels = np.concatenate([labels, box_labels], axis=1)
            else:
                prompts, labels = boxes, box_labels

        input_dict = {"image_embeddings": img_embeddings, "point_coords": prompts, "point_labels": labels}
        low_res_masks, iou_predictions = self.session.run(None, input_dict)

        masks = self.mask_postprocessing(low_res_masks, origin_image_size)

        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

from sam_registry import sam_registry
@sam_registry.register('EfficientVitSAM')
class EfficientVitSAM:
    """
    efficientVit sam onnx model
    """
    def __init__(self, model_type:str, encoder_model_path:str,decoder_onnx_path:str):
        self.sam_type = "EfficientVitSAM"
        self.model_type = EfficientVitSAMType.from_str(model_type)
        self.encoder_onnx_model_path = encoder_model_path
        self.decoder_onnx_path = decoder_onnx_path
        self.pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        self.pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]
        self.target_size = 1024
        self.encoder = EfficientVitSamEncoder(self.encoder_onnx_model_path)
        #self.decoder = EfficientVitSamDecoder(self.decoder_onnx_model_path, target_size=self.target_size)
    def preprocess(self, image:np.ndarray)->np.ndarray:
        if not isinstance(image,np.ndarray):
            raise ValueError("Input image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in RGB format")
        image = image[:,:,::-1]
        org_w,org_h = image.shape[1],image.shape[0]
        long_size = max(org_w,org_h)
        if long_size != 512:
            scale = 512 / long_size
            input = cv2.resize(image, (int(org_w*scale+0.5), int(org_h*scale+0.5)), interpolation=cv2.INTER_LINEAR)
            input = input.transpose((2,0,1))
        else:
            input = image.transpose((2,0,1))
        input = input.astype(np.float32)
        input = input / 255.0
        #normalize
        input[0] = (input[0] - self.pixel_mean[0]) / self.pixel_std[0]
        input[1] = (input[1] - self.pixel_mean[1]) / self.pixel_std[1]
        input[2] = (input[2] - self.pixel_mean[2]) / self.pixel_std[2]
        #pad to 512x512
        if input.shape[1] < 512:
            pad = 512 - input.shape[1]
            input = np.pad(input,((0,0),(0,pad),(0,0)),"constant",constant_values=0)
        if input.shape[2] < 512:
            pad = 512 - input.shape[2]
            input = np.pad(input,((0,0),(0,0),(0,pad)),"constant",constant_values=0)
        input = np.expand_dims(input, axis=0)
        return input
    
    def encode_image(self, image:np.ndarray)->np.ndarray:
        """
        encode the image with encoder model, return the encoded feature(1,256,64,64)
        """
        input = self.preprocess(image)
        feature = self.encoder(input)
        return feature
    
if __name__ == '__main__':
    import argparse
    import os
    import time
    parser = argparse.ArgumentParser(description="test efficient sam onnx encoder")
    parser.add_argument("--model_type",required=True,type=str,help="model type")
    parser.add_argument("--encoder_model_path",required=True,type=str,help="encoder path")
    parser.add_argument("--decoder_model_path",required=True,type=str,help="decoder path")
    parser.add_argument("--input",required=True,type=str,help="input image")
    args = parser.parse_args()
    if not os.path.exists(args.encoder_model_path):
        raise ValueError("model file not found!")
    predictor = EfficientVitSAM(args.model_type,args.encoder_model_path,args.decoder_model_path)
    if not os.path.exists(args.input):
        raise ValueError("input image file not found!")
    image = cv2.imread(args.input)
    feature = predictor.encode_image(image)
    decoder = EfficientVitSamDecoder(args.decoder_model_path)
    points_prompt = np.array([417,400,195,432]).reshape(-1,2)
    labels = np.array([1,0])
    masks,_,_ = decoder.run(feature, image.shape[:2],points_prompt,labels)
    canvas = draw_point_masks(image,masks,np.hstack([points_prompt,labels.reshape(-1,1)]))
    cv2.imshow("canvas",canvas)
    cv2.waitKey(0)
    start = time.time()
    for n in range(10):
        predictor.encode_image(image)
    print("Time:",time.time()-start)
        