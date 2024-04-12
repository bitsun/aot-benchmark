try:
    import onnxruntime as ort
    import onnx
    import numpy as np
    import cv2
except ImportError:
    pass
from sam_registry import sam_registry
@sam_registry.register('SegmentAnythingHQOnnx')
class SegmentAnythingHQOnnx:
    """
    inference with High Quality Segment Anything HQ model
    """
    def __init__(self, encoder_model_path,decoder_model_path, device='cpu'):
        self.sam_type = "SAM_HQ"
        self.encoder_model = ort.InferenceSession(encoder_model_path)
        self.device = device
        self.input_size = (684, 1024)
        onnx_model = onnx.load(encoder_model_path)
        self.decoder_onnx_path = decoder_model_path
        self.is_fp16=onnx.TensorProto.DataType.Name(onnx_model.graph.input[0].type.tensor_type.elem_type) == 'FLOAT16'
        self.input_name = onnx_model.graph.input[0].name
        if ort.get_device() == 'GPU':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            so = ort.SessionOptions()
            so.log_severity_level = 1  # INFO level
            self.ort_session = ort.InferenceSession(encoder_model_path,so,providers=providers)
        else:
            self.ort_session = ort.InferenceSession(encoder_model_path,providers=['CPUExecutionProvider'])
        self.encoder_input_name = self.ort_session.get_inputs()[0].name
    def encode_image(self, cv_image:np.ndarray)->np.ndarray:
        """
        image as numpy array
        """
        if not isinstance(cv_image,np.ndarray):
            raise ValueError("Input image must be a numpy array")

        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size[1] / cv_image.shape[1]
        scale_y = self.input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )
        encoder_inputs = {
            self.encoder_input_name: cv_image.astype(np.float32),
        }
        features = self.ort_session.run(None, encoder_inputs)
        image_embeddings, interm_embeddings = features[0], np.stack(
            features[1:]
        )
        return image_embeddings
    
if __name__ == '__main__':
    encoder_model_path = "C:\\Users\\bliu\\anylabeling_data\\models\\sam-hq_vit_h_quant-r20231111\\sam_hq_vit_h_encoder_quant.onnx"
    decoder_model_path = "C:\\Users\\bliu\\anylabeling_data\\models\\sam-hq_vit_h_quant-r20231111\\sam_hq_vit_h_decoder.onnx"
    segment_anything = SegmentAnythingHQOnnx(encoder_model_path,decoder_model_path)
    image = cv2.imread("E:\\Data\\LogoWorkDir\\test\\elon_test.jpg")
    image_embeddings, interm_embeddings = segment_anything.encode_image(image)
    #measure the time
    import time
    start = time.time()
    for n in range(10):
        image_embeddings, interm_embeddings = segment_anything.encode_image(image)
    print("Time:",time.time()-start)
    print(image_embeddings.shape)
    print(interm_embeddings.shape)