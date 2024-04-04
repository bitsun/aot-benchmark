
import importlib
import sys
import os
sys.path.append('.')
sys.path.append('..')

from networks.models.aot import AOT
from networks.models.deaot import DeAOT
from networks.engines import build_engine
from utils.checkpoint import load_network
from torchvision import transforms
import dataloaders.video_transforms as tr
import numpy as np
import torch
import torch.nn.functional as F
import cv2
#define enum  type
from enum import Enum
class TrackerType(Enum):
    aot = 1
    deaot = 2
    @staticmethod
    def from_str(label):
        if label.lower()=='aot':
            return TrackerType.aot
        elif label.lower()=='deaot':
            return TrackerType.deaot
        else:
            raise ValueError("tracker type is not supported")
class ModelType(Enum):
    aott = 1,
    aots = 2,
    aotb = 3,
    aotl = 4,
    r50_aotl = 5,
    swinb_aotl = 6,
    deaott = 7,
    deaots = 8,
    deaotb = 9,
    deaotl = 10,
    r50_deaotl = 11,
    swinb_deaotl = 12
    @staticmethod
    def from_str(label):
        # if label.lower()=="aott":
        #     return ModelType.AOTT
        # elif label.lower()=="aots":
        #     return ModelType.AOTS
        # elif label.lower()=="aotb":
        #     return ModelType.AOTB
        # elif label.lower()=="aotl":
        #     return ModelType.AOTL
        # elif label.lower()=="r50_aotl":
        #     return ModelType.R50_AOTL
        # elif label.lower()=="swinb_aotl":
        return ModelType[label.lower()]


class AOTTracker:
    def __init__(self,model_path:str,tracker_type="DeAOT",model_type="DeAOTL",
                 max_size=480 * 1.3,device='cuda',half=False):
        """
        constructor of AOTTracker
        model_path: path to the model
        tracker_type: the type of the tracker, it can be "AOT" or "DeAOT"
        model_type: the type of the model, it can be "AOTT","AOTS","AOTB","AOTL",
                    "R50_AOTL","SWINB_AOTL","DEAOTT","DEAOTS","DEAOTB","DEAOTL",
                    "R50_DEAOTL","SWINB_DEAOTL"
        max_size: the maximum resolution of the resized input image
        """
        self.model_path = model_path
        #parse enum type

        self.tracker_type = TrackerType.from_str(tracker_type)
        self.model_type = ModelType.from_str(model_type)
        #check if model type and tracker type is compatible
        if tracker_type==TrackerType.aot:
            assert model_type<= ModelType.swinb_aotl
        elif tracker_type==TrackerType.deaot:
            assert model_type<= ModelType.swinb_deaotl and model_type>=ModelType.deaott
        engine_config = importlib.import_module('configs.'+'pre_ytb_dav')
        #get the string of model type
        model_type = model_type.lower()
        cfg = engine_config.EngineConfig('default', model_type)
        cfg.TEST_CKPT_PATH = model_path
        cfg.TEST_MIN_SIZE = None
        cfg.TEST_MAX_SIZE = max_size * 800. / 480.
        if self.tracker_type==TrackerType.aot:
            model = AOT(cfg, encoder=cfg.MODEL_ENCODER)
        elif self.tracker_type==TrackerType.deaot:
            model = DeAOT(cfg, encoder=cfg.MODEL_ENCODER)
        else:
            raise ValueError("tracker type is not supported")
        self.model, _ = load_network(model, model_path, 0)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                          phase='eval',
                          aot_model=model,
                          gpu_id=0,
                          long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
        if isinstance(half,str):
            self.half = eval(half)
        else:
            self.half = half
        if self.half:
            self.model = self.model.half()
            self.model.cfg.half = True
            self.engine = self.engine.half()
            self.engine.cfg.half = True
        # self.transform = transforms.Compose([
        #     tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
        #                      cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
        #                      cfg.MODEL_ALIGN_CORNERS),
        #     tr.MultiToTensor()
        # ])
        self.multi_scale = cfg.TEST_MULTISCALE
        self.max_short_edge = cfg.TEST_MIN_SIZE
        self.max_long_edge = cfg.TEST_MAX_SIZE
        self.align_corners = cfg.MODEL_ALIGN_CORNERS
        self.model.eval()
        self.engine.restart_engine()
        self.freezed = False
        self.max_stride = 16
    def preprocess(self, image):
        """
        do a preprocessing of input image
        """
        assert isinstance(image, np.ndarray) and image.ndim == 3\
            and image.shape[2] == 3 and image.dtype == np.uint8
        h, w = image.shape[:2]
        scale = 1
        # restrict short edge
        sc = 1.
        if self.max_short_edge is not None:
            if h > w:
                short_edge = w
            else:
                short_edge = h
            if short_edge > self.max_short_edge:
                sc *= float(self.max_short_edge) / short_edge
        new_h, new_w = sc * h, sc * w

        # restrict long edge
        sc = 1.
        if self.max_long_edge is not None:
            if new_h > new_w:
                long_edge = new_h
            else:
                long_edge = new_w
            if long_edge > self.max_long_edge:
                sc *= float(self.max_long_edge) / long_edge

        new_h, new_w = sc * new_h, sc * new_w

        new_h = int(new_h * scale)
        new_w = int(new_w * scale)

        if self.align_corners:
            if (new_h - 1) % self.max_stride != 0:
                new_h = int(
                    np.around((new_h - 1) / self.max_stride) *
                    self.max_stride + 1)
            if (new_w - 1) % self.max_stride != 0:
                new_w = int(
                    np.around((new_w - 1) / self.max_stride) *
                    self.max_stride + 1)
        else:
            if new_h % self.max_stride != 0:
                new_h = int(
                    np.around(new_h / self.max_stride) * self.max_stride)
            if new_w % self.max_stride != 0:
                new_w = int(
                    np.around(new_w / self.max_stride) * self.max_stride)

        if new_h == h and new_w == w:
            tmp = image.astype(np.float32)
            tmp = tmp / 255.
            tmp -= (0.485, 0.456, 0.406)
            tmp /= (0.229, 0.224, 0.225)
            tmp = tmp.transpose((2, 0, 1))
            return torch.from_numpy(tmp).cuda()
        
        tmp = cv2.resize(image,dsize=(new_w, new_h),interpolation=cv2.INTER_CUBIC)
        tmp = tmp.astype(np.float32)
        tmp = tmp / 255.
        tmp -= (0.485, 0.456, 0.406)
        tmp /= (0.229, 0.224, 0.225)
        tmp = tmp.transpose((2, 0, 1))
        return torch.from_numpy(tmp)
    
    def clear(self):
        """
        clear the tracker,all the reference frames and masks and internal memory will be cleared
        """
        self.engine.restart_engine()

    def freeze(self):
        """
        freeze the tracker,internal memory will not be updated when the tracking is running
        """
        self.freezed = True

    def add_reference_frame(self,frame,label):
        """
        initialize the tracker with a reference frame and its mask
        frame: a numpy array of the reference frame
        label:a numpy array of the mask of the reference frame
        """
        #it should be a rgb image
        assert (isinstance(frame,np.ndarray) and frame.ndim==3\
            and frame.shape[2]==3 and frame.dtype==np.uint8)
        assert (isinstance(label,np.ndarray) and label.ndim==2\
            and label.dtype==np.uint8)
        self.clear()
        #determin the number of objects
        obj_nums = np.max(label)
        if obj_nums==0:
            raise ValueError("no object in the reference frame")
        self.engine.restart_engine()
        frame_tensor = self.preprocess(frame)
        frame_tensor = frame_tensor.unsqueeze(0).cuda()
        label_tensor = torch.from_numpy(label).float().cuda().unsqueeze(0).unsqueeze(0)
        if label_tensor.shape[0]!=frame_tensor.shape[0] or \
            label_tensor.shape[1]!=frame_tensor.shape[1]:
            label_tensor = F.interpolate(label_tensor,size=frame_tensor.size()[2:],
                                                  mode="nearest")
        if self.half:
            frame_tensor = frame_tensor.half()
            label_tensor = label_tensor.half()
        self.engine.add_reference_frame(frame_tensor,label_tensor,
                                        frame_step=0,obj_nums=obj_nums)
        self.freezed = False
    def track(self,frame):
        """
        track the objects in the frame
        frame: a numpy array of the frame
        return: a tuple. The 1st element is the mask label which has the same size 
               as the input frame. The 2nd element is the mask confidences score map
        """
        assert (isinstance(frame,np.ndarray) and frame.ndim==3\
            and frame.shape[2]==3 and frame.dtype==np.uint8)
        frame_tensor = self.preprocess(frame).cuda().unsqueeze(0)
        if self.half:
            frame_tensor = frame_tensor.half()
        with torch.no_grad():
            self.engine.match_propogate_one_frame(frame_tensor)
            pred_logit = self.engine.decode_current_logits(
                        (frame.shape[0], frame.shape[1]))
            pred_prob = torch.softmax(pred_logit, dim=1)
            result = torch.max(pred_prob, dim=1,keepdim=True)
            pred_prob = result.values
            if self.half:
                pred_label = result.indices.half()
            else:
                pred_label = result.indices.float()
            # update memory
            if not self.freezed:
                _pred_label = F.interpolate(pred_label,size=self.engine.input_size_2d,
                                                mode="nearest")
                self.engine.update_memory(_pred_label)
        return pred_label,pred_prob


