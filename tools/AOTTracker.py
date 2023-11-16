
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
class AOTTracker:
    def __init__(self,model_path:str,tracker_type="AOT",
                 max_size=480 * 1.3,device='cuda'):
        """
        constructor of AOTTracker
        model_path: path to the model
        """
        self.model_path = model_path
        self.tracker_type = tracker_type
        engine_config = importlib.import_module('configs.'+'pre_ytb_dav')
        cfg = engine_config.EngineConfig('default', "deaotl")
        cfg.TEST_CKPT_PATH = model_path
        cfg.TEST_MIN_SIZE = None
        cfg.TEST_MAX_SIZE = max_size * 800. / 480.
        if tracker_type=="AOT":
            model = AOT(cfg, encoder=cfg.MODEL_ENCODER)
        elif tracker_type=="DeAOT":
            model = DeAOT(cfg, encoder=cfg.MODEL_ENCODER)
        self.model, _ = load_network(model, model_path, 0)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                          phase='eval',
                          aot_model=model,
                          gpu_id=0,
                          long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

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
        #determin the number of objects
        obj_nums = np.max(label)
        self.engine.restart_engine()
        frame_tensor = self.preprocess(frame)
        frame_tensor = frame_tensor.unsqueeze(0).cuda()
        label_tensor = torch.from_numpy(label).float().cuda().unsqueeze(0).unsqueeze(0)
        if label_tensor.shape[0]!=frame_tensor.shape[0] or \
            label_tensor.shape[1]!=frame_tensor.shape[1]:
            label_tensor = F.interpolate(label_tensor,size=frame_tensor.size()[2:],
                                                  mode="nearest")
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
        with torch.no_grad():
            self.engine.match_propogate_one_frame(frame_tensor)
            pred_logit = self.engine.decode_current_logits(
                        (frame.shape[0], frame.shape[1]))
            pred_prob = torch.softmax(pred_logit, dim=1)
            result = torch.max(pred_prob, dim=1,keepdim=True)
            pred_prob = result.values
            pred_label = result.indices.float()
            # update memory
            if not self.freezed:
                _pred_label = F.interpolate(pred_label,size=self.engine.input_size_2d,
                                                mode="nearest")
                self.engine.update_memory(_pred_label)
        return pred_label,pred_prob


if __name__ == '__main__':
    import argparse
    import os
    import cv2
    import PIL
    from demo import _palette
    parser = argparse.ArgumentParser(description="AOT tracker")

    tracker = AOTTracker("E:\\Data\\Model\\Tracking\\AOT\\DeAOTL_PRE_YTB_DAV.pth",
                         max_size=480*1.3,tracker_type="DeAOT")
    input_dir = './datasets/Demo/images/1001_3iEIq5HBY1s'

    #red all jpg images in this folder
    image_files = os.listdir(input_dir)
    #remove all the files that do not have extension .jpg
    image_files = [os.path.join(input_dir,f) for f in image_files if f.endswith('.jpg')]
    image_files.sort()
    frames = []

    label = PIL.Image.open('./datasets/Demo/masks/1001_3iEIq5HBY1s/00002058.png')
    label = np.array(label, dtype=np.uint8)
    label[np.where(label>=10)]=0
    frame = cv2.imread(image_files[0])
    frame = frame[:,:,::-1]
    tracker.add_reference_frame(frame,label)
    print('added reference frame')
    for i,image_file in enumerate(image_files):
        if i==0:
            continue
        frame = cv2.imread(image_file)
        #frame = cv2.resize(frame,(1024,576),interpolation=cv2.INTER_LINEAR)
        l,prob = tracker.track(frame)
        # label = PIL.Image.fromarray(
        #     label.squeeze().cpu().numpy().astype(
        #                 'uint8')).convert('P')
        # label.putpalette(_palette)
        # #format string pad zero
        # i = str(i).zfill(5)
        # label.save(os.path.join('./result', '{}.png'.format(i)))