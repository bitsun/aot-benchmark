import argparse
import grpc
from concurrent import futures
import tracker_grpc_server_pb2 as pb2
import tracker_grpc_server_pb2_grpc as pb2_grpc
from AOTTracker import AOTTracker,ModelType,TrackerType
import json
import traceback
import numpy as np
from enum import Enum
#from mobile_sam import SamPredictor,sam_model_registry 
import sam,sam_hq,efficientvit_sam
import torch
import cv2
import logging
from logging.handlers import TimedRotatingFileHandler
from sam_registry import sam_registry
logger:logging.Logger = logging.getLogger(__name__)
class SegmentAnythingService(pb2_grpc.SegmentAnything) :
    def __init__(self,config_path) :
        with open(config_path) as f:
            config = json.load(f)
        assert config['SAM'] is not None
        self.sam = sam_registry.get(config['SAM']['type'],**config['SAM']['args'])
        #self.model_type = config['SAM']['model_type']
        #self.model_path = config['SAM']['model_path']
        #self.decoder_onnx_path = config['SAM']['decoder_onnx_path']
        self.decoder_onnx_path = self.sam.decoder_onnx_path
        #self.sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        #if torch.backends.cuda.is_built() :
        #    self.sam.to("cuda")
        #self.predictor = SamPredictor(self.sam)
        self.lock = threading.Lock()
    def set_image(self,request,context):
        try:
            #decode bytes to image with jpeg decoding method
            image_bgr = np.frombuffer(request.data, np.uint8)
            image_bgr = np.reshape(image_bgr,(request.height,request.width,request.num_channels))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            raise Exception("not implemented")
            return pb2.BooleanResponse(success=True,error_msg="")
        except Exception as e:
            error_msg=traceback.format_exc()
            logger.error("error in set_image:{}".format(error_msg))
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
    def encode_image(self,request,context):
        self.lock.acquire()
        try:
            #decode bytes to image with jpeg decoding method
            image_bgr = np.frombuffer(request.data, np.uint8)
            image_bgr = np.reshape(image_bgr,(request.height,request.width,request.num_channels))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            #self.predictor.set_image(image_rgb)
            result_bytes = self.sam.encode_image(image_rgb).astype(np.float16).tobytes()
            #result_bytes = self.predictor.features.detach().cpu().to(torch.float16).numpy().tobytes()
            return pb2.ImageEmbeddingResponse(success=True,error_msg="",data=result_bytes)
        except Exception as e:
            error_msg=traceback.format_exc()
            logger.error("error in encode_image:{}".format(error_msg))
            return pb2.ImageEmbeddingResponse(success=False,error_msg=error_msg,data=None)
        finally:
            self.lock.release()
    def get_decoder_onnx_model(self,request,context):
        try:
            #read the whole file as bytes from self.decoder_onnx_path
            with open(self.decoder_onnx_path,"rb") as f:
                onnx_bytes = f.read()
            length = len(onnx_bytes)
            n = 0
            while n<length:
                if n+1024*1024>=length:
                    yield pb2.OnnxFileSegment(data=onnx_bytes[n:length],error_msg="",remaining_bytes=0)
                yield pb2.OnnxFileSegment(data=onnx_bytes[n:min(n+1024*1024,length)],error_msg="",remaining_bytes=length-n)
                n+=1024*1024
        except Exception as e:
            error_msg=traceback.format_exc()
            return pb2.OnnxFileSegment(data=None,error_msg=error_msg,remaining_bytes=0)
    def get_sam_type(self,request,context):
        #create enum from string
        try:
            sam_type = pb2.SamType.Value(self.sam.sam_type)
            return pb2.SamTypeResponse(sam_type=sam_type,error_msg="")
        except Exception as e:
            error_msg=traceback.format_exc()
            return pb2.SamTypeResponse(sam_type=pb2.SamType.Unknown,error_msg=error_msg)       
class AoTTrackerService(pb2_grpc.TrackAnythingServicer) :
    def __init__(self,tracker_config_path):
        self.tracker_config_path = tracker_config_path
        with open(tracker_config_path) as f:
            config = json.load(f)
        assert config['tracker'] is not None
        tracker_class = globals()[config['tracker']['type']]
        self.tracker = tracker_class(**config['tracker']['args'])
    def set_template_mask(self,request,context):
        try:
            if request.mask.num_channels != 1:
                raise Exception("mask must be single channel")
            if request.mask.height != request.frame.height or request.mask.width != request.frame.width:
                raise Exception("mask and image must have same size")
            #get bgr image
            #get the rgb image
            image_bgr = np.frombuffer(request.frame.data, np.uint8)
            image_bgr = np.reshape(image_bgr,(request.frame.height,request.frame.width,request.frame.num_channels))
            #get the mask
            mask = np.frombuffer(request.mask.data, np.uint8)
            mask = np.reshape(mask,(request.mask.height,request.mask.width))
            #make mask continuous
            mask = np.ascontiguousarray(mask)
            self.tracker.add_reference_frame(image_bgr,mask)
            return pb2.BooleanResponse(success=True)
        except Exception as e:
            error_msg=traceback.format_exc()
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
    def clear(self,request,context):
        try:
            self.tracker.clear()
            return pb2.BooleanResponse(success=True,error_msg="")
        except Exception as e:
            error_msg=traceback.format_exc()
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
    def freeze(self,request,context):
        try:
            self.tracker.freeze()
            return pb2.BooleanResponse(success=True,error_msg="")
        except Exception as e:
            error_msg=traceback.format_exc()
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
    def track(self,request,context):
        try:
            image_bgr = np.frombuffer(request.data, np.uint8)
            image_bgr = np.reshape(image_bgr,(request.height,request.width,request.num_channels))
            mask,prob = self.tracker.track(image_bgr)
            mask = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            prob = prob.squeeze(0).detach().cpu().numpy()
            #mask = np.zeros((request.height,request.width),dtype=np.uint8)
            #prob = np.zeros((1,request.height,request.width),dtype=np.float32)
            scores = np.max(prob,axis=0)
            mask_data = mask.tobytes()
            mask_image = pb2.Image(height=mask.shape[0],width=mask.shape[1],num_channels=1,data=mask_data)
            score_image = pb2.Image(height=scores.shape[0],width=scores.shape[1],num_channels=1,data=scores.tobytes())
            return pb2.TrackResponse(success=True,scores=score_image,mask=mask_image,error_msg="")
        
        except Exception as e:
            error_msg=traceback.format_exc()
            return pb2.TrackResponse(success=False,scores=None,mask=None,error_msg=error_msg)

import threading
from datetime import datetime
import logging
class State(Enum):
    IDLE = 0
    RESERVED = 1
    BUSY = 2

class StatefulTracker:
    """
    a stateful tracker is a tracker with state, which can be reserved for a client for a period of time.
    If the client does not use the tracker for a period of time, the tracker will be reset
    and can be reserved by other clients
    """
    def __init__(self,tracker):
        self.tracker = tracker
        self.state = State.IDLE
        #the minimal value of int64
        self.token = None
        self.last_track_time = None
    def reset(self):
        self.tracker.clear()
        self.state = State.IDLE
        self.last_track_time = None
        self.token = None

class StatefulAoTTrackerService(pb2_grpc.StatefulTrackerService) :
    def __init__(self,tracker_config_path,num_instances=1):
        self.trackers = []
        self.tracker_config_path = tracker_config_path
        with open(tracker_config_path) as f:
            config = json.load(f)
        assert config['tracker'] is not None
        tracker_class = globals()[config['tracker']['type']]
        for n in range(num_instances):
            tracker = tracker_class(**config['tracker']['args'])
            self.trackers.append(StatefulTracker(tracker))
        self.lock = threading.Lock()

    def next_available_tracker_instance(self,request,context):
        """
        reserve a tracker instance for the client
        """
        #lock the following code
        self.lock.acquire()
        try:
            for i,tracker in enumerate(self.trackers):
                if tracker.state==State.IDLE:
                    tracker.state = State.RESERVED
                    token = int(datetime.now().timestamp()*1000000)
                    tracker.token = token
                    tracker.last_track_time = datetime.now()
                    logger.info("tracker {} is reserved by client with token {}".format(i,token))
                    return pb2.InstanceResponse(instance_id=i,token = token, error_msg="")
        finally:
            self.lock.release()
        return pb2.InstanceResponse(instance_id=-1,error_msg="no available instance")

    def clear(self,request,context):
        try:
            if request.instance_id<0 or request.instance_id>=len(self.trackers):
                raise Exception("invalid instance id")
            if request.token!=self.trackers[request.instance_id].token:
                raise Exception("invalid token")
            self.trackers[request.instance_id].tracker.clear()
            logger.info("tracker {} is clared".format(request.instance_id))
            return pb2.BooleanResponse(success=True,error_msg="")
        except Exception as e:
            error_msg=traceback.format_exc()
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
    def set_template_mask(self,request,context):
        try:
            if request.instance_id<0 or request.instance_id>=len(self.trackers):
                raise Exception("invalid instance id")
            if request.token!=self.trackers[request.instance_id].token:
                raise Exception("invalid token")
            if request.mask.num_channels != 1:
                raise Exception("mask must be single channel")
            if request.mask.height != request.frame.height or request.mask.width != request.frame.width:
                raise Exception("mask and image must have same size")
            #get bgr image
            #get the rgb image
            image_bgr = np.frombuffer(request.frame.data, np.uint8)
            image_bgr = np.reshape(image_bgr,(request.frame.height,request.frame.width,request.frame.num_channels))
            #get the mask
            mask = np.frombuffer(request.mask.data, np.uint8)
            mask = np.reshape(mask,(request.mask.height,request.mask.width))
            stateful_tracker = self.trackers[request.instance_id]
            stateful_tracker.tracker.add_reference_frame(image_bgr,mask)
            stateful_tracker.state = State.BUSY
            stateful_tracker.last_track_time = datetime.now()
            logger.info("tracker {} is set with template mask".format(request.instance_id))
            return pb2.BooleanResponse(success=True)
        except Exception as e:
            error_msg=traceback.format_exc()
            logger.error("error in set_template_mask:{}".format(error_msg))
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
    
    def freeze(self,request,context):
        try:
            if request.instance_id<0 or request.instance_id>=len(self.trackers):
                raise Exception("invalid instance id")
            if request.token!=self.trackers[request.instance_id].token:
                raise Exception("invalid token")
            self.trackers[request.instance_id].tracker.freeze()
            logger.info("tracker {} is frozen".format(request.instance_id))
            return pb2.BooleanResponse(success=True,error_msg="")
        except Exception as e:
            error_msg=traceback.format_exc()
            logger.error("error in freeze tracker {}:{}".format(request.instance_id,error_msg))
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
    def finish(self,request,context):
        try:
            if request.instance_id<0 or request.instance_id>=len(self.trackers):
                raise Exception("invalid instance id")
            if request.token!=self.trackers[request.instance_id].token:
                raise Exception("invalid token")
            self.lock.acquire()
            self.trackers[request.instance_id].reset()
            logger.info("tracker {} is finished with tracking".format(request.instance_id))
            return pb2.BooleanResponse(success=True,error_msg="")
        except Exception as e:
            error_msg=traceback.format_exc()
            logger.error("error in finish tracker {}:{}".format(request.instance_id,error_msg))
            return pb2.BooleanResponse(success=False,error_msg=error_msg)
        finally:
            self.lock.release()
        
    def track(self,request,context):
        try:
            if request.instance_id<0 or request.instance_id>=len(self.trackers):
                raise Exception("invalid instance id")
            if request.token!=self.trackers[request.instance_id].token:
                logger.error("invalid token,input token:{},expected token:{}".format(request.token,self.trackers[request.instance_id].token))
                raise Exception("invalid token")
            image_bgr = np.frombuffer(request.frame.data, np.uint8)
            image_bgr = np.reshape(image_bgr,(request.frame.height,request.frame.width,request.frame.num_channels))
            mask,prob = self.trackers[request.instance_id].tracker.track(image_bgr)
            mask = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
            prob = prob.squeeze(0).detach().float().cpu().numpy()
            #mask = np.zeros((request.frame.height,request.frame.width),dtype=np.uint8)
            #prob = np.zeros((1,request.frame.height,request.frame.width),dtype=np.float32)
            scores = np.max(prob,axis=0)
            mask_data = mask.tobytes()
            mask_image = pb2.Image(height=mask.shape[0],width=mask.shape[1],num_channels=1,data=mask_data)
            score_image = pb2.Image(height=scores.shape[0],width=scores.shape[1],num_channels=1,data=scores.tobytes())
            self.trackers[request.instance_id].last_track_time = datetime.now()
            return pb2.TrackResponse(success=True,scores=score_image,mask=mask_image,error_msg="")
        
        except Exception as e:
            logger.error("error in track:{}".format(e))
            error_msg=traceback.format_exc()
            return pb2.TrackResponse(success=False,scores=None,mask=None,error_msg=error_msg)

def active_tracker_monitor(ticker,tracker_service):
    while not ticker.wait(15):
        now = datetime.now()
        try:
            tracker_service.lock.acquire()
            for i,tracker in enumerate(tracker_service.trackers):
                if (tracker.state==State.BUSY or tracker.state == State.RESERVED) and (now-tracker.last_track_time).total_seconds()>10:
                    #log the reset
                    logger.info("reset tracker {} because it is idle for more than 10 sec".format(i))
                    tracker.reset()
        finally:
            tracker_service.lock.release()

def serve(max_workers,port,config_path):  
    tracker_service =StatefulAoTTrackerService(config_path,max_workers)
    sam_service = SegmentAnythingService(config_path)
    ticker = threading.Event()
    watchdog = threading.Thread(target=active_tracker_monitor,args=(ticker,tracker_service))
    watchdog.start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb2_grpc.add_StatefulTrackerServiceServicer_to_server(tracker_service,server)
    pb2_grpc.add_SegmentAnythingServicer_to_server(sam_service,server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    logging.info("server started, listening port {}".format(port))
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segment anything server")
    parser.add_argument('--max-workers',type=int,default=2,help='max number of workers')
    parser.add_argument('--port',type=int,default=50051,help='port number')
    parser.add_argument('--config-path',type=str,required=True,help='tracking model path')
    args = parser.parse_args()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler = TimedRotatingFileHandler('aot_tracker_{}.log'.format(args.port), 
                                   when='midnight',
                                   backupCount=10)
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    serve(args.max_workers,args.port,args.config_path)