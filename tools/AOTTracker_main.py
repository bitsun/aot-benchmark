import os
import numpy as np
from AOTTracker import AOTTracker,TrackerType,ModelType
import argparse
import os
import cv2
import PIL
from demo import _palette,color_palette, overlay

def init_tracker(input_dir,tracker):
    image_files = ["0001.jpg","brandwall_01.jpg","referee_01.jpg","perimeter_01.jpg"]
    #image_files = ["perimeter_01.jpg"]
    for idx,image_file in enumerate(image_files):
        label = PIL.Image.open(os.path.join(input_dir,image_file.replace('.jpg','.png')))
        label = np.array(label, dtype=np.uint8)
        label[np.where(label>=10)]=0
        frame = cv2.imread(os.path.join(input_dir,image_file))
        frame = frame[:,:,::-1]
        label_offset = tracker.num_ref_obj
        tracker.add_reference_frame(frame,label,label_offset,idx==0)
    tracker.freeze()

def test_tracker(input_dir,tracker):
    image_files = ["perimeter_03.jpg"]
    for image_file in image_files:
        frame = cv2.imread(os.path.join(input_dir,image_file))
        frame = frame[:,:,::-1]
        pred_label,prob = tracker.track(frame)
        prob = prob.squeeze().cpu().numpy()
        pred_label = pred_label.squeeze().cpu().numpy().astype('uint8')
        #pred_label[np.where(prob<0.7)]=0
        mask_img = PIL.Image.fromarray(pred_label).convert('P')
        mask_img.putpalette(_palette)
        mask_img = mask_img.convert('RGB')
        mask_img = np.array(mask_img).astype(np.uint8)
        mask_img = mask_img[:,:,::-1]
        cv2.imshow("mask",mask_img)
        overlay_image = overlay(frame,pred_label,color_palette)
        cv2.imshow("overlay",overlay_image)
        cv2.waitKey(0)
        cv2.waitKey(0)

if __name__ == '__main__':
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(description="AOT tracker")

    tracker = AOTTracker("e:\\Data\\Model\\Tracking\\AOT\\DeAOTS_PRE_YTB_DAV.pth",
                         max_size=480*1.3,tracker_type="DeAOT",model_type="DeAOTS",half=False)
    #get the current dir of the script
    script_path = os.path.abspath(__file__)

    # Get the directory of the script
    script_dir = os.path.dirname(script_path)
    input_dir = os.path.join(script_dir,'../datasets/ivitec')
    visualize_mask = False
    visualize_overlay = True
    #init_tracker(input_dir,tracker)
    #test_tracker(input_dir,tracker)
    #red all jpg images in this folder
    image_files = os.listdir(input_dir)
    #remove all the files that do not have extension .jpg
    image_files = [os.path.join(input_dir,f) for f in image_files if f.endswith('.jpg')]
    image_files.sort()
    frames = []

    label = PIL.Image.open(os.path.join(input_dir,"0001.png"))
    label = np.array(label, dtype=np.uint8)
    label[np.where(label>=10)]=0
    frame = cv2.imread(image_files[0])
    frame = frame[:,:,::-1]
    batch_size = 1
    tracker.set_batch_size(batch_size)

    tracker.add_reference_frame(frame,label)
    tracker.freeze()
    
    print('added reference frame')
    frames = []
    for n in range(50):
        for i,image_file in enumerate(image_files):
            if i==0:
                continue
            frame = cv2.imread(image_file)
            frames.append(frame)
    #measure the time
    import time
    start = time.time() 
    frames1 = []
    for k,frame in enumerate(frames):
        frame = cv2.resize(frame,(1024,576),interpolation=cv2.INTER_LINEAR)
        frames1.append(frame)
        if len(frames1)<batch_size:
            continue
        
        pred_label,prob = tracker.track(frames1)
        prob = prob.squeeze(0).cpu().numpy()
        pred_label = pred_label.squeeze(0).cpu().numpy().astype('uint8')
        pred_label[np.where(prob<0.7)]=0
        if visualize_mask:
            for i in range(batch_size):
                mask_img = PIL.Image.fromarray(pred_label[i]).convert('P')
                mask_img.putpalette(_palette)
                #convert label to rgb
                mask_img = mask_img.convert('RGB')
                #convert to numpy array
                mask_img = np.array(mask_img).astype(np.uint8)
                mask_img = mask_img[:,:,::-1]
                cv2.imshow("mask",mask_img)
                cv2.waitKey(0)
        if visualize_overlay:
            for i in range(batch_size):
                overlay_image = overlay(frames1[i],pred_label[i],color_palette)
                cv2.imshow("overlay",overlay_image)
                cv2.waitKey(0)
            #format string pad zero
            #i = str(i).zfill(5)
            #label.save(os.path.join('./result', '{}.png'.format(i)))
        frames1.clear()
    end = time.time()
    print(end - start)