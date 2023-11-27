import numpy as np
from AOTTracker import AOTTracker,TrackerType,ModelType
import argparse
import os
import cv2
import PIL
from demo import _palette,color_palette, overlay
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="AOT tracker")

    tracker = AOTTracker("E:\\Data\\Model\\Tracking\\AOT\\DeAOTS_PRE_YTB_DAV.pth",
                         max_size=480*1.3,tracker_type="DeAOT",model_type="DeAOTS")
    input_dir = 'E:\\Data\\LogoWorkDir\\test1'
    visualize_mask = False
    visualize_overlay = True
    #red all jpg images in this folder
    image_files = os.listdir(input_dir)
    #remove all the files that do not have extension .jpg
    image_files = [os.path.join(input_dir,f) for f in image_files if f.endswith('.jpg')]
    image_files.sort()
    frames = []

    label = PIL.Image.open('E:\\Data\\LogoWorkDir\\test1\\0001.png')
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
        pred_label,prob = tracker.track(frame)
        prob = prob.squeeze().cpu().numpy()
        pred_label = pred_label.squeeze().cpu().numpy().astype('uint8')
        #pred_label[np.where(prob<0.5)]=0
        if visualize_mask:
            mask_img = PIL.Image.fromarray(pred_label).convert('P')
            mask_img.putpalette(_palette)
            #convert label to rgb
            mask_img = mask_img.convert('RGB')
            #convert to numpy array
            mask_img = np.array(mask_img).astype(np.uint8)
            mask_img = mask_img[:,:,::-1]
            cv2.imshow("mask",mask_img)
            cv2.waitKey(0)
        if visualize_overlay:
            overlay_image = overlay(frame,pred_label,color_palette)
            cv2.imshow("overlay",overlay_image)
            cv2.waitKey(0)
            # #format string pad zero
            # i = str(i).zfill(5)
            # label.save(os.path.join('./result', '{}.png'.format(i)))