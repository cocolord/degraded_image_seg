from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
from PIL import Image
import numpy as np
#import scipy.misc as misc
pwd_dir = '/home/juan/Donglusen/seg_output/hrnet_output'
config_file = './configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py'
checkpoint_file = '/home/juan/Donglusen/seg_output/hrnet_output/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:1')
path_img_list = ["PTZ//continuousPan/","PTZ//intermittentPan/","PTZ//twoPositionPTZCam/","PTZ//zoomInZoomOut/","cameraJitter//badminton/","cameraJitter//boulevard/","cameraJitter//sidewalk/","cameraJitter//traffic/"]
    
# # test a single image and show the results
img_dir = '/home/juan/Donglusen/exp/zls/cameraJitter/traffic/input/'  # or img = mmcv.imread(img), which will only load it once
output_dir = os.path.join(pwd_dir,'traffic/')
# os.mkdir(os.path.join(pwd_dir,output_dir))
files = os.listdir(img_dir)
print(files)
for f in files:
    img = os.path.join(img_dir,f)
    result = inference_segmentor(model, img)
    #print(img.size())
    #array_res = np.array(result)
    #outmask = os.path.splitext(os.path.split(f)[-1])[0] + '.npy'
    #np.save(os.path.join(mask_dir,outmask),array_res)
    #print(array_res.size)
    #im = Image.fromarray(array_res)
    #print(type(array_res))
    # # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    outname = os.path.splitext(os.path.split(f)[-1])[0] + '.png'  # f.split('.')[0]
    #model.show_result(img, result, out_file=os.path.join(output_dir,outname))
    model.show_result(img, result, out_file=os.path.join(output_dir,outname) , 
    palette =[[0, 0, 0], [128, 0,0], [64, 0, 0], [128, 128, 0],
               [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
               [0, 128, 0], [192, 0, 0], [135, 206, 235], [192, 128, 0],
               [64, 0,128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
               [0, 64, 0], [128, 64, 0], [255, 255, 0]])
    #im.save(os.path.join(mask_dir,outname))

# test a video and show the results