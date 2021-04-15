from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
from PIL import Image
import numpy as np
import time
INFER_ON_CPU = True
SHOW = False
#import scipy.misc as misc
pwd_dir = '/home/juan/Donglusen/seg_output/hrnet_output'
config_file = './configs/deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes.py'
checkpoint_file = 'deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth'



# build the model from a config file and a checkpoint file
if INFER_ON_CPU:
    model = init_segmentor(config_file, checkpoint_file, device='cpu')
else:
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
path_img_list = ["PTZ//continuousPan/","PTZ//intermittentPan/","PTZ//twoPositionPTZCam/","PTZ//zoomInZoomOut/","cameraJitter//badminton/","cameraJitter//boulevard/","cameraJitter//sidewalk/","cameraJitter//traffic/"]
if INFER_ON_CPU:
    norm_cfg = dict(type='BN', requires_grad=True)
    model.backbone.norm_cfg = norm_cfg
    model.decode_head.norm_cfg = norm_cfg
    model.auxiliary_head.norm_cfg = norm_cfg
# # test a single image and show the results
img_dir = '/home/juan/Donglusen/exp/zls/cameraJitter/traffic/input/'  # or img = mmcv.imread(img), which will only load it once
output_dir = os.path.join(pwd_dir,'traffic/')
# os.mkdir(os.path.join(pwd_dir,output_dir))
files = os.listdir(img_dir)
print(files)
avg_time = 0.0
for f in files:
    img = os.path.join(img_dir,f)
    st = time.time()
    result = inference_segmentor(model, img)
    ed = time.time()
    avg_time += ed - st
    print(config_file , 'cost', ed - st , 's')
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
    if SHOW:
        model.show_result(img, result, out_file=os.path.join(output_dir,outname) ,
        palette =[[0, 0, 0], [128, 0,0], [64, 0, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [0, 128, 0], [192, 0, 0], [135, 206, 235], [192, 128, 0],
                [64, 0,128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [255, 255, 0]])
print("avg cost %s per image, %s fps" % (avg_time/ len(files), 1.0/(avg_time/ len(files)) ) )
# test a video and show the results