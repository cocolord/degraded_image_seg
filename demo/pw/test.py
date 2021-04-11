from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os

config_file = '../../configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py'
checkpoint_file = 'ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
img = 'photos/000002.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# # visualize the results in a new window
# model.show_result(img, result, show=True)
# or save the visualization results to image files
model.show_result(img, result, out_file='photos/result.jpg')

# test a video and show the results