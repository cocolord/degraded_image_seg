from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os

config_file = 'configs/ocrnet/ocrnet_s101-d8_512x1024_80k_b16_cityscapes.py'
checkpoint_file = 'work_dirs/ocrnet_s101-d8_512x1024_80k_b16_cityscapes/iter_80000.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# # visualize the results in a new window
# model.show_result(img, result, show=True)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('data/buaa_darknight/IMG_4875.mov')
output_dir = './buaa_night'
for idx,frame in enumerate(video):
    result = inference_segmentor(model, frame)
    model.show_result(frame, result, wait_time=1,out_file=os.path.join(output_dir,str(idx))+'.jpg')