from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os

INFERENCE_IMAGE = False

config_file = 'configs/SETR/SETR_PUP_DeiT_convfuseunet_768x768_160k_cityscapes_bs_8.py'
checkpoint_file = 'work_dirs/pup_convfuse_unet_160000_bs8.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

if INFERENCE_IMAGE:
    # # test a single image and show the results
    img = '/home/juan/project/edlines_line_segment_detection/pics/1.png'  # or img = mmcv.imread(img), which will only load it once
    result = inference_segmentor(model, img)
    # # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='project/1_seg.jpg')
else:
    # test a video and show the results
    img_dir = '/home/juan/Donglusen/exp/zls/baseline/highway/input'
    output_dir = '/home/juan/Donglusen/exp/zls/baseline/highway/seg_all'
    img_list = sorted(os.listdir(img_dir))
    for img_instance in img_list:
        file_name = os.path.join(img_dir, img_instance)
        result = inference_segmentor(model, file_name)
        model.show_result(file_name, result, out_file = os.path.join(output_dir, img_instance.split('.')[0]+ '.png'))
        