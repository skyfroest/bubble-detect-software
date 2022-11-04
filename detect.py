# 为了做ppt，不保存目标框，只保存拟合的椭圆。面积换算成实际值
# 添加时间性能测试
# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    python detect_ltl.py --save-txt --save-conf --conf-thres 0.5 --name 'data0716_exp6_thin_overlopdetect' --line-thickness 1 --hide-labels --weights './runs/train/exp6/weights/best.pt' --img 512 --source './data/data0716/img_716_thin'
    python detect.py --save-txt --save-conf --save-crop --conf-thres 0.5 --name 'databubble0616-crop' --line-thickness 1 --hide-labels --weights 'runs/train/databubble0604_3/weights/best.pt' --img 512 --source 'D:\\whitebubble\\machinelearning\\dataset\\data0616\\img_616'
"""
from __future__ import division, print_function, absolute_import
import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import torch
from utils.ellipsefit_ltl import ellipse_detect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync

import tflearn
import numpy as np
from utils import classificationModel3
from utils import hyperparameterModel
import tensorflow as tf

# 加载焦距判断模型
params = hyperparameterModel.hyperparameterModel()
tflearn.init_graph()
g = classificationModel3.createModel(params)
model_focus = tflearn.DNN(g, tensorboard_verbose=0)
model_focus.load('./models/model.tfl')

@torch.no_grad()
def detect_img(weights=ROOT / 'models/best.pt',  # model.pt path(s)
        source=ROOT/'data',  # file/dir/URL/glob, 0 for webcam
        img=512,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt = True,

        project=ROOT / 'runs',  # save results to project/name
        name='exp',  # save results to project/name
        ):

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    imgsz = [img, img]  # expand
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen, bubble_num = [0.0, 0.0, 0.0], 0, 0
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inferencel's
        pred = model(im)

        # NMS 非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image。 i是索引，det是pred中的每个值
            area_total = 0
            vol_total = 0
            bubble_num = 0
            seen += 1
            result_info = []
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(project / p.name)  # im.jpg
            txt_path = str(project / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            txt_result_path = str(project / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]  # print string
            img_save = im0.copy()  # 最后保存要保存img_save，避免在原图上修改时，改变crop，使得检测椭圆时图片中有拟合的其他椭圆/像素值
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()   # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # for *xyxy, conf, cls in reversed(det):
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                    #    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #    line = (cls, *xywh, conf)  # label format
                    #    with open(txt_path + '.txt', 'a') as f:
                    #        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    # Add bbox to image
                    x1 = (xywh[0]-xywh[2]/2)*img
                    x2 = (xywh[0]+xywh[2]/2)*img
                    y1 = (xywh[1]-xywh[3]/2)*img
                    y2 = (xywh[1]+xywh[2]/2)*img
                    crop = imc[int(y1):int(y2), int(x1):int(x2)]
                    # 判断是否在焦内
                    nor_crop = cv2.resize(crop, (64, 64))
                    focus_label = PredictLabel(nor_crop, model_focus)

                    if focus_label:
                        if save_txt:
                            line = (*xywh, conf)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        bubble_num += 1
                        # print('crop shape is: ', crop.shape)
                        if abs(x1-x2)<15 or abs(y1-y2)<15: # 动态的上采样倍数吧
                            times = 3
                        elif abs(x1-x2)<30 or abs(y1-y2)<30:
                            times = 2
                        else:
                            times = 1
                        _ellipse, area, vol = ellipse_detect(crop, times)
                        cv2.ellipse(img_save, [[_ellipse[0]+x1, _ellipse[1] + y1], [_ellipse[2], _ellipse[3]], _ellipse[4]], (0, 0, 255), 1)  # 椭圆坐标对应需要改 xy是反的
                        area_total += area
                        vol_total += vol

            area_total = area_total*0.060606*0.060606
            vol_total = vol_total*0.060606*0.060606*0.060606
            density_num = bubble_num/(30*30*3)
            density_vol = vol_total/(30*30*3)
            # print results
            #cv2.putText(img_save, '%d mm2' % area_total, (10, img - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255))
            #cv2.putText(img_save, '%d bubbles' % bubble_num, (10, img - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255))
            print(path, '气泡总个数 %d' % det.shape[0], '焦内气泡个数 %d' % bubble_num, '焦内气泡的总面积 %d' % area_total)
            print(path, '气泡个数密度 %f个/mm3' % density_num, '气泡体积密度 %f%%' % (density_vol*100))
            result_img = img_save
            result_info.append(os.path.basename(path))
            result_info.extend([str(det.shape[0])+'个', str(bubble_num)+'个', str(round(area_total, 4))+'平方毫米', str(round(density_num, 4))+'个/立方毫米'])
            result_info.append('%f%%' % (round(density_vol*100, 4)))
            # result_info = str((os.path.basename(path)+'\n' + '气泡总个数 %d\n焦内气泡个数 %d\n焦内气泡的总面积 %d\n气泡个数密度 %f个/mm3\n气泡体积密度 %f%%' % (det.shape[0], bubble_num, area_total, density_num, (density_vol*100))))
            # Save results
            cv2.imwrite(save_path, img_save)
            with open(txt_result_path + '.txt', 'a') as f:
                f.write(str(result_info))
    return result_img, result_info

def PredictLabel(crop, model):
    X_test = np.expand_dims(crop, axis=0)
    X_test = X_test.astype('float')
    X = model.predict_label(X_test)[0][1]
    return X