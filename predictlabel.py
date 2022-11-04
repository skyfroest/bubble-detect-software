# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:37:26 2022
output: 0 气泡在焦外； 1 气泡在角内
@author: 10579
"""

from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
from utils import classificationModel3
from utils import hyperparameterModel
import time
from PIL import Image
import cv2


def PredictLabel(crop):
    t1 = time.time()
    params=hyperparameterModel.hyperparameterModel()
    #X_test = crop.resize((64,64),Image.ANTIALIAS)
    #X_test = np.array(X_test)
    X_test = cv2.resize(crop, (64, 64))
    X_test = np.expand_dims(X_test, axis=0)
    #print(X_test.shape)
    
    
    X_test=X_test.astype('float')
    tflearn.init_graph()
    g = classificationModel3.createModel(params)
    model = tflearn.DNN(g,tensorboard_verbose=0)
    #模型路径，前面的本地部分改成你的就行
    model.load('./models/model.tfl')
    t2 = time.time()
    # print(model.predict_label(X_test))
    for i in range (100):
        X=model.predict_label(X_test)[0][1]
    t3 = time.time()
    print('Speed: %.1fms 加载模型, %.1fms 处理数据' %((t2-t1)*1000,(t3-t2)*1000))
    return X

# crop = Image.open(r'./image/hr_img.jpg')
crop = cv2.imread('./image/hr_img.jpg')
res = PredictLabel(crop)
print(res)

