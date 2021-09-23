from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory,OTBDataset
from toolkit.utils.region import *
from toolkit.evaluation import OPEBenchmark
'''
COLOR = ((124, 252, 0),
         (139, 139, 0),
         (32, 178, 170),
         (139, 117, 0),
         (255, 106, 106),
         (238, 121, 66),
         (139, 90, 43),
         (255, 105, 180),
         (176, 48, 96),
         (148, 0, 211),
         (139, 62, 47),
         (139, 131, 134))
'''
COLOR=('#551A8B','#4F4F4F','#8B668B','#0000EE','#CD3278','#8B1C62','#8B8B83','#8B0000','#CD0000','#FF7F00','#8B5A00','#8B2323','#FF8C00','#D2691E','#CD950C','#006400')

# 生成锚点信息
def test(pint=False):
    config = 'config.yaml'
    dataset_root = './testing_dataset/OTB'
    cfg.merge_from_file(config)
    model=ModelBuilder()
    # 下载模型
    model=load_pretrain(model,'model.pth').cuda().eval()
    # 获取当前配置文件信息--》由此选定model信息
    tracker=build_tracker(model)
    # 创建数据集
    dataset=DatasetFactory.create_dataset(name='OTB100',dataset_root=dataset_root,load_img=False)
    total_lost=0
    # 目标捕获
    for v_idx,video in enumerate(dataset):
        toc=0
        pred_bboxes=[]
        scores=[]
        track_times=[]
        for idx,(img,gt_bbox) in enumerate(video):
            tic=cv2.getTickCount()
            if idx==0:
                # 第一帧不作预测
                cx,cy,w,h=get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_=[cx-(w-1)/2,cy-(h-1)/2,w,h]
                tracker.init(img[idx],gt_bbox_)
                scores.append(None)
                pred_bboxes.append(gt_bbox_)
            else:
                outputs=tracker.track(img[idx])
                pred_bbox=outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc+=cv2.getTickCount()-tic
            track_times.append((cv2.getTickCount()-tic)/cv2.getTickFrequency())
            if idx==0:
                cv2.destroyAllWindows()
            # 是否画图象
            if pint and idx>0:
                gt_bbox=list(map(int,gt_bbox))
                pred_bbox=list(map(int,pred_bbox))
                # 绿色--》真实
                cv2.rectangle(img[idx],(gt_bbox[0],gt_bbox[1]),(gt_bbox[0]+gt_bbox[2],gt_bbox[1]+gt_bbox[3]),(84, 255, 159),3)
                # 蓝色--》预测
                cv2.rectangle(img[idx],(pred_bbox[0],pred_bbox[1]),(pred_bbox[0]+pred_bbox[2],pred_bbox[1]+pred_bbox[3]),(25, 25, 112),3)
                # 红色--》字体
                cv2.putText(img[idx],str(idx),(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(238, 99, 99),2)
                cv2.imshow("Target Tracking",img[idx])
                cv2.resizeWindow("Target Tracking",960,960)
                cv2.waitKey(100)
        toc/=cv2.getTickFrequency()

        # 结果存储
        model_path='results/{}/'.format('OTB')
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path="{}/{}.txt".format(model_path,video.name)
        with open(result_path,'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('{:3d} Video_name:{:12s} Time:{:5.1f}s Speed:{:3.1f}fbs '.format(v_idx+1,video.name,toc,idx/toc))
    cv2.destroyAllWindows()

# 评测信息
def eval():
    tracker_path='./results/OTB'
    trackers=os.listdir(tracker_path)

    assert len(trackers)>0,"请检查您的训练样本信息"
    # 线程并发
    num=min(1,len(trackers))
    data_root='./testing_dataset/OTB'
    dataset=OTBDataset(name='OTB100',dataset_root=data_root)
    dataset.set_tracker(tracker_path,trackers)
    benchmark=OPEBenchmark(dataset)

    success_ret={}
    with Pool(processes=num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,trackers),desc='eval_success',total=len(trackers),ncols=100):
            success_ret.update(ret)
    # print(success_ret)

    precision_ret={}
    with Pool(processes=num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,trackers),desc='eval_precision',total=len(trackers),ncols=100):
            precision_ret.update(ret)
    # print(precision_ret)

    benchmark.show_result(success_ret,precision_ret)
    print(precision_ret)
    draw_success(success_ret)
    draw_precision(precision_ret)


def draw_success(success_ret):
    thresholds=np.arange(0,1.05,0.05)
    for i in range(len(list(thresholds))):
        thresholds[i]=round(thresholds[i],2)
    success= {}
    for tracker_name in success_ret.keys():
        rate=success_ret[tracker_name]
        for threshold in thresholds:
            success_item=0
            for i in range(len(rate)):
                if rate[i] > threshold:
                    success_item = success_item + 1
            if tracker_name not in success.keys():
                success.setdefault(tracker_name, []).append(float(success_item / len(rate)))
            else:
                success[tracker_name].append(float(success_item / len(rate)))
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    for idx,tracker_name in enumerate(success):
        plt.plot(thresholds, success[tracker_name], label=tracker_name,color=COLOR[idx])
    plt.legend()
    plt.show()


def draw_precision(precision_ret):
    thresholds = np.arange(0, 51, 1)
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    for idx,tracker_name in enumerate(precision_ret):
        plt.plot(thresholds,precision_ret[tracker_name],label=tracker_name,color=COLOR[idx])
    plt.legend()
    plt.show()

if __name__=='__main__':
    # test(True)
    # 数据初始化
    eval()
