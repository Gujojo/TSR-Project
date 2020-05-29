import os
import numpy as np
import torch
import json
import cv2
from mmdet.apis import init_detector, inference_detector

eps = 1e-6
threshold = 0.5
confidence_t = 0.2

config_file = './config/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = './work_dirs/faster_rcnn_r50_fpn_1x_coco/latest.pth'
anno_file = '/home/tsinghuaee206/task4/Detection/val.json'
img_root = '/home/tsinghuaee206/task4/Detection/val/'
out_root = './output/'
cuda_device = 'cuda:0'
acc_root = './acc.txt'

model = init_detector(config_file, checkpoint_file, device=cuda_device)


def compute_iou(box1_in, box2):
    '''
    Args:
      box1_in: (tensor) bounding boxes, sized [N,4].
      box2_in: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''
    N = box1_in.size(0)
    M = box2.size(0)

    box1 = torch.cat((box1_in[..., 0].unsqueeze(-1),
            box1_in[..., 1].unsqueeze(-1),
            (box1_in[..., 0] + box1_in[..., 2]).unsqueeze(-1),
            (box1_in[..., 1] + box1_in[..., 3]).unsqueeze(-1)),
            dim=-1)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    union = torch.max((area1 + area2 - inter), eps * torch.ones(area1.size()))
    iou_ = inter / union
    return iou_


def temp_iou(rect1, rect2):
    s1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    s2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    xx1 = max(rect1[0], rect2[0])
    yy1 = max(rect1[1], rect2[1])
    xx2 = min(rect1[2], rect2[2])
    yy2 = min(rect1[3], rect2[3])
    ww = max(0, xx2 - xx1)
    hh = max(0, yy2 - yy1)
    inter = ww * hh
    iou_ = inter / (s1 + s2 - inter)
    return iou_


def nms(rects, confidence_threshold=0.5, nms_threshold=0.25):
    tmp = list(rects)
    tmp.sort(key=lambda rects_tmp: rects_tmp[4], reverse=True)
    jj = 0
    for ii in range(len(tmp)-1, -1, -1):
        if tmp[ii][4] >= confidence_threshold:
            jj = ii
            break
    tmp = tmp[0: jj+1]
    rect_num = len(tmp)
    flag_array = [True] * rect_num
    nms_result = []
    for ii in range(rect_num-1):
        if flag_array[ii]:
            for jj in range(ii+1, rect_num):
                if flag_array[jj]:
                    tmp_iou = temp_iou(tmp[ii], tmp[jj])
                    if tmp_iou >= nms_threshold:
                        flag_array[jj] = False
    for ii in range(rect_num):
        if flag_array[ii]:
            nms_result.append(tmp[ii][:-1])
    return torch.tensor(nms_result)


def draw_anchor(image, result):

    image_path = os.path.join(img_root, image)
    img = cv2.imread(image_path)
    img_out = img
    out_file = os.path.join(out_root, 'output_' + image)
    for i in range(len(result)):
        x1 = int(result[i][0])
        y1 = int(result[i][1])
        x2 = int(result[i][2])
        y2 = int(result[i][3])
        img_out = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(out_file, img_out)


images = os.listdir(img_root)
with open(anno_file, 'r') as f:
    anno = json.load(f)
acc = []
total_correct = 0
total_p = 0
total_r = 0
for img in images:
    image_path = os.path.join(img_root, img)
    result = inference_detector(model, image_path)
    rectangles = []
    for i in result:
        if len(i):
            tmp = torch.tensor(i)
            index = np.argmax(tmp[:, -1])
            rectangles.append(tmp[index, :].tolist())
            # rectangles.append(tmp[index, :-1].tolist())
    rectangles = nms(rectangles, confidence_threshold=confidence_t)
    print("rectangles.shape", rectangles.shape)
    if rectangles.shape[0] == 0:
        for target in anno['annotations']:
            temp = str(target['image_id']) + '.jpg'
            if temp == img:
                total_r += 1
        continue
    bbox_gt = []
    cat_gt = []
    for target in anno['annotations']:
        temp = str(target['image_id']) + '.jpg'
        if temp == img:
            bbox_gt.append(target['bbox'])      # [M, 4]
            cat_gt.append(target['category_id'])    # [M, 1]
    iou = compute_iou(torch.tensor(bbox_gt), rectangles)
    iou = list(iou)
    correct = 0
    for i in range(0, len(iou)):
        idx = np.argmax(iou[i])
        if iou[i][idx] >= threshold:
            correct += 1
    recall = correct/len(iou)
    precise = correct/len(iou[0])
    total_correct += correct
    total_r += len(iou)
    total_p += len(iou[0])
    print('img_id: %d, recall: %.4f, precise: %.4f' % (int(img[0:-4]), recall, precise))
    acc.append([img, recall, precise])
    torch.save(acc, acc_root)
    draw_anchor(img, rectangles)
total_recall = total_correct / total_r
total_precision = total_correct / total_p
print('Model: ' + checkpoint_file)
print('Confidence Threshold: ' + str(confidence_t))
print('Total Recall: %.4f' % total_recall)
print('Total Precision: %.4f' % total_precision)
torch.save(total_recall, acc_root)
torch.save(total_precision, acc_root)
