import os
import torch
from torch import nn
import cv2
import json
from mmdet.apis import init_detector, inference_detector

config_file = '/home/tsinghuaee206/reppoint/config/reppoints_moment_r50_fpn_1x_coco.py'
checkpoint_file = '/home/tsinghuaee206/reppoint/work_dirs/reppoints_moment_r50_fpn_1x_coco/latest.pth'
cuda_device = 'cuda:0'
# test_path = '/home/tsinghuaee206/task4/Detection/test/'
test_path = '/home/tsinghuaee206/task4/exp/'
result_path = '/home/tsinghuaee206/reppoint/pred.json'
out_root = './output/'
max_num = 6

# Classifier from task2
basicPath = os.sys.path[0]
classList = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
             'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']
model = init_detector(config_file, checkpoint_file, device=cuda_device)


class Model(nn.Module):
    def __init__(self, bn=False):
        super(Model, self).__init__()
        if not bn:
            self. conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # 32*32
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 16*16
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8*8
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 4*4
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # 32*32
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 16*16
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8*8
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 4*4
                nn.ReLU(inplace=True)
            )

        self.fc_layers = nn.Sequential(
            nn.Linear(128*4*4, 128),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(128, 19),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # x = x.argmax(1)
        return x


def classify(image):
    net = Model()
    net.eval()
    pre = torch.load('model.pth', map_location=torch.device('cpu'))
    net.load_state_dict(pre)
    data = image.unsqueeze(1)
    if torch.cuda.is_available():
        net = net.cuda()
        img = data.cuda()
    else:
        img = data
    torch.no_grad()
    class_pred = net(img).argmax(1)
    pred_list = []
    for i in range(0, img.size(0)):
       pred_list.append(classList[class_pred[i]])
    return pred_list


# Detector of task4
def iou(rect1, rect2):
    s1 = (rect1['x2'] - rect1['x1']) * (rect1['y2'] - rect1['y1'])
    s2 = (rect2['x2'] - rect2['x1']) * (rect2['y2'] - rect2['y1'])
    xx1 = max(rect1['x1'], rect2['x1'])
    yy1 = max(rect1['y1'], rect2['y1'])
    xx2 = min(rect1['x2'], rect2['x2'])
    yy2 = min(rect1['y2'], rect2['y2'])
    ww = max(0, xx2 - xx1)
    hh = max(0, yy2 - yy1)
    inter = ww * hh
    iou_ = inter / (s1 + s2 - inter)
    return iou_


def nms(rects, confidence_threshold=0.2, nms_threshold=0.3):
    tmp = rects
    tmp.sort(key=lambda rects_tmp: rects_tmp['confidence'], reverse=True)
    jj = 0
    for ii in range(len(tmp)-1, -1, -1):
        if tmp[ii]['confidence'] >= confidence_threshold:
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
                    tmp_iou = iou(tmp[ii], tmp[jj])
                    if tmp_iou > nms_threshold:
                        flag_array[jj] = False
    for ii in range(rect_num):
        if flag_array[ii]:
            nms_result.append(tmp[ii])
    if len(nms_result) > max_num:
        nms_result = nms_result[0: max_num]
    return nms_result


def traffic_sign_detect(img_path, confidence_threshold=0.2, nms_threshold=0.3):
    img_result = inference_detector(model, img_path)
    rects = []
    for rect in img_result:
        if len(rect):
            tmp = dict(
                x1=rect[0][0],
                y1=rect[0][1],
                x2=rect[0][2],
                y2=rect[0][3],
                confidence=rect[0][4]
            )
            rects.append(tmp)
    if len(rects) == 0:
        print('Image: ' + img_path + '; Number of Traffic Signs Detected: 0')
        return [], []
    else:
        rect_final = nms(rects, confidence_threshold, nms_threshold)
        print('Image: ' + img_path + '; Number of Traffic Signs Detected: ' + str(len(rect_final)))
        img_content = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        rect_list = torch.zeros(len(rect_final), 64, 64)
        for ii in range(len(rect_final)):
            rect = img_content[
                   int(rect_final[ii]['y1']): int(rect_final[ii]['y2']) + 1,
                   int(rect_final[ii]['x1']): int(rect_final[ii]['x2']) + 1
                   ]
            rect = cv2.resize(rect, (64, 64))
            rect = torch.tensor(rect)
            rect_list[ii, :, :] = rect
        result_list = classify(rect_list)
        return rect_final, result_list


def draw_anchor(image_path, image_name, result):
    img = cv2.imread(image_path)
    out_file = os.path.join(out_root, image_name)
    img_out = img
    for i in range(len(result)):
        x1 = int(result[i]['x1'])
        y1 = int(result[i]['y1'])
        x2 = int(result[i]['x2'])
        y2 = int(result[i]['y2'])
        img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(out_file, img_out)


def test_set_detect(test_set_path, draw_flag=False, confidence_threshold=0.2, nms_threshold=0.3):
    json_result = {'imgs': dict()}
    images = os.listdir(test_set_path)
    for img in images:
        img_id = img.replace('.jpg', '')
        img_path = os.path.join(test_set_path, img)
        rect_final, result_list = traffic_sign_detect(img_path, confidence_threshold, nms_threshold)
        len_rect_final = len(rect_final)
        if len_rect_final:
            object_list = []
            for ii in range(len_rect_final):
                object_list.append({
                    'bbox': {
                        'xmax': float(rect_final[ii]['x2']),
                        'xmin': float(rect_final[ii]['x1']),
                        'ymax': float(rect_final[ii]['y2']),
                        'ymin': float(rect_final[ii]['y1']),
                    },
                    'category': result_list[ii],
                    'score': float(rect_final[ii]['confidence'])
                })
            json_result['imgs'][str(img_id)] = dict(objects=object_list)
            if draw_flag:
                draw_anchor(img_path, img, rect_final)
        else:
            json_result['imgs'][str(img_id)] = dict(objects=[])
    return json_result


if __name__ == '__main__':
    json_r = test_set_detect(test_path, draw_flag=True, confidence_threshold=0.4)
    outfile = open(result_path, 'w')
    json.dump(json_r, outfile)
    outfile.close()
