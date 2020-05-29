import os
import json


def temp2coco():
    basicPath = os.sys.path[0]
    trainPath = os.path.join(basicPath, "Detection/train")
    valPath = os.path.join(basicPath, "Detection/val")
    annoPath = os.path.join(basicPath, "Detection/train_annotations.json")

    classList = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
                 'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']
    class2id = {}  # 建立类和类编号的字典
    for i in range(len(classList)):
        class2id[classList[i]] = i

    train_dirs = os.listdir(trainPath)
    val_dirs = os.listdir(valPath)

    with open(annoPath, 'r') as f:
        anno = json.load(f)

    train_outfile = os.path.join(basicPath, "Detection/train.json")
    val_outfile = os.path.join(basicPath, "Detection/val.json")

    train_coco = {}
    train_coco['info'] = "spytensor created"
    train_coco['license'] = ["license"]
    train_coco['images'] = []
    train_coco['annotations'] = []
    train_coco['categories'] = []

    for i in range(len(classList)):
        cat = {}
        cat['id'] = i
        cat['name'] = classList[i]
        train_coco['categories'].append(cat)

    val_coco = {}
    val_coco['info'] = "spytensor created"
    val_coco['license'] = ["license"]
    val_coco['images'] = []
    val_coco['annotations'] = []
    val_coco['categories'] = train_coco['categories']


    sum_train = 0
    sum_val = 0
    for key in anno['imgs']:
        img = {}
        img["height"] = 2048
        img["width"] = 2048
        img["id"] = anno['imgs'][key]['id']
        img["file_name"] = anno['imgs'][key]['path'][6:]

        temp = str(key) + '.jpg'
        if temp in train_dirs:
            train_coco['images'].append(img)
            for j in range(len(anno['imgs'][key]['objects'])):
                target = anno['imgs'][key]['objects'][j]
                obj = {}
                obj['id'] = sum_train
                obj['image_id'] = int(key)
                obj['category_id'] = class2id[target['category']]
                obj['segmentation'] = []

                if 'ellipse_org' in target:
                    seg_temp = []
                    for v in target['ellipse_org']:
                        seg_temp.append(v[0])
                        seg_temp.append(v[1])
                    obj['segmentation'].append(seg_temp)

                obj['bbox'] = [target['bbox']['xmin']]
                obj['bbox'].append(target['bbox']['ymin'])
                w = target['bbox']['xmax'] - target['bbox']['xmin']
                h = target['bbox']['ymax'] - target['bbox']['ymin']
                obj['bbox'].append(w)
                obj['bbox'].append(h)

                obj['iscrowd'] = 0
                obj['area'] = float((w + 1)*(h + 1))

                train_coco['annotations'].append(obj)
                sum_train = sum_train + 1
        else:
            val_coco['images'].append(img)
            for j in range(len(anno['imgs'][key]['objects'])):
                target = anno['imgs'][key]['objects'][j]
                obj = {}
                obj['id'] = sum_val
                obj['image_id'] = int(key)
                obj['category_id'] = class2id[target['category']]
                obj['segmentation'] = []

                if 'ellipse_org' in target:
                    seg_temp = []
                    for v in target['ellipse_org']:
                        seg_temp.append(v[0])
                        seg_temp.append(v[1])
                    obj['segmentation'].append(seg_temp)

                obj['bbox'] = [target['bbox']['xmin']]
                obj['bbox'].append(target['bbox']['ymin'])
                w = target['bbox']['xmax'] - target['bbox']['xmin']
                h = target['bbox']['ymax'] - target['bbox']['ymin']
                obj['bbox'].append(w)
                obj['bbox'].append(h)

                obj['iscrowd'] = 0
                obj['area'] = float((w + 1) * (h + 1))

                val_coco['annotations'].append(obj)
                sum_val = sum_val + 1

    with open(train_outfile, 'w') as f:
        json.dump(train_coco, f)
    with open(val_outfile, 'w') as f:
        json.dump(val_coco, f)


if __name__ =='__main__':
    temp2coco()


