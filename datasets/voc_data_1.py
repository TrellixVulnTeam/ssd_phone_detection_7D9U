import sys
sys.path.append('/home/disk/yenanfei/DMS_phone/ssd_pytorch/')
from utils.general import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first
from datasets.vision import VisionDataset
from datasets.data_aug import PhotometricDistort
import os
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
import collections
import random
import cv2
import copy
import math
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
def parse_xml_annotation(parse_xml_annotation,label_map):
    size=parse_xml_annotation['size']
    width=float(size['width'])
    height=float(size['height'])
    xml_object=parse_xml_annotation['object']
    def parse_objects(xml_object):
        name=xml_object['name']
        bndbox=xml_object['bndbox']
        assert name in label_map,name
        name_id=int(label_map[name])
        center=(float(bndbox['xmin'])+float(bndbox['xmax']))*1.0/2.0,(float(bndbox['ymin'])+float(bndbox['ymax']))*1.0/2.0
        obj_width,obj_height=float(bndbox['xmax'])-float(bndbox['xmin']),float(bndbox['ymax'])-float(bndbox['ymin'])
        return name_id,center[0]*1.0/width,center[1]*1.0/height,obj_width*1.0/width,obj_height*1.0/height
    objects=list(map(parse_objects,xml_object))
    return objects

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
from numpy import random

def ssd_random_expand( image, boxes, labels,mask=None,mean=128):
    if random.randint(0,2):
        return image, boxes, labels,mask
    height, width, depth = image.shape
    ratio = random.uniform(1, 4)
    left = random.uniform(0, width*ratio - width)
    top = random.uniform(0, height*ratio - height)
    expand_image = np.zeros(
        (int(height*ratio), int(width*ratio), depth),
        dtype=image.dtype)
    expand_image[:, :, :] = mean
    expand_image[int(top):int(top + height),
                    int(left):int(left + width)] = image

    image = expand_image
    if boxes is not None:
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
    if mask is not None:
        if len(mask.shape)==2:
            expand_mask = np.ones(
                (int(height*ratio), int(width*ratio)),
                dtype=image.dtype)*255
        else:
            expand_mask = np.ones(
                (int(height*ratio), int(width*ratio),mask.shape[2]),
                dtype=image.dtype)*255
        expand_mask[int(top):int(top + height),
                        int(left):int(left + width)] = mask
        return image, boxes, labels,expand_mask
    else:
        return image, boxes, labels,None
import glog
def ssd_random_image_resolution(image,resolution_options=[(1280,720),(1920,1080),(1080,1080),(720,1280),(1080,1920),None]):
    resolution=random.choice(resolution_options)
    if resolution is not None:
        image=cv2.resize(image,(resolution[0],resolution[1]))
    height,width,channels=image.shape

    return image,(height,width)



def ssd_random_sample_crop( image,img_size, center_bbox_labels,sample_options=(None,(0.6, None),(0.7, None),(0.8, None),(0.9, None),(None, None),)):
    # return image,center_bbox_labels

    height, width,ch=image.shape
    labels,boxes=center_bbox_labels[:,0],xywh2xyxy(center_bbox_labels[:,1:])*(np.array([width,height,width,height]).reshape(1,4))
    image,boxes, labels,_=ssd_random_expand(image,boxes, labels,mean=128)
    height, width,ch=image.shape
    # glog.info("{} {}".format(width,height))
    while True:
        # randomly choose a mode
        mode = random.choice(sample_options)
        if mode is None:
            current_boxes=boxes*1.0/(np.array([width,height,width,height]).reshape(1,4))*img_size
            current_image=cv2.resize(image,(img_size,img_size))
            corner_bbox_labels=np.concatenate([labels[:,np.newaxis],current_boxes],axis=1)
            return current_image, corner_bbox_labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        # max trails (50)
        for _ in range(50):
            current_image = image

            w = random.uniform(0.3 * width, width)
            h = random.uniform(0.3 * height, height)

            # aspect ratio constraint b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            left = random.uniform(0,width - w)
            top = random.uniform(0,height - h)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])

            # calculate IoU (jaccard overlap) b/t the cropped and gt boxes

            overlap = jaccard_numpy(boxes, rect)

            # is min and max overlap constraint satisfied? if not try again
            if overlap.min() < min_iou and max_iou < overlap.max():
                continue

            # cut the crop from the image
            current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                          :]


            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            # select_clip1=np.clip(boxes[:,[0,2]],rect[0],rect[2])
            # select_clip2=np.clip(boxes[:,[1,3]],rect[1],rect[3])
            # mask=((select_clip1[:,1]-select_clip1[:,0])>10)*((select_clip2[:,1]-select_clip2[:,0])>10)

            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

            # mask in that both m1 and m2 are true
            mask = m1 * m2
            # print(mask)


            # have any valid boxes? try again if not
            if not mask.any():
                continue

            # take only matching gt boxes
            current_boxes = boxes[mask, :].copy()

            # take only matching gt labels
            current_labels = labels[mask]

            # should we use the box left and top corner or the crop's
            current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                              rect[:2])
            # adjust to crop (by substracting crop's left,top)
            current_boxes[:, :2] -= rect[:2]

            current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                              rect[2:])
            # adjust to crop (by substracting crop's left,top)
            current_boxes[:, 2:] -= rect[:2]
            # glog.info("{}".format((w,h)))
            # glog.info(current_boxes)

            current_boxes=current_boxes*1.0/(np.array([w,h,w,h]).reshape(1,4))*img_size
            current_center_bbox_labels=np.concatenate([current_labels[:,np.newaxis],current_boxes],axis=1)
            current_image=cv2.resize(current_image,(img_size,img_size))
            # glog.info("{}".format(current_center_bbox_labels))
            # exit(0)
            return current_image, current_center_bbox_labels



def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass
    return s


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])

import glog
def load_mosaic(self, index):
    # loads images in a mosaic




    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices

    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)
        # print(h,w)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)
    # import matplotlib.pyplot as plt
    # plt.imshow(img4)
    # plt.show()
    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
        # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove
    # glog.info(labels4)

    return img4, labels4

def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    orig_label = targets
    orig_img = img
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        orig_xy_left=xy[:, [0, 2]]
        orig_xy_right=xy[:, [1, 3]]
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]
    

    if (orig_label[0][0] == -1):
        neg_target =np.zeros((1,5))
        neg_target[0][0] = -1

        return img,neg_target
    else:
        if orig_xy_left.all()!=xy[:, [0, 2]].all() or orig_xy_right.all() != xy[:, [1, 3]].all():
            return orig_img, orig_label
        return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels



def reduce_img_size(path='path/images', img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)

import random
class VOCFormatDetectionDataset(VisionDataset):
    def __init__(
            self,
            root: str='/home/disk/public/Adas_Dataset/',
            file_list:str='datasets/trainval.txt',
            img_size:int=512,
            batch_size:int=16,
            augment:bool=False,
            hyp=None,
            rect:bool=False,
            image_weights:bool=False,
            cache_images:bool=False,
            single_cls:bool=False,
            stride:int=32,
            pad:int=0,
            rank:int=-1 ,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            voc_label_map={'phone':0,'cell phone':0}
            ):
        super(VOCFormatDetectionDataset, self).__init__(root, transforms, transform, target_transform)
        split_f=file_list
        self.photoDistort=PhotometricDistort()
        with open(os.path.join(split_f), "r") as f:
            image_xml_pair = [x.strip().split() for x in f.readlines()]#[:100]
            random.shuffle(image_xml_pair)
        self.img_files = [os.path.join(root, x[0]) for x in image_xml_pair]
        self.label_files = [os.path.join(root, x[1]) for x in image_xml_pair]
        assert (len(self.img_files) == len(self.label_files))
        n=len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)

        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic =self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # self.mosaic=None
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.voc_label_map=voc_label_map

        # Check cache
        fn,ext=os.path.splitext(file_list)
        cache_path = fn + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        labels, shapes = zip(*[cache[x] for x in self.img_files])
        # print(labels)
        # print(shapes)
        # exit(0)

        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = enumerate(self.label_files)

        if rank in [-1, 0]:
            pbar = tqdm(pbar)
        for i, file in pbar:
           
            l = self.labels[i]  # label
            
            if l is not None and l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found
                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
             
                self.labels[i] = np.array([[-1,0,0,0,0]])
                nf += 1
                # self.labels[i] = np.zeros_like(default)

                # print(self.labels[i],type(self.labels[i]),self.labels[i].shape)
                # exit(0)
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove
            
            if rank in [-1, 0]:
                pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                    cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)

            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)


    def __getitem__(self, index):
            if self.image_weights:
                index = self.indices[index]

            hyp = self.hyp
            if self.mosaic:
                # Load mosaic
                print(self.hyp['ssd_crop'])
                if self.hyp['ssd_crop']>np.random.rand() and len(self.labels[index])>0:
                    img, _, (h, w) = load_image(self, index)
                    img,(h,w)=ssd_random_image_resolution(img)
                    # glog.info("{} {} {}".format(h,w,self.labels[index]))
                
                    img, labels =ssd_random_sample_crop(img,self.img_size,self.labels[index])
            
                else:
                    img, labels = load_mosaic(self, index)

                shapes = None

                # MixUp https://arxiv.org/pdf/1710.09412.pdf
                if random.random() < hyp['mixup']:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                    r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                    img = (img * r + img2 * (1 - r)).astype(np.uint8)
                    labels = np.concatenate((labels, labels2), 0)

            else:
                print(2)
                # Load image
                img, (h0, w0), (h, w) = load_image(self, index)
                if not self.augment:
                    img=cv2.resize(img,(w,w))
                    h=w

                # Letterbox
                shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
                shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

                # Load labels
                labels = []
                x = self.labels[index]
                
                # if (x[:,1:]==np.array([0,0,0,0])).all():
                    
                #     labels = x.copy()
                #     labels[:, 1] = 0
                #     labels[:, 2] = 0
                #     labels[:, 3] = 0
                #     labels[:, 4] = 0
                #     print(labels)

                if x.size > 0:
                    
                    # Normalized xywh to pixel xyxy format
                    labels = x.copy()
                    labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                    labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                    labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                    labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
                    # print(labels)

                    # labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) #+ pad[0]  # pad width
                    # labels[:, 2] = h* (x[:, 2] - x[:, 4] / 2) #+ pad[1]  # pad height
                    # labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) #+ pad[0]
                    # labels[:, 4] = h* (x[:, 2] + x[:, 4] / 2) #+ pad[1]

            if self.augment:
               
                # Augment imagespace
                if not self.mosaic:
                    img, labels = random_perspective(img, labels,
                                                     degrees=hyp['degrees'],
                                                     translate=hyp['translate'],
                                                     scale=hyp['scale'],
                                                     shear=hyp['shear'],
                                                     perspective=hyp['perspective'])

                # Augment colorspace
                    
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
                if random.random()<hyp['photo']:
                    img, _, _,_=self.photoDistort(img,None,None,None)
                # Apply cutouts
                # if random.random() < 0.9:
                #     labels = cutout(img, labels)

            nL = len(labels)  # number of labels
            
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # # labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            # for i in range(nL):
            #     _,xx,yy,xx2,yy2=labels[i]
            #     rect = plt.Rectangle(((xx+xx2)/2,(yy+yy2)/2), xx2-xx, yy2-yy, fill=False, edgecolor = 'red',linewidth=1)
            #     ax.add_patch(rect)

            # plt.imshow(img)
            # plt.show()
            if nL:
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
                labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
                labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            if self.augment:
                # flip up-down
                if random.random() < hyp['flipud']:
                    img = np.flipud(img)
                    if nL:
                        if labels[0][0]!=-1:
                            labels[:, 2] = 1 - labels[:, 2]
                        

                # flip left-right
                if random.random() < hyp['fliplr']:
                    img = np.fliplr(img)
                    if nL:
                        if labels[0][0]!=-1:
                            labels[:, 1] = 1 - labels[:, 1]

            labels_out = torch.zeros((nL, 6))
            if len(labels)>0:
                labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])
                if nL:
                    labels_out[:, 1:] = torch.from_numpy(labels)


            # Convert
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
            img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = (np.ascontiguousarray(img)*1.0-128.0).astype('float32')


            bboxes=labels_out[:,2:].float()
            labels=labels_out[:,1].float()

            objects=torch.cat([bboxes,labels.unsqueeze(1)],dim=1)

            if objects.size(0)==0:
                objects=None
            return torch.from_numpy(img), objects, self.img_files[index], shapes

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    # Ancillary functions --------------------------------------------------------------------------------------------------
    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:  # not cached
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            h0, w0 = img.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized



    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    target = self.parse_voc_xml(
                        ET.parse(label).getroot())
                    l=np.array(parse_xml_annotation(target['annotation'],label_map=self.voc_label_map))



                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = [None, None]

                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        # for i, l in enumerate(label):
        #     l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), (label, path, shapes)
    # @staticmethod
    # def detection_collate(batch):
    #     """Custom collate fn for dealing with batches of images that have a different
    #     number of associated object annotations (bounding boxes).

    #     Arguments:
    #         batch: (tuple) A tuple of tensor images and lists of annotations

    #     Return:
    #         A tuple containing:
    #             1) (tensor) batch of images stacked on their 0 dim
    #             2) (list of tensors) annotations for a given image are stacked on
    #                                  0 dim
    #     """

    #     imgs = []
    #     objects,path,shapes=[],[],[],[],[],[]
    #     for sample in batch:
    #         imgs.append(sample[0])
    #         if sample[1] is not None:
    #             objects.append(torch.from_numpy(sample[1].astype('float32')))
    #         else:
    #             objects.append(None)



    #     return torch.stack(imgs, 0), (objects,path,shapes)


    def __len__(self):
        return len(self.img_files)


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', self._RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

    class _RepeatSampler(object):
        """ Sampler that repeats forever.

        Args:
            sampler (Sampler)
        """

        def __init__(self, sampler):
            self.sampler = sampler

        def __iter__(self):
            while True:
                yield from iter(self.sampler)


def create_dataloader(root='/home/disk/tanjing/projects/adas_multitask/vista/',file_list='datasets/trainval.txt', imgsz=512, batch_size=1, stride=32, single_cls=False, hyp=None, augment=False, cache=False, pad=0, rect=False,
                      rank=-1, world_size=1, workers=1):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(rank):
        dataset =VOCFormatDetectionDataset(
                    root=root,
                    file_list=file_list,
                    img_size=imgsz,
                    batch_size=batch_size,
                    augment=augment,
                    hyp=hyp,
                    rect=rect,
                    cache_images=cache,
                    single_cls=single_cls,
                    stride=stride,
                    pad=pad,
                    rank=rank
                    )

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=nw,
                                    sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=VOCFormatDetectionDataset.collate_fn)
    return dataloader, dataset


import yaml
if __name__ == '__main__':
    labelmap=["phone","cell phone"]
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    MEANS = (128, 128, 128)
    SIZE = 128
    with open('data/hyp.scratch.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    dataset=VOCFormatDetectionDataset(
            root='/home/disk/yenanfei/DMS_phone/PhoneDataset_recut',
            file_list='/home/disk/yenanfei/DMS_phone/PhoneDataset_recut/ImageSets/trainval.txt',
            img_size=128,
            batch_size=16,
            augment=True,
            hyp=hyp,
            rect=False,
            image_weights=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0,
            rank=-1
            )

    # dataloader,dataset=create_dataloader(root='/home/disk/public/Adas_Dataset/',file_list='datasets/trainval.txt', imgsz=512, batch_size=1, stride=32, single_cls=False, hyp=hyp, augment=True, cache=False, pad=0.0, rect=False,
    #                   rank=-1, world_size=1, workers=1)
    # import numpy as np

    index=0

    import copy
    cv2.namedWindow('frame',0)
    for data,objects,img_path,info in dataset:
        # bboxes=copy.copy(label[:,2:].float())
        # labels=copy.copy(label[:,1].float())
        # print(img_path)
        data=copy.copy(data)
        # objects=torch.cat([bboxes,labels.unsqueeze(1)],dim=1)
        # print(index)
        img = (data.permute(1, 2, 0).numpy()+128).astype('uint8').copy()
        # if label is None:
        #     continue
        # print(objects)
        if objects is not None:
            for i,obj in enumerate(objects):
                if obj is None:
                    continue

                x1,y1,x2,y2,cls = obj
                cls=int(cls)
                # print(x1,x2,y1,y2)
                if (x1.item() - x2.item()!=0):
                    cv2.rectangle(img,(int(x1*img.shape[1]), int(y1*img.shape[0])),(int(x2*img.shape[1]), int(y2*img.shape[0])), COLORS[cls % 9], 2)
                    cv2.putText(img, labelmap[cls], (int(x1*img.shape[1]), int(y1*img.shape[0])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls % 9], 2, cv2.LINE_AA)
        cv2.imshow('frame',img)
        if cv2.waitKey(0)&0xFF==ord('q'):
            break

        index=index+1
