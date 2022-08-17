import torch
import os
import numpy as np
import glog
from layers.box_utils import jaccard
# from utils import  SegEvaluator
import pickle
import time
from tqdm import tqdm
set_type = 'test'
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir

def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    path = os.path.join('result', filename)
    return path

def write_voc_results_file(all_boxes, dataset,labelmap):
    for cls_ind, cls in enumerate(labelmap):
        glog.info('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.label_files):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def voc_eval(objects_all,detpath,
             classname,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt

    recs=objects_all
    imagenames=objects_all.keys()
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):

            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def do_python_eval(objects_all,output_dir='output', use_07=True,labelmap=None,ovthresh=0.5):

    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    glog.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(objects_all,
           filename, cls,
           ovthresh=ovthresh, use_07_metric=use_07_metric)
        aps += [ap]
        glog.info('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    mAp=np.mean(aps)
    glog.info('Mean AP = {:.4f}'.format(mAp))
    glog.info('~~~~~~~~')
    glog.info('Results:')
    for ap in aps:
        glog.info('{:.3f}'.format(ap))
    glog.info('{:.3f}'.format(np.mean(aps)))
    glog.info('~~~~~~~~')
    glog.info('')
    glog.info('--------------------------------------------------------------')
    glog.info('Results computed with the **unofficial** Python eval code.')
    glog.info('Results should be very close to the official MATLAB eval code.')
    glog.info('--------------------------------------------------------------')
    return mAp


def evaluate_detections(box_list, output_dir, dataset,labelmap,objects_all,ovthresh):
    write_voc_results_file(box_list, dataset,labelmap)
    return do_python_eval(objects_all,output_dir,labelmap=labelmap,ovthresh=ovthresh)









def pytorch_detection(model):
    model.cuda()
    # file_path='/home/disk/yenanfei/DMS_phone/aliangsi/no_phone/'
    file_path='/home/disk/yenanfei/DMS_phone/video_test/former_test_result/orig/'

    count = 0
    all_file_list = os.listdir(file_path)
    for File in tqdm(all_file_list):
        cult_fram = cv2.imread(file_path+File)
        image = cv2.cvtColor(cult_fram,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(128,128))[:,:,np.newaxis]
        image = torch.from_numpy(image*1.0-128.0).permute(2, 0, 1).float()


        net_input=image.unsqueeze(0).cuda()
        image=(image.permute(1, 2, 0).numpy()+128).astype('uint8').copy()
        pred_det,pred_mask,_=model(net_input)

        detections = pred_det.detach().cpu().numpy()
        for i in range(detections.shape[1]):
            j = 0
            while detections[0, i, j, 0] >= 0.9:
                count+=1
                j += 1
    glog.info('--------------------------------------------------------------')
    glog.info("False Positive count {} out of {}".format(count,len(all_file_list)))
    glog.info('False Positive rate=======================>: {:.3f}'.format(count/len(all_file_list)))
    glog.info('--------------------------------------------------------------')









import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
def test_net(save_folder, net, cuda, dataset, labelmap,iou_thresh=0.5):
    
    # dump predictions and assoc. ground truth to text file for now
    net=net.module
    # pytorch_detection(net)
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    output_dir = get_output_dir('result', 'test')
    det_file = os.path.join(output_dir, 'detections.pkl')
    anno_file=os.path.join(output_dir, 'annotations.pkl')
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    has_label_file=os.path.isfile(anno_file)
    if has_label_file:
        with open(anno_file, 'rb') as f:
            objects_all = pickle.load(f)
    else:
        objects_all={}

    seg_evaluator=None
    pbar = tqdm(range(num_images), desc='Testing')
    for i in pbar:

        im, gt, img_path, img_info= dataset.__getitem__(i)
        h,w=img_info[0]

        if cuda:
            im=im.cuda()
        detections,seg_mask,_ = net(im.unsqueeze(0))


        # if seg_evaluator is None and seg_mask is not None and pred_mask is not None:
        #     seg_evaluator = SegEvaluator(pred_mask.size(1))
        #     seg_evaluator.reset()

        if seg_evaluator is not None and seg_mask is not None and pred_mask is not None:
            pred_mask=np.argmax(pred_mask.cpu().numpy(),axis=1)
            seg_evaluator.add_batch(seg_mask[np.newaxis,:,:], pred_mask)

        detect_time = _t['im_detect'].toc(average=False)

        if not has_label_file:
            objects=[]

            if gt is not None:
                for obj in gt:
                    obj_struct = {}
                    obj_struct['name'] = labelmap[int(obj[4])]
                    obj_struct['pose'] = 0
                    obj_struct['truncated'] = 0
                    obj_struct['difficult'] = False
                    bbox=[int(obj[0]*w)-1,int(obj[1]*h)-1,int(obj[2]*w)-1,int(obj[3]*h)-1]
                    obj_struct['bbox'] =bbox
                    objects.append(obj_struct)

            objects_all[dataset.label_files[i]]=objects



        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets


        pbar.desc='im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,num_images, detect_time)
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    if not has_label_file:
        with open(anno_file, 'wb') as f:
            pickle.dump(objects_all, f, pickle.HIGHEST_PROTOCOL)
    # if seg_evaluator is not None:
    #     Acc = seg_evaluator.Pixel_Accuracy()
    #     Acc_class = seg_evaluator.Pixel_Accuracy_Class()
    #     mIoU = seg_evaluator.Mean_Intersection_over_Union()
    #     FWIoU = seg_evaluator.Frequency_Weighted_Intersection_over_Union()

    #     glog.info('Evaluating segmentation')
    #     glog.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    # else:
    mIoU=0.0

    glog.info('Evaluating detections')
    mAp=evaluate_detections(all_boxes, output_dir, dataset,labelmap,objects_all,iou_thresh)
    
    return mIoU,mAp

