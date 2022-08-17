import yaml
import cv2
import numpy as np
import os
from inferencer import Inferencer

   
class PRnetLandmarkDetector:
    def __init__(self,model_config="prnet_config.yaml"):
        if not os.path.isfile(model_config):
            model_config_path=os.path.join(os.path.dirname(__file__),model_config)
        else:
            model_config_path=model_config
        with open(model_config_path,encoding="UTF-8") as f:
            config_dict = yaml.load(f)
        self.inferencer=Inferencer(config_dict)
        self.uv_kpt_ind=np.fromfile(config_dict['uv_kpt_ind'],sep=' ').reshape(2,-1).astype(int)
        self.canonical_vertices = np.load(config_dict['canonical_vertices']).T
        self.face_ind=np.fromfile(config_dict['face_ind'],sep=' ').astype(int)
    def get_point3d(self,image):
        h,w=image.shape[0],image.shape[1]
        pos_map=self.inferencer.infer(image)[0]
        headpose=self.get_headpose(pos_map)
        points_3d = (pos_map[:,self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:]].T)*np.array([w,h,w])
        return points_3d,headpose
        
    def get_headpose(self,pos_map):
        pred_face_vertices=pos_map.reshape(3,-1)[:,self.face_ind]*256

        def mean_center(face_vertices):
            mean_point=np.mean(face_vertices,axis=1).reshape(3,1)
            return face_vertices-mean_point
            
        center_pred_face_vertices=mean_center(pred_face_vertices) #3x 43867
        center_canonical_face_vertices=mean_center(self.canonical_vertices)#3x 43867

        covariance_matrix = center_pred_face_vertices.dot(center_canonical_face_vertices.T) #3x 43867 . 43867x3->3x3
        U,S,V = np.linalg.svd(covariance_matrix)
        R = U.dot(V)
        if np.linalg.det(R) < 0:
            R[:,2] *= -1
        from math import cos, sin, atan2, asin
        yaw = asin(R[2,0])
        pitch = atan2(R[2,1]/cos(yaw), R[2,2]/cos(yaw))
        roll = atan2(R[1,0]/cos(yaw), R[0,0]/cos(yaw))
        print("pitch:{} yaw:{} roll:{}".format(pitch*180/np.pi,yaw*180/np.pi,roll*180/np.pi))
        return pitch*180/np.pi,yaw*180/np.pi,roll*180/np.pi
        
class SSDFaceDetector:
    def __init__(self,model_config="ssd_config.yaml"):
        if not os.path.isfile(model_config):
            model_config_path=os.path.join(os.path.dirname(__file__),model_config)
        else:
            model_config_path=model_config
        with open(model_config_path,encoding="UTF-8") as f:
            config_dict = yaml.load(f)
        self.inferencer=Inferencer(config_dict)
        self.conf=config_dict['conf']
    def get_bboxes(self,image):
        h,w=image.shape[0],image.shape[1]
        detection_out=self.inferencer.infer(image)[0]
        bboxes=[]
        for i in range(detection_out.shape[1]):
            img_id,cls,prob,x1,y1,x2,y2=detection_out[0,i]
            if prob>self.conf:
                x1,y1,x2,y2,prob=int(x1*w),int(y1*h),int(x2*w),int(y2*h),prob
                bboxes.append([x1,y1,x2,y2,prob])
        return np.array(bboxes)
        
def get_images(source):
    if os.path.isdir(source):
        filenames=os.listdir(source)
        for filename in filenames:
            file_path=os.path.join(source,filename)
            img=cv2.imread(file_path)
            yield img
    elif os.path.isfile(source) and (os.path.splitext(source)[1].lower() in ['.mp4','.avi','.h264']):
        cap=cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret,frame=cap.read()
        while(ret):
            ret,frame=cap.read()
            print("fps：{}".format(fps))
            yield frame
    elif os.path.isfile(source) and (os.path.splitext(source)[1].lower() in ['.jpg','.bmp','.png','.jpeg']):
        file_path=source
        img=cv2.imread(file_path)
        yield img
    else:
        cap=cv2.VideoCapture(eval(source))
        ret,frame=cap.read()
        while(ret):
            ret,frame=cap.read()
            yield frame

def bbox_scaler(x1,y1,x2,y2,scale,width,height):

    bbox_h=y2-y1+1
    bbox_w=x2-x1+1
    center_x=(x1+x2)/2
    center_y=(y1+y2)/2
    x1=center_x-scale*bbox_w/2
    x2=center_x+scale*bbox_w/2
    y1=center_y-scale*bbox_h/2
    y2=center_y+scale*bbox_h/2
    x1=np.clip(x1,0,width)
    x2=np.clip(x2,0,width)
    y1=np.clip(y1,0,height)
    y2=np.clip(y2,0,height)
    return int(x1),int(y1),int(x2),int(y2)
    
if __name__=="__main__":
    import argparse,time
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmk-config', type=str, default="prnet_config.yaml", help='model.yaml path(s)')
    parser.add_argument('--det-config', type=str, default="ssd_config.yaml", help='model.yaml path(s)')
    parser.add_argument('--source', type=str, default=r'/home/disk/zhangmulan/XuchangeWarnVideo/XuChang/alarm_video/接打电话告警/2020-07-30/6502202007301736330017_32266_1_30.h264', help='source')  # file/folder, 0 
    parser.add_argument('--video-list', type=str, default=r'video_list.txt', help='video list')  # file/folder, 0 
    opt = parser.parse_args()
    lmk_det=PRnetLandmarkDetector(opt.lmk_config)
    face_det=SSDFaceDetector(opt.det_config)
    cv2.namedWindow('test',0)
    if opt.video_list.endswith('.txt'):
        with open(opt.video_list)as f:
            video_list=list(map(lambda x:x.strip(),f.readlines()))
        for video_path in video_list:
            opt.source=video_path
            print(opt.source)
 
            for image in get_images(opt.source):
                if image is None:
                    break
                
                height,width,channel=image.shape
                bboxes=face_det.get_bboxes(image)
                for x1,y1,x2,y2,prob in bboxes:
                    x1,y1,x2,y2=bbox_scaler(x1,y1,x2,y2,1.3,width,height)
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                    face_crop=image[y1:y2,x1:x2]
                    cv2.putText(image, "{:.2f}".format(prob), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
                    point3d,headpose=lmk_det.get_point3d(face_crop)
                    point3d=point3d+np.array([x1,y1,0])
                    
                    for i,point2d in enumerate(point3d[:,0:2]):
                        point2d=point2d.astype(int)
                        cv2.circle(image,(point2d[0],point2d[1]),2,(0,255,0))
                        
                cv2.imshow('test',image)
                if cv2.waitKey(0)&0xff==ord('q'):
                    exit(0)


