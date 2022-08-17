import cv2
import numpy as np
VIDEO_PATHS=['result_ori.avi','result.avi']
SIZE=256
def merge_video():
    for i,OFFSET in enumerate(VIDEO_PATHS):
        exec("cap{} = cv2.VideoCapture('{}')".format(i,OFFSET))

    cv2.namedWindow('frame',0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20
    size = (2*SIZE, SIZE)
    out = cv2.VideoWriter('merge_video.avi', fourcc, fps, size)
    j=0
    while(True):
        try:
            for i,OFFSET in enumerate(VIDEO_PATHS):
                exec("ret , image{} = cap{}.read()".format(i,i))
            canvas=np.zeros((SIZE,SIZE*2,3),dtype='uint8')
            for i,OFFSET in enumerate(VIDEO_PATHS):
                exec("canvas[:,{}:{},:]=image{}[:,:,:]".format(i*SIZE,(i+1)*SIZE,i))
            cv2.imshow('frame',canvas)

            j=j+1
            out.write(canvas)  # 写入视频对象
            if cv2.waitKey(1)&0xff==ord('q'):
                out.release()
                break
        except Exception as e:
            print(e)
            out.release()
            break
merge_video()