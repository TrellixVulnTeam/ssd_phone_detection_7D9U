# caffe_model=caffe.Net('./phone_128_lowrank_purning_90%.prototxt','./phone_128_lowrank_purning_90%.caffemodel',caffe.TEST)


#   Usage: phone_detection(caffe_model,lm67,orig_fram,conf_thresh)
#   return [[bbox,prob]]


def crop_calling_area(lmks,h,w):
    eye_index = [41,37,28,32]
    center = np.mean(lmks[eye_index,:],axis = 0) 
    #print 'center',center
    face_bbox=bbox_from_points(lmks)
    distance=lambda x,y:np.sqrt(np.sum(np.power(x-y,2)))
    up_distance=distance(center,lmks[46,:])
    jaw_distance=distance(center,lmks[7,:])
    left_distance = max(distance(center,lmks[0]),distance(center,lmks[14]))
    x0 = np.clip(center[0]-2.2*left_distance,0,w)#1.5
    y0 = np.clip(center[1]-2*up_distance,0,h)#1
    x1 = np.clip(center[0]+2.2*left_distance,0,w)#1.5
    y1 = max(center[1]+5*up_distance,center[1]+2*jaw_distance)
    y1 = np.clip(y1,0,h)#2
    return [x0,y0,x1,y1]




def phone_detection(caffe_model,lm67,orig_fram,conf_thresh):
    height,width,ch=orig_fram.shape
    area = crop_calling_area(lm67,height,width)
    cult_fram = orig_fram[int(area[1]):int(area[3]),int(area[0]):int(area[2])]
    if cult_fram is not None:

        img_gray=cv2.cvtColor(cult_fram,cv2.COLOR_BGR2GRAY)
        img_gray_resized=(cv2.resize(img_gray,(128,128))[:,:,np.newaxis]*1.0-128)

        net_input=np.transpose(img_gray_resized,(2,0,1))
        caffe_model.blobs['data'].data[...] = net_input

        # Forward pass.
        detections = caffe_model.forward(['detection_out','mbox_priorbox'])
        phone_results=detections['detection_out']

        scale=[0.1,0.1]
        h,w,_=cult_fram.shape

        phone_boxes=[]
        for i in range(phone_results.shape[2]):
            prob=phone_results[0,0,i,2]
            
            if prob<conf_thresh:
                continue
            bbox=np.array([phone_results[0,0,i,3]*w,phone_results[0,0,i,4]*h,phone_results[0,0,i,5]*w,phone_results[0,0,i,6]*h])
            phone_boxes.append([bbox,prob])

        return phone_boxes


