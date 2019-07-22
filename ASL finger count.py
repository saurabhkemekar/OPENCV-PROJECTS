import cv2
import numpy as np
import math

def get_angle(start,end,far):
	x,y = start
	x1,y1 = end
	x2,y2 = far
	a = math.sqrt((x1-x)**2+(y1-y)**2)
	b = math.sqrt((x2-x)**2+(y2-y)**2)
	c = math.sqrt((x1-x2)**2+(y1-y2)**2)	
	k = b**2+c**2-a**2
	k1= 2*b*c

	# now according to inverse trignometry property
	# check whether value inside cos inverse is negative or not
	if k<0:
		k = abs(k)
		theta = math.pi -  math.acos(k/k1)	
	else:
		theta = math.acos(k/k1)
		
	theta= (theta*180)/math.pi
	
	return theta

#_______________________________________________________________________________________________

cap = cv2.VideoCapture(0)

_,frame = cap.read()
(row,col) = frame.shape[:2]
print(row,col)
background = np.zeros((row,col),np.uint8)


def nothing(x):
        pass
cv2.namedWindow('track')

cv2.createTrackbar('low','track',0,255,nothing)
cv2.createTrackbar('high','track',0,255,nothing)
count =0
while (cap.isOpened()):
        count =0
        _,frame = cap.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
                
        #low = cv2.getTrackbarPos('low','track')
       # high = cv2.getTrackbarPos('high','track')
        
        mask = cv2.inRange(hsv,np.array([0,48,48]),np.array([20,255,255]))
        background[0:row-100,300:col] = mask[0:row-100,300:col]
       # cv2.imshow('m',background)
        
        #ret,threshold = cv2.threshold(background,low,255,cv2.THRESH_BINARY_INV)  
        
        kernel  = np.zeros((5,5),np.uint8)
        threshold = cv2.erode(background,kernel,iterations =1)       
        image,contour,heic = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(threshold,contour,-1,(0,255,0),1)
        # cv2.imshow('imhae,',img)
        cnt1 = contour[0]
        hand_area = cv2.contourArea(cnt1)
        hull0 = cv2.convexHull(cnt1)
        hull = cv2.convexHull(cnt1,returnPoints=False)
        defects = cv2.convexityDefects(cnt1,hull)
        hull_area = cv2.contourArea(hull0)
        empty_area = abs(hull_area-hand_area)
        try:
            ratio =  hand_area/hull_area
            print(ratio)
        except Exception:
            pass
        #print(defects)
        try:
            for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    
                    l = defects[i]
                    h = l[0]
                    start = tuple(cnt1[s][0])
                    end = tuple(cnt1[e][0])
                    far = tuple(cnt1[f][0])
                  
                    theta = get_angle(start,end,far)
                    
                    if   theta>5 and theta<90 :
                           
                            count = count+1
                            cv2.circle(frame,far,4,(0,0,255),-1)
                            cv2.line(frame,start,end,(0,255,0),2)
            if count ==0 and ratio ==0.007275437991132755 :
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'0',(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
            elif count ==0 and ratio>= 0.83  :
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'1',(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
                    
            elif count !=2:
                font = cv2.FONT_HERSHEY_SIMPLEX
                count = count +1
                
                count = str(count)
                cv2.putText(frame,count,(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
            elif count ==2:
                    if  ratio > 0.741 and ratio < 0.78:                         
                         cv2.putText(frame,'6',(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
                    elif  ratio >= 0.67 and ratio <= 0.68:
                         cv2.putText(frame,'7',(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
                    elif   ratio>0.68 and ratio < 0.7:
                        cv2.putText(frame,'8',(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
                    elif  ratio > 0.65 and ratio <= 0.67:
                         cv2.putText(frame,'3',(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
                    elif  ratio > 0.7 and ratio < 0.72:
                        
                         cv2.putText(frame,'9',(0,50), font, 2,(255,0,255),2,cv2.LINE_AA)
                              
        except AttributeError:
                    pass           
                                         
        
        cv2.imshow('back',frame)
        
        
        k = cv2.waitKey(23) & 0xFF
        
        if k==23:
                break
                
                
cap.release()
cv2.destroyAllWindows()
