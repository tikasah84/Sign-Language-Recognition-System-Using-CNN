import cv2
import numpy as np
import os
import string
# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")

#Create the directory for word
if not os.path.exists("data/train/Hello"):
    os.makedirs("data/train/Hello")

    
if not os.path.exists("data/train/Namaste"):
    os.makedirs("data/train/Namaste")

if not os.path.exists("data/train/Drink"):
    os.makedirs("data/train/Drink")

if not os.path.exists("data/train/I"):
    os.makedirs("data/train/I")

if not os.path.exists("data/train/Love"):
    os.makedirs("data/train/Love")

if not os.path.exists("data/train/You"):
    os.makedirs("data/train/You")

if not os.path.exists("data/train/Food"):
    os.makedirs("data/train/Food")




    


# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    _, frame = cap.read()
    # Simulating mirror image
  
    
    # Getting count of existing images
    count = {
             'Hello': len(os.listdir(directory+"/Hello")),
             'Namaste': len(os.listdir(directory+"/Namaste")),
             'Drink': len(os.listdir(directory+"/Drink")),
             'I': len(os.listdir(directory+"/I")),
             'Love': len(os.listdir(directory+"/Love")),
             'You': len(os.listdir(directory+"/You")),
             'Food':len(os.listdir(directory+"/Food")),
   
             }
    

    cv2.putText(frame, "Hello : "+str(count['Hello']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
    cv2.putText(frame, "Namaste : "+str(count['Namaste']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "Drink : "+str(count['Drink']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
    cv2.putText(frame, "I : "+str(count['I']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "Love : "+str(count['Love']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
    cv2.putText(frame, "You : "+str(count['You']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    

    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[10:410, 220:622]
#    roi = cv2.resize(roi, (64, 64))
    
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)

    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)



    
    test_image = cv2.resize(test_image,(300,300))
    cv2.imshow("test", test_image)
        
  
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'Hello/'+str(count['Hello'])+'.jpg', roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'Namaste/'+str(count['Namaste'])+'.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'Drink/'+str(count['Drink'])+'.jpg', roi)       
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['I'])+'.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'Love/'+str(count['Love'])+'.jpg', roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'You/'+str(count['You'])+'.jpg', roi) 
   
    
cap.release()
cv2.destroyAllWindows()

