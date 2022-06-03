
import pyttsx3
from predict import *


# Initialise Text to speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 105)
engine.setProperty('voice', 1)

window_name = "word prediction"

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 100)
# fps = int(cap.get(5))
#print("fps:", fps)

sentence = ""
while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	if ret is None:
		print("No Frame Captured")
		continue

	# bounding box which captures ASL sign to be detected by the system
	cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)  

	# Crop blue rectangular area(ROI)
	img1 = frame[10:410, 220:622]
	gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),2)
	th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	ret, test_image = cv2.threshold(th3, 90, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	test_image = cv2.resize(test_image,(300,300))
	
    
    

	

	naya = cv2.bitwise_and(test_image, test_image, mask=None)
	cv2.imshow("naya", naya)
	hand_bg_rm = naya
	hand = img1

	
	# Control Key
	c = cv2.waitKey(25) & 0xff

	# Speak the sentence
	if len(sentence) > 0 and c == ord('s'):
		engine.say(sentence)
		engine.runAndWait()
	# Clear the sentence
	if c == ord('c') or c == ord('C'):
		sentence = ""
	# Delete the last character
	if c == ord('d') or c == ord('D'):
		sentence = sentence[:-1]

	# Put Space between words
	if c == ord('m') or c == ord('M'):
		sentence += " "

	#If  valid hand area is cropped
	if hand.shape[0] != 0 and hand.shape[1] != 0:
		conf, label = which(hand_bg_rm)
		print(conf)
		if conf >= THRESHOLD:
			cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9, (255,0, 0))
		if c == ord('n') or c == ord('N'):
			sentence += label
	
	cv2.putText(frame, sentence, (50, 450), cv2.FONT_HERSHEY_COMPLEX, .9, (0, 0,255))
	cv2.imshow(window_name, frame)
	# If pressed ESC break
	if c == 27:
		cap.release()
		cv2.destroyAllWindows()
		exit()
cap.release()
cv2.destroyAllWindows()




