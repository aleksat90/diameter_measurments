import time
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def load_img(image):
    img = cv2.imread(image)
    
    # Blur using 3 * 3 kernel. 
    
    return img

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)

# img = load_img('krug.jpg')

# Read image. 
# img = cv2.imread('krug.jpg', cv2.IMREAD_COLOR) 
  
# Convert to grayscale. 
def writeText(input_image, org, text):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    #org = (50,50)
      
    # fontScale 
    fontScale = 0.5
       
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 1
       
    # Using cv2.putText() method 
    out_image = cv2.putText(input_image, text, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    return out_image

def detect_cricle_on_frame(frame):
    # Apply Hough transform on the blurred image. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray_blurred = cv2.blur(gray, (8, 8))
    # ret,gray_blurred = cv2.threshold(gray_blurred,60,255,cv2.THRESH_BINARY)
    # cv2.imshow('Binarna',gray_blurred)
    # gray_blurred = load_img(frame)
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                   param2 = 30, minRadius = 1, maxRadius =100) 
    try:
        # Draw circles that are detected. 
        if detected_circles is not None:       
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
            # print(detected_circles[0][0][0])
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
                # pt = detected_circles
                # a=pt[0][0][0]
                # b=pt[0][0][1]
                # r=pt[0][0][2]
                # a, b, r = pt[0], pt[1], pt[2] 
                print(a,b,r)
                # Draw the circumference of the circle. 
                cv2.circle(frame, (a, b), r, (0, 255, 0), 2) 
          
                # Draw a small circle (of radius 1) to show the center. 
                cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)
                frame = writeText(frame, (a,b), 'R='+str(r))
            cv2.imshow("Detected Circle", frame)
            print("Frame ploted")
            cv2.waitKey(10)
        print("return frame")
        # return frame
    except:
        print("Error not classified")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read(0)
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Call detect")
    slika = detect_cricle_on_frame(frame)
    # frame = detect_eyes(frame)
    # img = cv2.add(slika, frame)
    # cv2.imshow('Detektovanje krugova', slika)
    k = cv2.waitKey(1)
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()