import numpy as np
import cv2
import sys

class landingSite(object):
    # Image-wise parameters
    kernel_blur = np.ones((5,5),np.float32)/25
    kernel_morph = np.ones((2,2), np.uint8)
    image_width = 640
    image_height = 480
    min_area_ratio = 0.0009
    
    # Hough circle function parameters
    min_distance = 40
    param1 = 200
    param2 = 25 # Adjust this to change the accuracy of circle detection
    minRadius = 0
    maxRadius = 240
    
    # Define range of blue/red color in HSV
    blue = np.array([[100,100,100], [130,230,230]]) # [lower threshold, upper threshold]
    red = np.array([[0, 100, 70], [25, 255, 255]])
    red_wrap = np.array([[163, 100, 70], [179, 255, 255]])
    
    def __init__(self, src):
        self.src = src.copy()
        self.original = src.copy()
        self.image_height, self.image_width, self.channels = np.shape(src) # Get frame dimension
        self.red = self.image_process(self.red, self.red_wrap)
        self.blue = self.image_process(self.blue)
        
        # Show output
#         cv2.imshow('red', self.red)
        cv2.imshow('blue', self.blue)
        cv2.imshow('src', self.src)
        
    def image_process(self, color, color_wrap=None):
        # Blur the image
#         blur = cv2.GaussianBlur(self.original, (3,3), 0)
                
        # Convert from BGR to HSV
        hsv = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2HSV)
        
        if color_wrap is not None:
            lower_hue_red = cv2.inRange(hsv, color[0], color[1])
            upper_hue_red = cv2.inRange(hsv, color_wrap[0], color_wrap[1])
            dst = cv2.addWeighted(lower_hue_red, 1.0, upper_hue_red, 1.0, 0.0)
        else:
            dst = cv2.inRange(hsv, color[0], color[1])
        
        # Smooth the image
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.kernel_morph)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel_morph)
        
        contours, hierachy = cv2.findContours(dst.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each blob found
#         for idx, cnt in enumerate(contours):
#             cnt_area = cv2.contourArea(cnt)
#               
#             # Delete blobs that are too small
#             if cnt_area < (self.image_height*self.image_width)*self.min_area_ratio:
#                 cv2.drawContours(dst, contours, idx, np.array([0]), thickness=cv2.cv.CV_FILLED)
    
        if hierachy is not None:
            hierachy = np.array(hierachy[0])
             
            # Process each blob found
            for idx, (cnt, hrc) in enumerate(zip(contours, hierachy)):
                cnt_area = cv2.contourArea(cnt)
                  
                # Delete blobs that are too small
                if cnt_area < (self.image_height*self.image_width)*self.min_area_ratio:
                    cv2.drawContours(dst, contours, idx, np.array([0]), thickness=cv2.cv.CV_FILLED)
                if hrc[3] != -1:
                    cv2.drawContours(dst, contours, idx, np.array([255]), thickness=cv2.cv.CV_FILLED)
    
#         edges = cv2.Canny(dst, 200, 100)
#         cv2.imshow('edges', edges)
        
        # Find circles
        self.find_circles(dst)      
        
        return dst
    
    def find_circles(self, dst):
        circles = cv2.HoughCircles(dst.copy(), cv2.cv.CV_HOUGH_GRADIENT, 1.1, self.min_distance, 
                                            param1=self.param1, param2=self.param2, 
                                            minRadius=self.minRadius, maxRadius=self.maxRadius)
       
        # If circles are found
        if circles is not None:
            # Convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            
            # Find the largest circle
            biggest_circle_idx = np.argmax(circles[:,2])
            
            # Loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # Draw the circle in the output image
                cv2.circle(self.src, (x, y), r, (0, 255, 255), 2)
            
            x, y, r = circles[biggest_circle_idx]
            cv2.circle(self.src, (x, y), r, (0, 255, 0), 4)
            
        return 
            
if __name__ == '__main__':  
    # Instructions
    print "Press q to exit, p to pause"
    
    # Windows for output
#     cv2.namedWindow('red', cv2.WINDOW_NORMAL)
    cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
    cv2.namedWindow('src', cv2.WINDOW_NORMAL)
#     cv2.moveWindow('src', 0, 0)
     
    # Take in file
    if len(sys.argv) < 2:
        print "Please enter an video input."
        sys.exit()
        
    # Read file
    cap = cv2.VideoCapture(sys.argv[1])
    
    frame_counter = 0 # Used to loop video
    while(cap.isOpened()):
        # Get frame from video file
        retval, src = cap.read()
        frame_counter += 1
        
        # If the last frame is reached, reset the capture and the frame_counter
        if frame_counter == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            frame_counter = 0
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
            
        # Do the detection
        landingSite(src)

        # User input
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()