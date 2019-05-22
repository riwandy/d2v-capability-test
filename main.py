import cv2
import glob
import numpy as np
from math import atan2,degrees,acos,cos,sin,radians,tan

filename    = "testimg/*"
images      = sorted(glob.glob(filename))

# Helper function to calculate distance between 2 points
def distance(x1,y1,x2,y2):
    dist = (abs(x2-x1)**2)+(abs(y2-y1)**2)**0.5
    return dist


output = []

# Main program
for fname in images:

    # load image and crop
    img = cv2.imread(fname)
    crop_img = img[160:240,160:240]

    # convert img to grayscale, equalize to get sharper image and blur to avoid noises
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    width, height = gray.shape
    equ = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equ,(9,9),0)

    # find edges from the blurred image
    edges = cv2.Canny(blur,85,255)

    # set done flag to false
    done = False

    # use Probabilistic Hough Transform to detect straight lines in edges detected with canny
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=15, maxLineGap=5)

    # iterate through all lines found with Probabilistic Hough Transform
    if lines is not None:
        for line in lines:
            for line2 in lines:
                # get the two endpoints of every line
                x1, y1, x2, y2 = line[0]
                x3, y3, x4, y4 = line2[0]

                if x2-x1 == 0 or x4-x3 == 0:
                    continue
                
                slope1 = (y2-y1)/float(x2-x1)
                slope2 = (y4-y3)/float(x4-x3)
                
                if 1+slope1*slope2 == 0:
                    continue

                theta = abs((slope2-slope1)/(1+slope1*slope2))
                theta = np.arctan(theta)
                theta = theta*90

                # filter corners
                if theta>70 and theta<130:
                    range = 10
                    the_x = None
                    the_y = None
                    if distance(x1,y1,x3,y3) <= range:
                        the_x = int((x1+x3)/2)
                        the_y = int((y1+y3)/2)

                        length1 = ((x2-the_x)**2+(y2-the_y)**2)**0.5
                        length2 = ((x4-the_x)**2+(y4-the_y)**2)**0.5

                        vec1x = x2 - the_x
                        vec1y = the_y - y2
                        
                        vec2x = x4 - the_x
                        vec2y = the_y - y4

                        dstx = the_x+2*(vec1x+vec2x)
                        dsty = the_y-2*(vec1y+vec2y)

                        dotProduct = (vec1x*vec2x + vec1y*vec2y)
                        degree = degrees(acos(dotProduct/(length1*length2)))
                    elif distance(x2,y2,x3,y3) <= range:
                        the_x = int((x2+x3)/2)
                        the_y = int((y2+y3)/2)
                        
                        length1 = ((x1-the_x)**2+(y1-the_y)**2)**0.5
                        length2 = ((x4-the_x)**2+(y4-the_y)**2)**0.5

                        vec1x = x1 - the_x
                        vec1y = the_y - y1
                        
                        vec2x = x4 - the_x
                        vec2y = the_y - y4

                        dstx = the_x+2*(vec1x+vec2x)
                        dsty = the_y-2*(vec1y+vec2y)

                        dotProduct = (vec1x*vec2x + vec1y*vec2y)
                        degree = degrees(acos(dotProduct/(length1*length2)))
                    elif distance(x1,y1,x4,y4) <= range:
                        the_x = int((x1+x4)/2)
                        the_y = int((y1+y4)/2)
                        
                        
                        length1 = ((x2-the_x)**2+(y2-the_y)**2)**0.5
                        length2 = ((x3-the_x)**2+(y3-the_y)**2)**0.5

                        vec1x = x2 - the_x
                        vec1y = the_y - y2
                        
                        vec2x = x3 - the_x
                        vec2y = the_y - y3

                        dstx = the_x+2*(vec1x+vec2x)
                        dsty = the_y-2*(vec1y+vec2y)

                        dotProduct = (vec1x*vec2x + vec1y*vec2y)
                        degree = degrees(acos(dotProduct/(length1*length2)))
                    elif distance(x2,y2,x4,y4) <= range:
                        the_x = int((x2+x4)/2)
                        the_y = int((y2+y4)/2)
                        
                        
                        length1 = ((x1-the_x)**2+(y1-the_y)**2)**0.5
                        length2 = ((x3-the_x)**2+(y3-the_y)**2)**0.5

                        vec1x = x1 - the_x
                        vec1y = the_y - y1
                        
                        vec2x = x3 - the_x
                        vec2y = the_y - y3

                        dstx = the_x+2*(vec1x+vec2x)
                        dsty = the_y-2*(vec1y+vec2y)

                        dotProduct = (vec1x*vec2x + vec1y*vec2y)
                        degree = degrees(acos(dotProduct/(length1*length2)))

                    if(the_x != None and the_y != None):
                        if not done:
                            # cv2.line(imgClr, (x3, y3), (x4, y4), (0, 0, 255), 1)
                            
                            # prepare matrices for rotating one of the vector
                            matrix_cw = np.array([[cos(radians(degree/2)), sin(radians(degree/2))],
                                               [-sin(radians(degree/2)), cos(radians(degree/2))]])

                            matrix_ccw = np.array([[cos(radians(degree/2)), -sin(radians(degree/2))],
                                               [sin(radians(degree/2)), cos(radians(degree/2))]])

                            vector = np.array([vec1x, vec1y])

                            # calculate the origin of the arrow if the matrix is rotated clockwise or counter clockwise
                            point_cw = np.matmul(matrix_cw,vector)
                            point_ccw = np.matmul(matrix_ccw,vector)

                            # calculate the origin from center of cropped image, closer one will be drawn
                            dist_cw = distance(int(the_x+point_cw[0]),int(the_y-point_cw[1]),40,40)
                            dist_ccw = distance(int(the_x+point_ccw[0]),int(the_y-point_ccw[1]),40,40)

                            if dist_cw>dist_ccw:
                                cv2.arrowedLine(img, (int(the_x+3*point_ccw[0])+160,int(the_y-3*point_ccw[1])+160), (the_x+160, the_y+160), (0,0,255), 1, cv2.LINE_AA)
                                angle = abs(degrees(atan2((point_ccw[1]-the_y),(the_x-point_ccw[0]-40))))
                            else:
                                cv2.arrowedLine(img, (int(the_x+3*point_cw[0])+160,int(the_y-3*point_cw[1])+160), (the_x+160, the_y+160), (0,0,255), 1, cv2.LINE_AA)
                                angle = abs(degrees(atan2((point_cw[1]-the_y),(the_x-point_cw[0]-40))))
                            output.append(fname[8:]+","+str(round(angle,2)))
                            
                            
                            # set flag after an arrow has been drawn
                            done = True
                        
    
    cv2.imshow("output",img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        continue
    elif k == ord('x'):
        break

# prepare output file
f = open("result.txt", "w")
f.write("FILE NAME, DIRECTION\n")
for line in output:
    f.write(line+"\n")
f.close()