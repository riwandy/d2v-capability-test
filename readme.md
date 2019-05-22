# Detection of bolt direction
This is a simple program to detect one of the 6 possible direction of a given bolt of image under assumption that the bolt is in the middle of a 409 x 409 pixels image.

## Description
This program is written and tested on **_Python 3.6_**. The image processing is done by utilizing some functions from open source OpenCV library. The Image is preprocessed by performing **_Histogram Equalization_** to get more contrast and **_Gaussian Blur_** to blur out unwanted details when running edge-detection algorithm. The edges from the image is then extracted using Canny. The extracted edges is then fed into **_Probabilistic Hough Transformation_** algorithm to get straight line edges. By looping through all these lines, we filter out those that form an angle around 120 degrees since every corner of the bolt is roughly 120 degrees. One of these selected corners is then selected and an arrow **_corner_degree/2_** degrees from one of the angle-forming line is drawn with one end being the corner hence the line should cross the center of the bolt and the other corner across it. The angle in the **_result.txt_** is de degree of the arrow from horizontal axis.

## Instruction
To run the program simply run this command via terminal.
```bash
python3 main.py
```

