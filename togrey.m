RGB = imread('13.png');
newmap = rgb2gray(RGB);
imwrite(newmap,'13.png');