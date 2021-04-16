%Load single  image
A = imread('02_compare.png'); 
 
% the average of 3^2, or 9 values(filters the multidimensional array A with the multidimensional filter h)

B = imread('02_watermarked.png');


%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
[mssim, ssim_map] = ssim_index(A, B);

fprintf('The SSIM value is %0.4f.\n',mssim);
% The SSIM value is 0.9407.
  
figure, imshow(ssim_map,[]);
title(sprintf('ssim Index Map - Mean ssim Value is %0.4f',ssim_map));