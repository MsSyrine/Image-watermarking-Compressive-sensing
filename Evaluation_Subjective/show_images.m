
 
%Store some images  %get all pictures ahead of time
imgs = cell(5,1);
imgs{1} = imread('18_compare.png');
imgs{2} = imread('black.png');
imgs{3} = imread('18_watermarked.png');

%
axis ij

for c = 1:4
for ii = 1:length(imgs)
    H = imshow(imgs{ii});
    pause(2)
    delete(H)
end
end