% clc;
% clear all;
I = imread('02_watermarked.png');
I1=I;
[row coln]= size(I);
I= double(I);
%---------------------------------------------------------
% Subtracting each image pixel value by 128 
%--------------------------------------------------------
I = I - (128*ones(256));

quality = input('What quality of compression you require - ');

%----------------------------------------------------------
% Quality Matrix Formulation
%----------------------------------------------------------
Q50 = [ 16 11 10 16 24 40 51 61;
     12 12 14 19 26 58 60 55;
     14 13 16 24 40 57 69 56;
     14 17 22 29 51 87 80 62; 
     18 22 37 56 68 109 103 77;
     24 35 55 64 81 104 113 92;
     49 64 78 87 103 121 120 101;
     72 92 95 98 112 100 103 99];
 
 if quality > 50
     QX = round(Q50.*(ones(8)*((100-quality)/50)));
     QX = uint8(QX);
 elseif quality < 50
     QX = round(Q50.*(ones(8)*(50/quality)));
     QX = uint8(QX);
 elseif quality == 50
     QX = Q50;
 end
 
 
 
 
%----------------------------------------------------------
% Formulation of forward DCT Matrix and inverse DCT matrix
%----------------------------------------------
DCT_matrix8 = dct(eye(8));
iDCT_matrix8 = DCT_matrix8';   %inv(DCT_matrix8);




%----------------------------------------------------------
% Jpeg Compression
%----------------------------------------------------------
dct_restored = zeros(row,coln);
QX = double(QX);
%----------------------------------------------------------
% Jpeg Encoding
%----------------------------------------------------------
%----------------------------------------------------------
% Forward Discret Cosine Transform
%----------------------------------------------------------

for i1=[1:8:row]
    for i2=[1:8:coln]
        zBLOCK=I(i1:i1+7,i2:i2+7);
        win1=DCT_matrix8*zBLOCK*iDCT_matrix8;
        dct_domain(i1:i1+7,i2:i2+7)=win1;
    end
end
%-----------------------------------------------------------
% Quantization of the DCT coefficients
%-----------------------------------------------------------
for i1=[1:8:row]
    for i2=[1:8:coln]
        win1 = dct_domain(i1:i1+7,i2:i2+7);
        win2=round(win1./QX);
        dct_quantized(i1:i1+7,i2:i2+7)=win2;
    end
end




%-----------------------------------------------------------
% Jpeg Decoding 
%-----------------------------------------------------------
% Dequantization of DCT Coefficients
%-----------------------------------------------------------
for i1=[1:8:row]
    for i2=[1:8:coln]
        win2 = dct_quantized(i1:i1+7,i2:i2+7);
        win3 = win2.*QX;
        dct_dequantized(i1:i1+7,i2:i2+7) = win3;
    end
end
%-----------------------------------------------------------
% Inverse DISCRETE COSINE TRANSFORM
%-----------------------------------------------------------
for i1=[1:8:row]
    for i2=[1:8:coln]
        win3 = dct_dequantized(i1:i1+7,i2:i2+7);
        win4=iDCT_matrix8*win3*DCT_matrix8;
        dct_restored(i1:i1+7,i2:i2+7)=win4;
    end
end
I2=dct_restored;



% ---------------------------------------------------------
% Conversion of Image Matrix to Intensity image
%----------------------------------------------------------


K=mat2gray(I2);
imname='02.jpg';
imwrite(K,imname);

%----------------------------------------------------------
%Display of Results
%----------------------------------------------------------
figure(1);imshow(I1);title('original image');
figure(2);imshow(K);title('restored image from dct');

% % % % %---------------------------------------
% % %  %% % % % %Data Extraction:
% % % % %---------------------------------------
new_Stego = imread(imname);
[LL,LH,HL,HH] = dwt2(new_Stego,'bior2.2'); 
message1 = '';
msgbits = '';   msgChar  = '';
for ii = 1:size(HH,1)*size(HH,2) 
    if HH(ii) > 0
        msgbits = strcat (msgbits, '1');
    elseif HH(ii) < 0
        msgbits = strcat (msgbits, '0');
    else return; 
    end
 
end


%  %---------------------------------------
 %% % % % %comparing /testing
 %---------------------------------------
%original message: Binary of the message value.
data2=zeros();
for(i=1:length(msg))
d=msg(i)+0;
data2=[data2 d];
end
data2 =reshape(dec2bin(data2, 8).'-'0',1,[]);
mydata=char((reshape((data2+'0'), 8,[]).'));

%extracted message: Binary of the message value.
binary = reshape(msgbits.'-'0',1,[]);
mybin=char((reshape((binary(1:704)+'0'), 8,[]).'));

 
%  taille=size(message);
% %  message1(1:taille)-message;
% t(1,:) = message+1; % +1 to make sure there are no zeros
% t(2,1:numel(message1)) = message1+1; % if needed, this right-pads with zero or causes t to grow
% res = sum(t(1,:)~=t(2,:)); % result = 6

wsize=size(mybin,1);
%mybin(wsize,1:size(mydata,1)) =mybin+0; % if needed, this right-pads with zero or causes t to grow

res = sum( mydata(1:wsize,:)~= mybin );
 D1 = pdist2(mydata( 1:wsize,:) ,mybin  ,'hamming');

 mean2(D1) %average of the hamming distance