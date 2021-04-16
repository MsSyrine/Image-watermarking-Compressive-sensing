% CalcPSNR returns the PSNR between the original image and its approximation
%
% Calculates the Peak Signal to Noise Ratio (PSNR) between 2 matrices containing
% pixel intensity values.
%
% Usage:            psnr = CalcPSNR( mImage1, mImage2 );
%
% Inputs:
%   mImage1         matrix of pixel intensity values representing the original image
%   mImage2         matrix of pixel intensity values representing the approximated image
%   maxIntensity    maximum allowed pixel intensity, defualt is 256 (8 bit image) 
%
% Outputs:
%    psnr       the PSNR resulting from the  approximation

% See   http://www.ncrg.aston.ac.uk/Projects/HNLApprox/
%       http://www.nonlinear-approx.info/

vue=1; %breakdancers
%vue=3; %ballet
%vue=40; %champagne

m=16;
mm=5;
k=0;
p=zeros(1,mm);

for nn=21:10:51
    k=k+1;
    for n=1:m
        %imageFile = ['c' num2str(n) 'v' num2str(vue) '.png'];
        mImag1 = double( imread('01_compare.png') );
        [h w z]=size(mImag1);
        for i = 1:h
            for j = 1:w
                mImage1(i,j)=mImag1(i,j,1);
            end
        end
       % imageFile = ['c' num2str(n) 'v' num2str(vue) '_Q' num2str(nn) '_watermarked.png'];
        mImag2 = double( imread('01_watermarked.png') );
        [h w z]=size(mImag2);
        for i = 1:h
            for j = 1:w
                mImage2(i,j)=mImag2(i,j,1);
            end
        end
        maxIntensity = 255;
        for i = 1:h
            for j = 1:w
                mError(i,j) = mImage1(i,j) - mImage2(i,j);
            end
        end
        [ y, x ] = size(mError);
        mse = sum( mError(:).^2 )/( y*x );
        psnr = 20*log10(maxIntensity/sqrt(mse));
        p(k)=p(k)+psnr;
    end
end
disp(['PSNR : [' num2str(p(1)/m) ', ' num2str(p(2)/m) ', ' num2str(p(3)/m) ', ' num2str(p(4)/m) ', ' num2str(p(5)/m) ' ]']);
disp('                            ');

 p46=0;
 for n=1:m
        %imageFile = ['c' num2str(n) 'v' num2str(vue) '.bmp'];
        mImag1 = double( imread('01_compare.png') );
        [h w z]=size(mImag1);
        for i = 1:h
            for j = 1:w
                mImage1(i,j)=mImag1(i,j,1);
            end
        end
       % imageFile = ['c' num2str(n) 'v' num2str(vue) '_Q46_HEVC.bmp'];
        mImag2 = double( imread('01_watermarked.png') );
        [h w z]=size(mImag2);
        for i = 1:h
            for j = 1:w
                mImage2(i,j)=mImag2(i,j,1);
            end
        end
        maxIntensity = 255;
        for i = 1:h
            for j = 1:w
                mError(i,j) = mImage1(i,j) - mImage2(i,j);
            end
        end
        [ y, x ] = size(mError);
        mse = sum( mError(:).^2 )/( y*x );
        psnr = 20*log10(maxIntensity/sqrt(mse));
        p46=p46+psnr;
    end
    disp(   p46/m );   

 p36=0;
 for n=1:m
       % imageFile = ['c' num2str(n) 'v' num2str(vue) '.bmp'];
        mImag1 = double( imread('01_compare.png') );
        [h w z]=size(mImag1);
        for i = 1:h
            for j = 1:w
                mImage1(i,j)=mImag1(i,j,1);
            end
        end
        imageFile = ['01_watermarked.png'];
        mImag2 = double( imread(imageFile) );
        [h w z]=size(mImag2);
        for i = 1:h
            for j = 1:w
                mImage2(i,j)=mImag2(i,j,1);
            end
        end
        maxIntensity = 255;
        for i = 1:h
            for j = 1:w
                mError(i,j) = mImage1(i,j) - mImage2(i,j);
            end
        end
        [ y, x ] = size(mError);
        mse = sum( mError(:).^2 )/( y*x );
        psnr = 20*log10(maxIntensity/sqrt(mse));
        p36=p36+psnr;
    end
    disp(   p36/m );   
%}


%{
 p11=0;
 for n=1:m
        imageFile = ['c' num2str(n) 'v' num2str(vue) '.bmp'];
        mImag1 = double( imread(imageFile) );
        [h w z]=size(mImag1);
        for i = 1:h
            for j = 1:w
                mImage1(i,j)=mImag1(i,j,1);
            end
        end
        imageFile = ['c' num2str(n) 'v' num2str(vue) '_Q01_HEVC.bmp'];
        mImag2 = double( imread(imageFile) );
        [h w z]=size(mImag2);
        for i = 1:h
            for j = 1:w
                mImage2(i,j)=mImag2(i,j,1);
            end
        end
        maxIntensity = 255;
        for i = 1:h
            for j = 1:w
                mError(i,j) = mImage1(i,j) - mImage2(i,j);
            end
        end
        [ y, x ] = size(mError);
        mse = sum( mError(:).^2 )/( y*x );
        psnr = 20*log10(maxIntensity/sqrt(mse));
        p11=p11+psnr;
    end
    disp(   p11/m );   

 p5=0;
 for n=1:m
        imageFile = ['c' num2str(n) 'v' num2str(vue) '.bmp'];
        mImag1 = double( imread(imageFile) );
        [h w z]=size(mImag1);
        for i = 1:h
            for j = 1:w
                mImage1(i,j)=mImag1(i,j,1);
            end
        end
        imageFile = ['c' num2str(n) 'v' num2str(vue) '_Q5_HEVC.bmp'];
        mImag2 = double( imread(imageFile) );
        [h w z]=size(mImag2);
        for i = 1:h
            for j = 1:w
                mImage2(i,j)=mImag2(i,j,1);
            end
        end
        maxIntensity = 255;
        for i = 1:h
            for j = 1:w
                mError(i,j) = mImage1(i,j) - mImage2(i,j);
            end
        end
        [ y, x ] = size(mError);
        mse = sum( mError(:).^2 )/( y*x );
        psnr = 20*log10(maxIntensity/sqrt(mse));
        p5=p5+psnr;
    end
    disp(   p5/m );   
  
    %}
    


    

