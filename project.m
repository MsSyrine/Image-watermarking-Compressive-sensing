clear all;
clear D1;
close all;
getd = @(p)path(p,path);
getd('toolbox_signal/');%toolbox to be used in generating+acquiring CS coeffecients
getd('toolbox_general/');
% % % % We first make use of P low pass linear measurements to
% % % remove the low frequency content of the image. 
% % % % Natural images are not only sparse over a wavelet domain.
% % % They also exhibit a fast decay of the coefficient through the scale. 
% % % The coarse (low pass) wavelets caries much of the image energy. 
% % % It thus make sense to measure directly the low pass coefficients.
% % % % We load an image f?Rn2 of n×n pixels.
name = '01';
n = 256;
f1 = load_image(name, n);
f = rescale(f1,n);
% % %  %---------------------------------------
% % %  %% % % % %CS coeffs :
% % %  %---------------------------------------
% Shortcuts for the wavelet transform {?f,?m?}m.7
% We only compute up to a scale J so that only k0 sub-bands are transformed.
k0 = 2;
J = log2(n)-k0;
Wav  = @(f)perform_wavelet_transf(f,J,+1);
WavI = @(x)perform_wavelet_transf(x,J,-1);
% % Compute the wavelet transform.
fw = Wav(f);
figure;
% % Display the coefficients.
% % an approximation fLow using the P=22J=(n/k0)2 low pass coefficients.
plot_wavelet(fw, J);
ex1
fwLow = zeros(n);
fwLow(1:2^J,1:2^J) = fw(1:2^J,1:2^J);
fLow = WavI(fwLow);
myplot = @(f1)imageplot(clamp(f1), ['PSNR=' num2str(psnr(f,f1),3) 'dB']);
figure;
myplot(fLow);
% % We consider a compressed sensing operator that corresponds to randomized orthogonal projections.
% % 
% % Extract the high pass wavelet coefficients, x0={?f,?m?}m?I0.
A = ones(n,n); A(1:2^J,1:2^J) = 0;
I0 = find(A==1);
x0 = fw(I0);
N = length(x0);%Number of coefficients
% % % % Number P0=22J=(n/k0)2 of low pass measurements.
P0 = (n/2^k0)^2;
% % % % Number of CS measurements.
P = 4 * P0;
% % % Generate random permutation operators S1,S2:RN?RN 
% % % so that Sk(x)i=x?k(i) where ?k??N is a random permutation of {1,…,N}.
sigma1 = randperm(N)';
sigma2 = randperm(N)';
S1 = @(x)x(sigma1);
S2 = @(x)x(sigma2);
% % % % The adjoint (and also inverse) operators S?1,
% % % % S?2 (denoted S1S,S2S) corresponds to the inverse permutation ??k such that ??k??k(i)=i.
sigma1S = 1:N; sigma1S(sigma1) = 1:N;
sigma2S = 1:N; sigma2S(sigma2) = 1:N;
S1S = @(x)x(sigma1S);
S2S = @(x)x(sigma2S);

% % % We consider a CS operator ?:RN?RP that corresponds to a projection on randomized atoms
% % % (?x)i=?x,??2(i)?
% % % where ?i is a scrambled orthogonal basis
% % % ?i(x)=ci(?1(x))
% % % where {ci}i is the orthogonal DCT basis.
% % % 
% % % This can be rewritten in compact operator form as
% % % ?x=(S2?C?S1(x))?P
% % % where S1,S2 are the permutation operators, and ?P selects the P first entries of a vector.
downarrow = @(x)x(1:P);
Phi = @(x)downarrow(S2(dct(S1(x))));
% % % The adjoint operator is
% % % ??x=S?1?C??S?2(x?P)
% % % where ?P append N?P zeros at the end of a vector, and C? is the inverse DCT transform.
uparrow = @(x)[x; zeros(N-P,1)];
PhiS = @(x)S1S(idct(S2S(uparrow(x))));
% % % Perform the CS (noiseless) measurements.
y = Phi(x0);;%y contains the CS Coefficients of the image
% % figure;
% % imagesc(reshape(y,218,218));
% % colormap gray
% 
% % % % %  %---------------------------------------
% % % % %  %% % % % watermark construction 
% % % % %  %---------------------------------------
% 
y1= y.';
ch=num2str(y1);
ch1=ch ;
 
l= strsplit(ch);
l2=l.';
l3=str2num(char(l2));
%mes1='0xfad2d8d7a6c8d8211e5c218211c421d2196e8ddc6a0e099e2521ea7921da621c';
% msg=strcat(ch1,'_',mes1);
% % % % %  without blockchain ID
 msg=strcat(ch1);
% save('pfile.mat','msg')
%  
%  
%   % % f1 contains the loaded image. 
%   % % sX = size(f1);
% % % % %  %---------------------------------------
% % % % %  %% % % % %Data Embedding:
% % % % %  %---------------------------------------
imname='01_watermarked.png';
% 
coverImage=f1;
message=char(msg);
[LL,LH,HL,HH] = dwt2(coverImage,'bior2.2'); 
if size(message) > size(coverImage,1) * size(coverImage,2)
    error ('message too big to embed');
end
bit_count = 0; steg_coeffs = [4, 4.75, 5.5, 6.25, 7];
area=zeros(1,size(message,2)+1); % preallocate
for jj=1:size(message,2)+1
    if jj > size(message,2)
        charbits = [0,0,0,0,0,0,0,0];
    else charbits = dec2bin(message(jj),8)'; 
        charbits = charbits(:)'-'0';
    end
    for ii=1:8
        bit_count = bit_count + 1;
        if charbits(ii) == 1
            if HH(bit_count) <= 0
                HH(bit_count) = steg_coeffs(randi(numel(steg_coeffs)));
            end
        else
            if HH(bit_count) >= 0
                HH(bit_count) = -1 * steg_coeffs(randi(numel(steg_coeffs)));
            end
        end
    end
end
stego_image = idwt2(LL,LH,HL,HH,'bior2.2');
imwrite(uint8(stego_image),imname);
 
% % % % % % % figure('name','Watermarked Image');
% % % % % % % imshow(uint8(stego_image));
%  
% 
% % % % % % % % %  %---------------------------------------
% % % % % % %  %% % % % %Filtres:
% % % % % % % % %  %---------------------------------------
% % % % % % % % % %  
% % imname='20_watermarked.png';
% %  I=imread(imname);
% %  imfiltered='24_Filtered.png';
% %   
% % % % % % % % % % %filtre gaussien
% Iblur1 = imgaussfilt(I,1);
% imwrite(uint8(Iblur1),imfiltered);
% figure;
% imshow(I);
% title('Original image');
% % % % figure;
% % % % imshow(Iblur1);
% % % % title('Smoothed image, \sigma = 1');
% %  
% % % % %filtre 3*3 mean
% blurredImage = conv2(single(I), ones(3)/9, 'same');
% imwrite(uint8(blurredImage),imfiltered);
% figure(1);
% imshow(I);
% title('Original image');
% % % figure(2);
% % % imshow(uint8(blurredImage));
% % % title('Smoothed image');
 
% % % %  % % %filtre 5*5 mean
% blurredImage = conv2(single(I), ones(5)/25, 'same');
% imwrite(uint8(blurredImage),imfiltered);
% figure(1);
% % % imshow(I);
% % % title('Original image');
% % % figure(2);
% % % imshow(uint8(blurredImage));
% % % title('Smoothed image');
 
% % % % % % % filtre median 3*3 
% blurredImage = medfilt2(I, [3 3]);
% imwrite(uint8(blurredImage),imfiltered);
% % % figure(1);
% % % imshow(I);
% % % title('Original image');
% % % figure(2);
% % % imshow(uint8(blurredImage));
% % % title('Smoothed image');

% % % % %filtre median 5*5 
% blurredImage = medfilt2(I, [5 5]);
% imwrite(uint8(blurredImage),imfiltered);
% % % figure(1);
% % % imshow(I);
% % % title('Original image');
% % % figure(2);
% % % imshow(uint8(blurredImage));
% % % title('Smoothed image');
% % %  
% % % % % %---------------------------------------
% % % %  %% % % % %Data Extraction:
% % % % % %---------------------------------------
% new_Stego = imread(imfiltered);
% [LL,LH,HL,HH] = dwt2(new_Stego,'bior2.2'); 
% message1 = '';
% msgbits = '';   msgChar  = '';
% for ii = 1:size(HH,1)*size(HH,2) 
%     if HH(ii) > 0
%         msgbits = strcat (msgbits, '1');
%     elseif HH(ii) < 0
%         msgbits = strcat (msgbits, '0');
%     else return; 
%     end
%  
% end

%  %---------------------------------------
%  %% % % % %comparing /testing
%  %---------------------------------------
% % %original message: Binary of the message value.
% data2=zeros();
% for(i=1:length(msg))
% d=msg(i)+0;
% data2=[data2 d];
% end
% data2 =reshape(dec2bin(data2, 8).'-'0',1,[]);
% mydata=char((reshape((data2+'0'), 8,[]).'));
% 
% % %extracted message: Binary of the message value.
% binary = reshape(msgbits.'-'0',1,[]);
% mybin=char((reshape((binary(1:218)+'0'), 8,[]).'));
% 
% wsize=size(mybin,1);
% %mybin(wsize,1:size(mydata,1)) =mybin+0; % if needed, this right-pads with zero or causes t to grow
% 
% res = sum( mydata(1:wsize,:)~= mybin );
%  D1 = pdist2(mydata( 1:wsize,:) ,mybin  ,'hamming');
% 
%  mean2(D1) %average of the hamming distance
 %---------------------------------------
 %recovery cs
 %---------------------------------------
 
% %ex2
% f1w = fw;
% f1w(I0) = PhiS(y);
% fL2 = WavI(f1w);
% figure; myplot( fL2 );
% 
% %ex3
% ProxF = @(x,gamma)x + PhiS(y-Phi(x)); 
% ProxG = @(x,gamma)max(0,1-gamma./max(1e-21,abs(x))).*x;
% rProxF = @(x,gamma)2*ProxF(x,gamma)-x;
% rProxG = @(x,gamma)2*ProxG(x,gamma)-x;
% mu = 1;
% gamma = 1;
% %ex4
% G = []; 
% F = [];
% tx = zeros(N,1);
% niter = 300;
% for i=1:niter
%     tx = (1-mu/2)*tx + mu/2*rProxG( rProxF(tx,gamma),gamma );
%     x = ProxF(tx,gamma);
%     G(i) = norm(x,1);
%     F(i) = norm(y-Phi(x));
% end
% clf;
% h = plot(G);
% set(h, 'LineWidth', 2);
% axis tight;
% %ex5
% fCSw = fw;
% fCSw(I0) = x;
% fCS = WavI(fCSw);
% figure; myplot( fCS );
% 
% w = 4;
% v = 1:w:n;
% dv = 0:w-1;
% [dX,dY,X,Y] = ndgrid(dv,dv,v,v);
% q = size(X,3);
% dX = reshape(dX, [w*w q*q]);
% dY = reshape(dY, [w*w q*q]);
% X = reshape(X, [w*w q*q]);
% Y = reshape(Y, [w*w q*q]);
% I = find( sum(X+dX>n | Y+dY>n)  );
% X(:,I) = [];
% Y(:,I) = [];
% dX(:,I) = [];
% dY(:,I) = [];
% U = zeros(n,n);
% U(I0) = 1:N;
% Ind = X+dX + (Y+dY-1)*n;
% I = U(Ind);
% I(:,sum(I==0)>0) = [];
% G = @(x)sum( sqrt(sum(x(I).^2)) );
% [A,tmp] = meshgrid( randperm(size(I,2)) , ones(w*w,1));
% x = zeros(N,1); x(I) = A;
% Z = zeros(n,n); Z(I0) = x;
% figure;
% imageplot(Z);
% colormap jet(256);
% %ex6
% Energy = @(x)sqrt(sum(x(I).^2));
% SoftAtten = @(x,gamma)max(0, 1-gamma./abs(x));
% EnergyAtten = @(x,gamma)repmat(SoftAtten(Energy(x),gamma), [w*w 1] );
% Flatten = @(x)x(:);
% ProxG = @(x,gamma)accumarray(I(:), Flatten(EnergyAtten(x,gamma)), [N 1], @prod) .* x;
% rProxG = @(x,gamma)2*ProxG(x,gamma)-x;
% 
% %RD Iterative CS recovery (l1 norm)
% g = []; 
% tx = zeros(N,1);
% niter = 210;
% for i=1:niter
%     tx = (1-mu/2)*tx + mu/2*rProxG( rProxF(tx,gamma),gamma );
%     x = ProxF(tx,gamma);
%     g(i) = G(x);
% end
% figure;
% h = plot(g);
% set(h, 'LineWidth', 2);
% axis tight;
% 
% %ex8
% fCSBlockw = fw;
% fCSBlockw(I0) = x;
% fCSBlock = WavI(fCSBlockw);
