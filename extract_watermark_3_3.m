
% % % % %---------------------------------------
% % %  %% % % % %Data Extraction:
% % % % %---------------------------------------
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

 %---------------------------------------
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
mybin=char((reshape((binary(1:6072)+'0'), 8,[]).'));

 
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