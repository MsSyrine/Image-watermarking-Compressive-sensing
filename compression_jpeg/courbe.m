hold on
 
xdata=[10 20 30 40 50 60 70 80 90 100 ];
% distances=[0.4933 0.4910 0.4876 0.4800 0.4779 0.4545 0.4345 0.4217 0.3915 0.3107 ];
distances=[0.4965 0.4901 0.4858 0.4807 0.4726 0.4581 0.4403 0.4200 0.3982 0.3190];
ydata=[0.1 0.2 0.3 0.4 0.5 ];
% for k = 1:length(xdata)
%     for j = 1:length(ydata)
%         if(distances(k,j) < 1e6)
%             plot([xdata(k) xdata(j)], [ydata(k) ydata(j)]);
%         end
%     end
% end
figure,
plot(xdata,distances,'o');

 title('Robustness to JPEG Compression ')
ylabel('Hamming Distance');
xlabel('Quality ratio');


% set(gca,'XTick',[10 20 30 40 50 60 70 80 90 100])
% set(gca,'xticklabel',({'10' ,'20', '30' ,'40', '50' ,'60' ,'70' ,'80' ,'90' ,'100'}))
