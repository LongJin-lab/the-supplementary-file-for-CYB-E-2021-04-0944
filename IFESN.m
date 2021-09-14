% load the data
data = load('forecasting.txt');
dataLen=size(data);
trainLen = round(dataLen*0.7);
testLen = dataLen-trainLen;
initLen =1000;
resSize = trainLen*0.5;
desired=1.0e-5;
[data]=NormalizationX(data,1,-1);

% generate the ESN reservoir
inSize = 1;
outSize = 1;
a = 1; % leaking rate
W = rand(resSize,resSize)-0.5;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* (1.25 /rhoW);
rhoW = abs(eigs(W,1,'LM',opt));
Win = (rand(resSize,1)-0.5) .* 1;
% allocated memory for the design (collected states) matrix
X = zeros(inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';
MSE=zeros(1,resSize);
MAE=zeros(1,resSize);
NRMSE=zeros(1,resSize);

% run the reservoir with the data and collect X
x = zeros(resSize,1);
Y = zeros(outSize,testLen);
for t = 1:trainLen
    u = data(t);
    x = (1-a)*x + a*tanh( Win*[u] + W*x );
    if t > initLen
        X(:,t-initLen) = [u;x];
    end
end
u = data(trainLen+1);
X_in=X(1,:);
X_temp=X(2,:);  
z=X_temp'*(1/(X_temp*X_temp'));
out_u=pinv(X_in);
traintime=toc;
for q = 2:resSize  
   X_temp_add=X(q+1,:); 
   b=X_temp_add';
   b_t=b';
   part1=(1/(b_t*b))*(b_t*b*eye(size(b_t,2))-b*b_t)*...
       (z*(X_temp*b)*(b_t*z)/(b_t*b-(b_t*z)*(X_temp*b))+z);
   part2=(1/(b_t*b))*(b-part1*X_temp*b);
   zz=[part1,part2];
   z=zz; 
   X_temp=[X_temp;X_temp_add];
   Wout=Yt*zz;   
   xx=X_temp(:,end);   
   W_q=W(1:q,1:q);
   Win_q=Win(1:q,:);  
   tt=1;
   for t = 1:testLen       
       u = data(trainLen+tt+1);
       xx = (1-a)*xx + a*tanh( Win_q*[u] + W_q*xx );
       y = Wout*[xx]+out_u(q,:)*[u];
       Y(:,t) = y;
       tt=tt+1;      
   end
   errorLen = testLen;
   mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen; 
   if mse<=desired   
      fprintf('The optimal number of nodes is %d, with the mse %x\n',q,mse);
      break;
   end
   ave=mean(data(trainLen+2:trainLen+errorLen+1));
   nrmse = sqrt(sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./sum((data(trainLen+2:trainLen+errorLen+1)'-ave).^2));
   mae = mean(abs(data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)));  
   MSE(1,q)=mse;
   MAE(1,q)=mae;
   NRMSE(1,q)=nrmse;  
   disp( ['MSE = ', num2str( mse )] );
   disp( ['MAE = ', num2str( mae )] );
   disp( ['NRMSE = ', num2str( nrmse )] );   
end
   figure(1);
   plot( data(trainLen+2:trainLen+testLen+1), 'color', [0,0.75,0] );
   hold on;
   plot( Y', 'b' );
   hold off;
   axis tight;
   title('Target and generated signals');
   legend('Target signal', 'generated signal');     

function [data]=NormalizationX(data,ymax,ymin)
    xmax=max(data);
    xmin=min(data);
    data = (ymax-ymin)*(data-xmin)/(xmax-xmin) + ymin;
end
