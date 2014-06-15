%% Face recognition
% Principal Component Analysis, Neural Network
w=load_database();

display('LOADED DATABASE SUCCESSFULLY !!!');

%% Initializations
% We randomly pick an image from our database and use the rest of the
% images for training. Training is done on 399 pictues. We later
% use the randomly selected picture to test the algorithm.

ri=round(400*rand(1,1));            % Randomly pick an index.
r=w(:,ri);                          % r contains the image we later on will use to test the algorithm
v=w(:,[1:ri-1 ri+1:end]);           % v contains the rest of the 399 images.
clear w;
j=ri/10;
N=40;                               % Number of signatures used for each image.
%% Subtracting the mean from v
O=uint8(ones(1,size(v,2))); 
m=uint8(mean(v,2));                 % m is the mean of all images.
vzm=v-uint8(double(m)*double(O));   % vzm is v with the mean removed. 

%% Calculating eignevectors of the correlation matrix
% We are picking N of the 400 eigenfaces.
L=double(vzm)'*double(vzm);
[V,D]=eig(L);
V=double(vzm)*V;
V=V(:,end:-1:end-(N-1));            % Pick the eignevectors corresponding to the 10 largest eigenvalues. 

display('CALCULATED EIGEN VALUES SUCCESSFULLY !!!');

%% Calculating the signature for each image
cv=zeros(size(v,2),N);
for i=1:size(v,2);
cv(i,:)=double(vzm(:,i))'*V;    % Each row in cv is the signature for one image.
end

display('CALCULATED EIGEN FACES SUCCESSFULLY !!!');

%% Recognition 
%  Now, we run the algorithm and see if we can correctly recognize the face. 
subplot(121); 
imshow(reshape(r,112,92));title('Looking for ...','FontWeight','bold','Fontsize',16,'color','red');

%% Creating Neural Network using the Neural Network Toolbox 

display('ENTERING NEURAL NETWORK...');

cv=cv';                             % Size of training matrix= 40*399
p=r-m;                              % Subtract the mean
Test=double(p)'*V;
Test=Test(:,end:-1:end-(N-1)); 

t = zeros(40,400);                  % Setting Target
for i= 1:1:40 
	for j= 1:1:10 
		t(i,10*(i-1)+j)=1;    
	end
end
t=t(:,[1:ri-1 ri+1:end]);

net=newff(minmax(cv), t, [N,70],{'tansig' 'logsig'},'trainrp'); 
net = init(net);



display('TARGET SET');

% Changing the default parameters

net.trainParam.show = 50;  
net.trainParam.epochs = 400; 
net.trainParam.goal = 1e-4;
net.trainParam.lr = 0.3;
net.trainParam.max_fail = 10;

% Training the Network

[net,tr]=train(net,cv,t);  

display('DONE TRAINING');

Test=Test';

display('ENTER TESTING');

a = sim(net, Test);            % Testing

display('DONE TESTING');

[C, I]=min(abs((ones(40,1)-a)));
fprintf('\n Looking for %d', j);
fprintf('\n Found Match %d \n',I);
if(I==j)
    display('EUREKA !!!');
end

subplot(122);
imshow(reshape(v(:,(ri+1)),112,92));title('Found!','FontWeight','bold','Fontsize',16,'color','red');



