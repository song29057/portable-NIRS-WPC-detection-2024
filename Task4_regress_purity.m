clear all;clc

%% load HSI
% X = load('D:\Postdoc\Paper 11\Datasets\HSI-adulterants.csv');
% X = X(1:55,:);  %% 麦芽
% X = transabs1(X);
% X = savgol(X,29,2,1);

% X = load('D:\Postdoc\Paper 11\Datasets\HSI-adulterants.csv');
% X = X(56:110,:);    %% 面粉
% X = transabs1(X);
% X = savgol(X,19,2,1);

% X = load('D:\Postdoc\Paper 11\Datasets\HSI-adulterants.csv');
% X = X(111:165,:);    %% 奶粉
% X = transabs1(X);
% X = savgol(X,33,2,1);
% X = msc(X,mean(X,1));

%% load NIR
% X = load('D:\Postdoc\Paper 11\Datasets\NIR-adulterants.csv');
% X = X(1:55,:);  %% 麦芽
% X = savgol(X,29,2,1);
% X = X(:,41:436);

X = load('D:\Postdoc\Paper 11\Datasets\NIR-adulterants.csv');
X = X(56:110,:);  %% 面粉
X = savgol(X,17,2,1);
X = X(:,41:436);
X = msc(X,mean(X,1));

% X = load('D:\Postdoc\Paper 11\Datasets\NIR-adulterants.csv');
% X = X(111:165,:);  %% 奶粉
% X = savgol(X,35,2,1);
% X = X(:,41:436);
% X = snv(X);

%% load Y
Y = 0:0.05:0.5;
Y = repmat(Y,5,1);Y = reshape(Y,55,1);
m = size(Y,1);

%% indices
times = 5;
index = 1 + rem(0:m-1,times); 
Yp = zeros(55,1);

% index = 1 + rem(0:10-1,times);
% index = repmat(index,5,1);
% index = reshape(index,50,1);
% index = [0;0;0;0;0;index];


% index = 1 + rem(0:11-1,times); 
% index = repmat(index,5,1);
% index = reshape(index,55,1);

for run = 1:times
    train = find(index~=run);
    test = find(index==run);
    Xtrain = X(train,:);Ytrain = Y(train,:);
    Xtest = X(test,:);Ytest = Y(test,:);
    fold = 5;
    indices = 1 + rem(0:size(Xtrain,1)-1,fold);         %% 1-12345

    %% PLSR
    % LV = 10;
    % cv = plscv(Xtrain,Ytrain,LV,fold,'autoscaling',0,2);  %% autoscaling center
    % optLV = cv.optLV;
    % Ycv = cv.Ypred(:,optLV);
    % model = pls(Xtrain,Ytrain,optLV,'autoscaling');
    % beta(run,:) = (model.regcoef_pretreat');
    % [yp,~] = plsval(model,Xtest,Ytest,optLV);
    % Yp(test,:) = yp;
    % [~,~,r2cv] = regress_results1(Ytrain,Ycv);
    % R2cv(run) = r2cv;
    % num_LV(run,:) = optLV;

    %% K-ELM
    C = 1:20;Gamma = -10:10;
    for i = 1:size(C,2)
        for j = 1:size(Gamma,2)
            for k = 1:fold
                test_ = (indices == k);  train_ = ~test_;
                xtrain = Xtrain(train_,:);    xtest = Xtrain(test_,:);
                ytrain = Ytrain(train_,:);    ytest = Ytrain(test_,:);        
                train_data = [ytrain xtrain];
                test_data = [ytest xtest];
                [yfit, ycv,~] = kelm(train_data,test_data,0,2^C(i),'RBF_kernel',2^Gamma(j));
                Ycv(test_,:) = ycv';
            end
            [rmsecv,r2cv] = regress_results(Ytrain,Ycv);
            R2cv_all(i,j) = r2cv;
            RMSEcv_all(i,j) = rmsecv;
        end
    end
    [a,b] = find(RMSEcv_all==min(min(RMSEcv_all)));a = a(end);b = b(end);
    train_data = [Ytrain Xtrain];test_data = [Ytest Xtest];
    [~,yp,OutputWeight] = kelm(train_data,test_data,0,2^C(a),'RBF_kernel',2^Gamma(b));
    Yp(test,:) = yp';
    num_C(run,:) = a;
    num_G(run,:) = b;

    %% run
    fprintf('%d run \n',run)
end
% CV = mean(R2cv,2)
[RMSEp,MAEp,R2p] = regress_results1(Y,Yp); %% (Y(6:end),Yp(6:end));
results = [RMSEp,MAEp,R2p];
results = roundn(results,-4)
% CV = roundn(mean(R2cv,2),-4)
% Beta = roundn(mean(beta,1)',-5);
% acc = [CV results(end)]
% num_LV = mean(num_LV);

%% Plot 2.5 2.2
purple_hue = 300; % 紫色在HSV空间的大致色调值
red_hue = 0;      % 红色在HSV空间的色调值
% 设置饱和度和亮度
saturation = 0.5;
value = 0.9; % 降低亮度使得颜色稍微浅一些
% 创建一个15x3的HSV颜色矩阵
num_colors = 15;
hsv_matrix = zeros(num_colors, 3); % 初始化HSV矩阵
% 计算色调值的范围
hue_range = linspace(purple_hue, red_hue, num_colors);
% 填充HSV矩阵
hsv_matrix(:,1) = hue_range / 360; % 将色调值转换为0-1范围
hsv_matrix(:,2) = saturation;
hsv_matrix(:,3) = value;
% 将HSV颜色矩阵转换为RGB颜色矩阵
C = hsv2rgb(hsv_matrix);

Y = reshape(Y,5,11);
Y = mean(Y,1);
Y = Y';

Yp = reshape(Yp,5,11);
Yp_std = std(Yp,1)';
Yp = mean(Yp,1)';
Yp = roundn(Yp,-3);

c1 = [0.0000  0.4470  0.7410];  %% blue
c2 = [0.8500  0.3250  0.0980];  %% red
c3 = [0.9290  0.6940  0.1250];  %% yellow
c4 = [0.4940  0.1840  0.5560];  %% purple
c5 = [0.4660  0.6740  0.1880];  %% green
c6 = [0.3010  0.7450  0.9330];  %% blue
c7 = [0.6350  0.0780  0.1840];  %% red

mi = min([Y;Yp]);mi = roundn(mi,-2);
ma = max([Y;Yp]);ma = roundn(ma,-2);
int = (ma - mi)/20;
x0 = mi - int;x1 = ma + int;
y0 = mi - int;y1 = ma + int;
figure
h4 = plot(0:0.05:0.5,0:0.05:0.5,'color',[0.7 0.7 0.7]);hold on
% h1 = scatter(Y,Yp,30,'o','MarkerEdgeColor',c2,'MarkerFaceColor',[1 1 1]);hold on;

e = errorbar(Y,Yp,Yp_std,'o');
e.MarkerSize = 2;
e.Color = C(14,:);  %%  C(5,:) C(10,:) C(14,:)
e.MarkerEdgeColor = [0 0 0];
e.MarkerFaceColor = [0 0 0];

% xlim([x0 x1]);ylim([y0 y1])
xlim([-0.035 0.535]);ylim([-0.035 0.535])
xlabel('True'); 
ylabel('Predicted'); 
a3 = ['R^2_P = ',num2str(roundn(R2p,-2))];
a2 = ['RMSE_P = ',num2str(roundn(RMSEp,-2))];
a1 = ['MAE_P = ',num2str(roundn(MAEp,-2))];
% t1 = text(0.02,0.47,a1); 
% t2 = text(0.02,0.42,a2); 
% t3 = text(0.02,0.37,a3); 
box on
yticks(0:0.1:0.5)
xticks(0:0.1:0.5)
% yticklabels({0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1})
% xticklabels({0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1})