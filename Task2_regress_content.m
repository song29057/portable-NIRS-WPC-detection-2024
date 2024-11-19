clear all;clc

%% load Y
Y0 = load('D:\Postdoc\Paper 11\Datasets\Y-brands.csv'); %% 1: Protein, 2: Carbohydrates

%% load NIRS
% X = load('D:\Postdoc\Paper 11\Datasets\NIR-brands.csv');
% i = 1;Y = Y0(:,i);
% X = savgol(X,25,2,1);
% X = X(:,41:436);
% X = snv(X);

% X = load('D:\Postdoc\Paper 11\Datasets\NIR-brands.csv');
% i = 2;Y = Y0(:,i);
% X = savgol(X,35,2,2);
% X = X(:,41:436);
% X = snv(X);

%% load HSI
% X = load('D:\Postdoc\Paper 11\Datasets\HSI-brands.csv');
% X = transabs(X);
% i = 1;Y = Y0(:,i);
% X = savgol(X,33,2,1);

X = load('D:\Postdoc\Paper 11\Datasets\HSI-brands.csv');
X = transabs(X);
i = 2;Y = Y0(:,i);
X = savgol(X,35,2,1);

%% indices
times = 5;
index = 1 + rem(0:15-1,times); 
index = repmat(index,12,1);
index = reshape(index,180,1);

Yp = zeros(180,1);

for run = 1:times
    train = find(index~=run);
    test = find(index==run);
    Xtrain = X(train,:);Ytrain = Y(train,:);
    Xtest = X(test,:);Ytest = Y(test,:);
    fold = 5;
    indices = 1 + rem(0:size(Xtrain,1)-1,fold);         %% 1-12345
    % indices = crossvalind('Kfold',ones(size(Xtrain,1),1),fold); 

    %% PLSR
    LV = 10;
    cv = plscv(Xtrain,Ytrain,LV,fold,'autoscaling',0,2);  %% autoscaling center
    optLV = cv.optLV;
    Ycv = cv.Ypred(:,optLV);
    model = pls(Xtrain,Ytrain,optLV,'autoscaling');
    beta(run,:) = (model.regcoef_pretreat');
    [yp,~] = plsval(model,Xtest,Ytest,optLV);
    Yp(test,:) = yp;
    [~,~,r2cv] = regress_results1(Ytrain,Ycv);
    R2cv(run) = r2cv;
    num_LV(run,:) = optLV;

    %% K-ELM
    % C = 1:20;Gamma = 1:20;
    % for i = 1:size(C,2)
    %     for j = 1:size(Gamma,2)
    %         for k = 1:fold
    %             test_ = (indices == k);  train_ = ~test_;
    %             xtrain = Xtrain(train_,:);    xtest = Xtrain(test_,:);
    %             ytrain = Ytrain(train_,:);    ytest = Ytrain(test_,:);        
    %             train_data = [ytrain xtrain];
    %             test_data = [ytest xtest];
    %             [yfit, ycv,~] = kelm(train_data,test_data,0,2^C(i),'lin_kernel',[2^Gamma(j),1]);
    %             Ycv(test_,:) = ycv';
    %         end
    %         [rmsecv,r2cv] = regress_results(Ytrain,Ycv);
    %         R2cv_all(i,j) = r2cv;
    %         RMSEcv_all(i,j) = rmsecv;
    %     end
    % end
    % [a,b] = find(RMSEcv_all==min(min(RMSEcv_all)));a = a(end);b = b(end);
    % train_data = [Ytrain Xtrain];test_data = [Ytest Xtest];
    % [~,yp,OutputWeight] = kelm(train_data,test_data,0,2^C(a),'lin_kernel',[2^Gamma(b),1]);
    % % 'RBF_kernel' for RBF Kernel
    % % 'lin_kernel' for Linear Kernel
    % % 'poly_kernel' for Polynomial Kernel
    % Yp(test,:) = yp';
    % num_C(run,:) = a;
    % num_G(run,:) = b;

    %% run
    fprintf('%d run \n',run)

end

[RMSEp,MAEp,R2p] = regress_results1(Y,Yp);
results = [RMSEp,MAEp,R2p];
results = roundn(results,-4)
% Beta = roundn(mean(beta,1)',-5);
% num_LV = mean(num_LV);

%% Plot 2.5:2.2
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

Y = reshape(Y,12,15);
Y = mean(Y,1);
Y = Y';

Yp = reshape(Yp,12,15);
Yp_std = std(Yp,1)';
Yp = mean(Yp,1)';
Yp = roundn(Yp,-3);
Tab = [Y Yp Yp_std];Tab = roundn(Tab,-3);

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
h4 = plot(0:0.05:1,0:0.05:1,'color',[0.8 0.8 0.8]);hold on
h1 = scatter(Y,Yp,15,'o','MarkerEdgeColor',c1);hold on;  %% c1 c2

% e = errorbar(Y,Yp,Yp_std,'o');
% e.MarkerSize = 2;
% e.Color = C(14,:);     %%  C(5,:) C(14,:)
% e.MarkerEdgeColor = [0 0 0];
% e.MarkerFaceColor = [0 0 0];


xlabel('True'); 
ylabel('Predicted'); 
a3 = ['R^2_P = ',num2str(roundn(R2p,-2))];
a2 = ['RMSE_P = ',num2str(roundn(RMSEp,-2))];
a1 = ['MAE_P = ',num2str(roundn(MAEp,-2))];
% t1 = text(0.05,0.85,a1); 
% t2 = text(0.05,0.85,a2); 
% t3 = text(0.05,0.7,a3); 
box on

xlim([0.1 0.9]);ylim([0.1 0.9])
yticks(0.1:0.2:0.9)
xticks(0.1:0.2:0.9)

% xlim([0 0.8]);ylim([0 0.8])
% yticks(0:0.2:0.8)
% xticks(0:0.2:0.8)

% yticklabels({0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1})
% xticklabels({0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1})