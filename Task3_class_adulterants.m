clear all;clc

%% load NIR
X = load('D:\Postdoc\Paper 11\Datasets\NIR-adulterants.csv');
X = savgol(X,35,2,1);
X = X(:,41:436);
X = snv(X);

%% load HSI
% X = load('D:\Postdoc\Paper 11\Datasets\HSI-adulterants.csv');
% X = transabs1(X);
% X = savgol(X,29,2,1);

%% lable
outlier = [1:5,56:60,111:115];X(outlier,:) = [];
Y = 1:1:3;Y = repmat(Y,50,1);Y = reshape(Y,150,1);
m = size(Y,1);

%% repeated run
times = 5;
index = 1 + rem(0:m-1,times); 
Yp = zeros(150,1);
for run = 1:times
    train = find(index~=run);
    test = find(index==run);
    Xtrain = X(train,:);Ytrain = Y(train,:);
    Xtest = X(test,:);Ytest = Y(test,:);
    fold = 5;
    indices = 1 + rem(0:size(Xtrain,1)-1,fold);       
    % indices = crossvalind('Kfold',ones(size(Xtrain,1),1),fold); 

    %% PLS-DA
    LV = 10;
    for k = 1:fold
        test_ = (indices == k);  train_ = ~test_;
        xtrain = Xtrain(train_,:);    xtest = Xtrain(test_,:);
        ytrain = Ytrain(train_,:);    ytest = Ytrain(test_,:);
        [~,~,ycv] = fast_PLSDA(xtrain,ytrain,xtest,ytest,LV);
        Ycv(test_,:) = ycv;
    end
    for i = 1:LV
        correct = find(Ytrain - Ycv(:,i) == 0);
        acc = size(correct,1)/size(Ytrain,1);
        results_CV_all(i,:) = acc;
    end
    [results_CV,para] = max(results_CV_all);
    b = find(results_CV_all == results_CV);b = b(1); 
    [score,accuracy,yp] = fast_PLSDA(Xtrain,Ytrain,Xtest,Ytest,b);yp = yp(:,end);
    Yp(test,:) = yp;
    num_LV(run,:) = b;

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
    %             [~,~,~,~,ycv] = kernel_elm(train_data,test_data,1,2^C(i),'RBF_kernel',2^Gamma(j));
    %             [~,ycv] = max(ycv,[],1);
    %             Ycv(test_,:) = ycv';
    %         end
    %         [rmsecv,r2cv] = regress_results(Ytrain,Ycv);
    %         R2cv_all(i,j) = r2cv;
    %         RMSEcv_all(i,j) = rmsecv;
    %     end
    % end
    % [a,b] = find(RMSEcv_all==min(min(RMSEcv_all)));a = a(1);b = b(1);
    % train_data = [Ytrain Xtrain];test_data = [Ytest Xtest];
    % [~,~,~,~,yp] = kernel_elm(train_data,test_data,1,2^C(a),'RBF_kernel',2^Gamma(b));
    % [~,yp] = max(yp,[],1);yp = yp';
    % Yp(test,:) = yp;
    % num_C(run,:) = a;
    % num_G(run,:) = b;

    %% run
    fprintf('%d run \n',run)
end

correct = find(Y-Yp == 0);
accuracy_predict = size(correct,1)/size(Y,1);
results = roundn(accuracy_predict,-4)'

%% PCA  3.5:2.5
figure
% score = tsne(X,'NumDimensions',2);%,'Exaggeration',3,'LearnRate',1000);
[~,score,~,~,explained,~] = pca(zscore(X));
e1 = roundn(explained(1),-1);
e2 = roundn(explained(2),-1);
e3 = roundn(explained(3),-1);
e4 = roundn(explained(4),-1);

purple_hue = 300; % 紫色在HSV空间的大致色调值
red_hue = 0;      % 红色在HSV空间的色调值

% 设置饱和度和亮度
saturation = 0.6;
value = 0.9; % 降低亮度使得颜色稍微浅一些

% 创建一个15x3的HSV颜色矩阵
num_colors = 5;
hsv_matrix = zeros(num_colors, 3); % 初始化HSV矩阵

% 计算色调值的范围
hue_range = linspace(purple_hue, red_hue, num_colors);

% 填充HSV矩阵
hsv_matrix(:,1) = hue_range / 360; % 将色调值转换为0-1范围
hsv_matrix(:,2) = saturation;
hsv_matrix(:,3) = value;

% 将HSV颜色矩阵转换为RGB颜色矩阵
C = hsv2rgb(hsv_matrix);

plot(-100:1:100,zeros(201,1),'--','color','k');hold on
plot(zeros(201,1),-100:1:100,'--','color','k');hold on

for i = 1:1:3
    confellipse2([score(Y==i,1),score(Y==i,2)],0.95);hold on
end

h2 = scatter((score(Y==2,1)),(score(Y==2,2)),15,'o','MarkerEdgeColor',C(3,:),'MarkerFaceColor',[1 1 1]);hold on   
h3 = scatter((score(Y==3,1)),(score(Y==3,2)),15,'^','MarkerEdgeColor',C(2,:),'MarkerFaceColor',[1 1 1]);hold on  
h1 = scatter((score(Y==1,1)),(score(Y==1,2)),20,C(1,:),'+');hold on 

% for i = 1:1:3
%     text(mean((score(Y==i,1))),mean((score(Y==i,2))),{i},'color','k','fontsize',8,'fontweight','bold');hold on
% end

box on
xlabel(['PC 1 (',num2str(e1),'%)'])
ylabel(['PC 2 (',num2str(e2),'%)'])

f = legend([h1,h2,h3],'Maltodextrin','Wheat flour','Milk powder','location','northwest');
f.ItemTokenSize = [9,12];
set(f,'Box','off');

xlim([-40 40])
ylim([-20 30])

% xlim([-80 40])
% ylim([-25 40])