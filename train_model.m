%% Extract dataglove and ECoG data 
% Dataglove should be (samples x 5) array 
% ECoG should be (samples x channels) array

clear;clc;close all

load('raw_training_data.mat')

% Remove bad channels
train_ecog{1}(:, 55) = [];
train_ecog{2}(:, 21) = [];
train_ecog{2}(:, 38) = [];

% Load training ecog from each of three patients
s1_train_ecog = train_ecog{1};
s2_train_ecog = train_ecog{2};
s3_train_ecog = train_ecog{3};

% Load training dataglove finger flexion values for each of three patients
s1_train_dg = train_dg{1};
s2_train_dg = train_dg{2};
s3_train_dg = train_dg{3};

%% Pre-processing pipeline

% Subtract average value
s1_train_ecog = s1_train_ecog-mean2(s1_train_ecog);
s2_train_ecog = s2_train_ecog-mean2(s2_train_ecog);
s3_train_ecog = s3_train_ecog-mean2(s3_train_ecog);


%% Split Data

% Split data into a train and test set (use at least 50% for training)

s1_temptrain_ecog = train_ecog{1}(1:240000,:);
s1_temptest_ecog = train_ecog{1}(240001:end,:);
s1_temptrain_dg= train_dg{1}(1:240000,:);
s1_temptest_dg= train_dg{1}(240001:end,:);

s2_temptrain_ecog = train_ecog{2}(1:240000,:);
s2_temptest_ecog = train_ecog{2}(240001:end,:);
s2_temptrain_dg= train_dg{2}(1:240000,:);
s2_temptest_dg= train_dg{2}(240001:end,:);

s3_temptrain_ecog = train_ecog{3}(1:240000,:);
s3_temptest_ecog = train_ecog{3}(240001:end,:);
s3_temptrain_dg= train_dg{3}(1:240000,:);
s3_temptest_dg= train_dg{3}(240001:end,:);


%% Get Features

% Set parameters
fs = 1e3;               %Hz
window_length = 0.1;    %s
window_overlap = 0.05;  %s

% Run getWindowedFeats function
s1_featMat_train = getWindowedFeats(s1_temptrain_ecog, fs, window_length, window_overlap);
s1_featMat_test = getWindowedFeats(s1_temptest_ecog, fs, window_length, window_overlap);

s2_featMat_train = getWindowedFeats(s2_temptrain_ecog, fs, window_length, window_overlap);
s2_featMat_test = getWindowedFeats(s2_temptest_ecog, fs, window_length, window_overlap);

s3_featMat_train = getWindowedFeats(s3_temptrain_ecog, fs, window_length, window_overlap);
s3_featMat_test = getWindowedFeats(s3_temptest_ecog, fs, window_length, window_overlap);

%% Create R matrix

% Set N value (N-1 --> how many prior windows to include in R)
N = 3;

% Run create_R_matrix for all 3 subjects, test and train
s1_R_train = create_R_matrix(s1_featMat_train, N);
s1_R_test = create_R_matrix(s1_featMat_test, N);

s2_R_train = create_R_matrix(s2_featMat_train, N);
s2_R_test = create_R_matrix(s2_featMat_test, N);

s3_R_train = create_R_matrix(s3_featMat_train, N);
s3_R_test = create_R_matrix(s3_featMat_test, N);

%% Downsample


% Classifier 1: Get angle predictions using optimal linear decoding. That is, 
% calculate the linear filter (i.e. the weights matrix) as defined by 
% Equation 1 for all 5 finger angles.

% Downsample dg data so it can be predicted upon
N = size(s1_temptrain_ecog,1)/(size(s1_featMat_train,1)+1);

s1_temptrain_dg = downsample(s1_temptrain_dg,N);
s2_temptrain_dg = downsample(s2_temptrain_dg,N);
s3_temptrain_dg = downsample(s3_temptrain_dg,N);

%% Linear Model Prediction

% Compute weights matrices for each subject

s1_f_model1 = mldivide(s1_R_train'*s1_R_train,s1_R_train'*s1_temptrain_dg);
s2_f_model1 = mldivide(s2_R_train'*s2_R_train,s2_R_train'*s2_temptrain_dg);
s3_f_model1 = mldivide(s3_R_train'*s3_R_train,s3_R_train'*s3_temptrain_dg);

save('s1_f_model1.mat','s1_f_model1')
save('s2_f_model1.mat','s2_f_model1')
save('s3_f_model1.mat','s3_f_model1')

% Predict on model 1

s1_yhat_model1_intermed = s1_R_test*s1_f_model1;
s2_yhat_model1_intermed = s2_R_test*s2_f_model1;
s3_yhat_model1_intermed = s3_R_test*s3_f_model1;


%% Post Processing

s1_yhat_model1_intermed(s1_yhat_model1_intermed < 0) = 0;
s2_yhat_model1_intermed(s2_yhat_model1_intermed < 0) = 0;
s3_yhat_model1_intermed(s3_yhat_model1_intermed < 0) = 0;

s1_yhat_model1  = movmean(s1_yhat_model1_intermed, 7);
s2_yhat_model1  = movmean(s2_yhat_model1_intermed, 7);
s3_yhat_model1  = movmean(s3_yhat_model1_intermed, 7);


%% Correlate data Spline
x = linspace(0, length(s1_temptest_dg), length(s1_temptest_dg)/50)';
xx = 0:length(s1_temptest_dg)-1;

for i = 1:5
    s1_yhat_model1_interp(:,i) = spline(x,s1_yhat_model1(:,i), xx);
    s2_yhat_model1_interp(:,i) = spline(x,s2_yhat_model1(:,i), xx);
    s3_yhat_model1_interp(:,i) = spline(x,s3_yhat_model1(:,i), xx);
end


s1_yhat_model1_interp = movmean(s1_yhat_model1_interp, 1000);
s1_yhat_model1_interp = movmean(s1_yhat_model1_interp, 1000);
s1_yhat_model1_interp = movmean(s1_yhat_model1_interp, 1000);

corr_model1 = zeros(3,4);
for jj = 1:size(s1_yhat_model1,2)
    if jj == 4
        continue
    end
    corr_model1(1,jj) = corr(s1_yhat_model1_interp(:,jj),s1_temptest_dg(:,jj));
    corr_model1(2,jj) = corr(s2_yhat_model1_interp(:,jj),s2_temptest_dg(:,jj));
    corr_model1(3,jj) = corr(s3_yhat_model1_interp(:,jj),s3_temptest_dg(:,jj));
end

% Delete ring finger column
corr_model1(:,4) = [];
mean(corr_model1,1)
