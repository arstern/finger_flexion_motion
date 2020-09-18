function [predicted_dg] = make_predictions(test_ecog)
% INPUTS: test_ecog - 3 x 1 cell array containing ECoG for each subject, where test_ecog{i}
% to the ECoG for subject i. Each cell element contains a N x M testing ECoG,
% where N is the number of samples and M is the number of EEG channels.
% OUTPUTS: predicted_dg - 3 x 1 cell array, where predicted_dg{i} contains the
% data_glove prediction for subject i, which is an N x 5 matrix (for
% fingers 1:5)
% Run time: The script has to run less than 1 hour.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove bad channels
test_ecog{1}(:, 55) = [];
test_ecog{2}(:, 21) = [];
test_ecog{2}(:, 38) = [];
% Subtract average value
test_ecog{1} = test_ecog{1} - mean2(test_ecog{1});
test_ecog{2} = test_ecog{2} - mean2(test_ecog{2});
test_ecog{3} = test_ecog{3} - mean2(test_ecog{3});
% Set parameters
fs = 1e3; %Hz
window_length = 0.1; %s
window_overlap = 0.05; %s
% Load in Trained Model
load('s1_f_model1.mat')
load('s2_f_model1.mat')
load('s3_f_model1.mat')
% Get windowed features
s1_featMat_leaderboard = getWindowedFeats(test_ecog{1}, fs, window_length, window_overlap);
s2_featMat_leaderboard = getWindowedFeats(test_ecog{2}, fs, window_length, window_overlap);
s3_featMat_leaderboard = getWindowedFeats(test_ecog{3}, fs, window_length, window_overlap);
% Calculate R matrix
N = 3;
s1_R_leaderboard = create_R_matrix(s1_featMat_leaderboard, N);
s2_R_leaderboard = create_R_matrix(s2_featMat_leaderboard, N);
s3_R_leaderboard = create_R_matrix(s3_featMat_leaderboard, N);
% Make predictions
s1_yhat_leaderboard = s1_R_leaderboard*s1_f_model1;
s2_yhat_leaderboard = s2_R_leaderboard*s2_f_model1;
s3_yhat_leaderboard = s3_R_leaderboard*s3_f_model1;
% Zero Clamp
s1_yhat_leaderboard(s1_yhat_leaderboard < 0) = 0;
s2_yhat_leaderboard(s2_yhat_leaderboard < 0) = 0;
s3_yhat_leaderboard(s3_yhat_leaderboard < 0) = 0;
% Smooth
s1_yhat_leaderboard = movmean(s1_yhat_leaderboard, 7);
s2_yhat_leaderboard = movmean(s2_yhat_leaderboard, 7);
s3_yhat_leaderboard = movmean(s3_yhat_leaderboard, 7);
% Spline interpolation
x = linspace(0, 147500, length(s1_yhat_leaderboard))';
xx = 0:147500-1;
s1_yhat_leaderboard_interp = zeros(147500, 5);
s2_yhat_leaderboard_interp = zeros(147500, 5);
s3_yhat_leaderboard_interp = zeros(147500, 5);
for i = 1:5
 s1_yhat_leaderboard_interp(:,i) = spline(x,s1_yhat_leaderboard(:,i), xx);
 s2_yhat_leaderboard_interp(:,i) = spline(x,s2_yhat_leaderboard(:,i), xx);
 s3_yhat_leaderboard_interp(:,i) = spline(x,s3_yhat_leaderboard(:,i), xx);
end
% Assign interpolated predictions to predicted_dg
predicted_dg{1} = s1_yhat_leaderboard_interp;
predicted_dg{2} = s2_yhat_leaderboard_interp;
predicted_dg{3} = s3_yhat_leaderboard_interp;
predicted_dg = predicted_dg.';
save('predicted_dg.mat', 'predicted_dg')
end
