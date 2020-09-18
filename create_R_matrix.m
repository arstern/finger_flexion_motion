function [R] = create_R_matrix(features, N_wind)
%
%
% Input: features: (samples x (channels*features))
% N_wind: Number of windows to use
%
% Output: R: (samples x (N_wind*channels*features))
%
%
%% Create R Matrix
% Initiate R matrix
R = ones(size(features, 1), N_wind*size(features, 2) + 1);
% Populate first N_wind-1 rows of R matrix
for jj = 1:N_wind - 1
 R(jj, 2:end) = repmat(features(jj, :),[1 N_wind]);
end
% Populate the remaining rows of R matrix
for jj = N_wind:size(features, 1)
 temp = features(jj - (N_wind - 1):jj, :);
 R(jj, 2:end) = temp(:)';
end
% Replicate last row and append to bottom to conserve matrix dimensions
R(end + 1, :) = R(end, :);
end
