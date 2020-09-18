function [all_feats] = getWindowedFeats(raw_data, fs, window_length, window_overlap)
%
% Inputs: raw_data: The raw data for all patients
% fs: The raw sampling frequency (Hz)
% window_length: The length of window (s)
% window_overlap: The overlap in window (s)
%
% Output: all_feats: All calculated features
% First, filter the raw data
clean_data = filter_data(raw_data);
% Compute number of windows required
NumWins = @(xLen, fs, winLen, winDisp) floor(((xLen - winLen*fs + winDisp*fs)/(winDisp*fs)));
nWins = NumWins(size(raw_data, 1), fs, window_length, window_overlap);
nFeats = 6; % Number of features being calculated per window per channel
windowStart = 1; % Then, loop through sliding windows
numChannels = size(clean_data, 2); % Number of channels in data
all_feats = zeros(nWins, size(raw_data, 2)*nFeats);
for jj = 1:nWins
 % Within loop calculate feature for each segment (call get_features)
 windowIndices = [windowStart, windowStart + window_length*fs];
 window = clean_data(windowIndices(1):windowIndices(2)-1, :);
 % Apply get_features to window
 features = get_features(window,fs);
 all_feats(jj, 1:numChannels) = features;
 % Update start point of the window for next iteration
 windowStart = windowStart + window_overlap*fs;
end
% Compute spectrogram over windows in 5 frequency bands
f = 1:200;
band1 = zeros(nWins, size(raw_data, 2));
band2 = zeros(nWins, size(raw_data, 2));
band3 = zeros(nWins, size(raw_data, 2));
band4 = zeros(nWins, size(raw_data, 2));
band5 = zeros(nWins, size(raw_data, 2));
for i = 1:numChannels
 [S,F,~] = spectrogram(clean_data(:, i), window_length*1E3, round((window_overlap)*1E3), f, fs);
 S_real = abs(real(S));
 band1(:, i) = mean(S_real(5:15, :))';
 band2(:, i) = mean(S_real(20:25, :))';
 band3(:, i) = mean(S_real(75:115, :))';
 band4(:, i) = mean(S_real(125:160, :))';
 band5(:, i) = mean(S_real(160:175, :))';
end
i = 1;
% Add frequency bands to feature matrix
all_feats(:, numChannels*i+1:numChannels*(i+1)) = band1;
i = i+1;
all_feats(:, numChannels*i+1:numChannels*(i+1)) = band2;
i = i+1;
all_feats(:, numChannels*i+1:numChannels*(i+1)) = band3;
i = i+1;
all_feats(:, numChannels*i+1:numChannels*(i+1)) = band4;
i = i+1;
all_feats(:, numChannels*i+1:numChannels*(i+1)) = band5;
% Return feature matrix
all_feats = normalize(all_feats);
end
