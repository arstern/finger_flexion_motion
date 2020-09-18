function [features] = get_features(clean_data, fs)
%
%
% Input: clean_data: (samples x channels)
% fs: sampling frequency
%
% Output: features: (1 x (channels*features))
% (e.g. Ch1_F1, Ch1_F2 ... Ch2_F1, Ch2_F2 ...)
%
%% Get features
% Feature 1: Average Time Voltage
avg_time_voltage = mean(clean_data, 1);
features = avg_time_voltage;
end
