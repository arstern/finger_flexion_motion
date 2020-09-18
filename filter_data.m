function clean_data = filter_data(raw_eeg)
%
%
% Input: raw_eeg (samples x channels)
%
% Output: clean_data (samples x channels)
%
%
%% Filter Data
% 60hz
[b,a] = butter(4, [59 61]./(1000/2), 'stop');
% 120hz
[b2,a2] = butter(4, [119 121]./(1000/2), 'stop');
% bandpass
[bband,aband] = butter(4, [1 175]/(1000/2), 'bandpass');
% Apply filters
clean_data = filtfilt(b, a, raw_eeg);
clean_data = filtfilt(b2, a2, clean_data);
clean_data = filtfilt(bband, aband, clean_data);
end
