%---- fft¤lµ{¦¡ ---------------------------
function [f,mag] = daqfft(ach1,fs,blocksize)
%    [F,MAG]=DAQFFT(X,FS,BLOCKSIZE) calculates the FFT of X
%    using sampling frequency FS and the SamplesPerTrigger
%    provided in BLOCKSIZE

mag = abs(fft(hilbert(ach1)))/blocksize;   
%xfft = abs(fft(hilbert(data),Fs_4CH))/Fs_4CH;
% One-side Spectrum Real Magnitude
% Avoid taking the log of 0.
% index = find(mag == 0);
% mag(index) = 1e-17;
% mag = 20*log10(mag/2^0.5/20E-6);
mag = mag(1:fix(blocksize/2)+1);
f = (0:length(mag)-1)*fs/blocksize;
f = f(:);