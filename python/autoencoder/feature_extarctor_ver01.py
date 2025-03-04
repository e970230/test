import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.stats import kurtosis, skew


def daqfft(y, fs, blocksize):
   
    Y = np.fft.fft(y, blocksize) / blocksize
    Y = (np.abs(Y)[:blocksize // 2 + 1]) * 2
    Y[1:-1] = Y[1:-1] *2
    f = np.arange(len(Y)) * fs / blocksize
    return f, Y

def extract_vibration_features(input_origin_data, spindle_fs, spindle_fs_target, BD_delta_F, BD_spindle_start,
                                BD_spindle_end, fs, tolerance):

    
    
    spendle_fs_list = spindle_fs_target * spindle_fs
    BD_start_fs = BD_spindle_start * spindle_fs
    BD_end_fs = BD_spindle_end * spindle_fs
    BD_list = np.arange(BD_start_fs, BD_end_fs, BD_delta_F)[:-1]

    features = []
    for signal_data in input_origin_data:
        ch_features = []
        for ch in range(3):
            if signal_data.shape[1] <= ch:
                continue

            fs_signal = signal_data[:, ch]
            rms_val = np.sqrt(np.mean(fs_signal**2))
            kurt_val = kurtosis(fs_signal, fisher=False, bias=False)
            skew_val = skew(fs_signal)
            std_val = np.std(fs_signal)
            crest_val = np.max(np.abs(fs_signal)) / (rms_val if rms_val != 0 else 1e-9)

            time_value = [rms_val, kurt_val, skew_val, std_val, crest_val]

            window = signal.get_window('hann', len(fs_signal))
            sig_win = fs_signal * window
            f_axis, amp_spectrum = daqfft(sig_win, fs, len(sig_win))

            fft_value = []
            for freq in spendle_fs_list:
                idx = np.where((f_axis > (freq - tolerance)) & (f_axis < (freq + tolerance)))[0]
                fft_value.append(np.max(amp_spectrum[idx]) if len(idx) > 0 else 0)

            BD_feature = []
            for band in BD_list:
                idx_start = np.where((f_axis > (band - tolerance)) & (f_axis < (band + tolerance)))[0]
                start_peak = np.max(amp_spectrum[idx_start]) if len(idx_start) > 0 else 0
                idx_end = np.where((f_axis > (band + BD_delta_F - tolerance)) &
                                   (f_axis < (band + BD_delta_F + tolerance)))[0]
                end_peak = np.max(amp_spectrum[idx_end]) if len(idx_end) > 0 else 0
                s_idx, e_idx = min(idx_start[0], idx_end[0]), max(idx_start[0], idx_end[0])
                BD_feature.append(np.sum(amp_spectrum[s_idx:e_idx + 1]))

            BD_feature = np.array(BD_feature)
            BD_per = BD_feature / (np.sum(BD_feature) if np.sum(BD_feature) != 0 else 1e-9)

            extract_value = np.hstack([time_value, fft_value, BD_feature, BD_per])
            ch_features.append(extract_value)

        if len(ch_features) == 3:
            features.append(np.concatenate(ch_features))

    return np.array(features)

def extract_current_features(input_origin_data, spindle_fs, current_data_spindle_fs_target, BD_delta_F,
                              current_spindle_start, current_spindle_end, fs_servo, tolerance):

    current_spindle_fs_list = current_data_spindle_fs_target * spindle_fs
    current_BD_start_fs = current_spindle_start * spindle_fs
    current_BD_end_fs = current_spindle_end * spindle_fs
    current_BD_list = np.arange(current_BD_start_fs, current_BD_end_fs, BD_delta_F)[:-1]

    features = []
    for signal_data in input_origin_data:
        fs_signal = signal_data[:,0]

        rms_val = np.sqrt(np.mean(fs_signal**2))
        kurt_val = kurtosis(np.array(fs_signal), fisher=False, bias=False)
        skew_val = skew(fs_signal)
        peak2peak_val = np.ptp(fs_signal)
        std_val = np.std(fs_signal)
        time_values = [rms_val, kurt_val, skew_val, peak2peak_val, std_val]

        window = signal.get_window('hann', len(fs_signal))
        sig_win = fs_signal * window
        f_axis, amp_spectrum = daqfft(sig_win, fs_servo, len(sig_win))

        ext_values = []
        for freq in current_spindle_fs_list:
            idx = np.where((f_axis > (freq - tolerance)) & (f_axis < (freq + tolerance)))[0]
            ext_values.append(np.max(amp_spectrum[idx]) if len(idx) > 0 else 0)

        BD_feature = []
        for band in current_BD_list:
            idx_start = np.where((f_axis > (band - tolerance)) & (f_axis < (band + tolerance)))[0]
            start_peak = np.max(amp_spectrum[idx_start]) if len(idx_start) > 0 else 0
            idx_end = np.where((f_axis > (band + BD_delta_F - tolerance)) &
                               (f_axis < (band + BD_delta_F + tolerance)))[0]
            end_peak = np.max(amp_spectrum[idx_end]) if len(idx_end) > 0 else 0
            s_idx, e_idx = min(idx_start[0], idx_end[0]), max(idx_start[0], idx_end[0])
            BD_feature.append(np.sum(amp_spectrum[s_idx:e_idx + 1]))

        BD_feature = np.array(BD_feature)
        BD_per = BD_feature / (np.sum(BD_feature) if np.sum(BD_feature) != 0 else 1e-9)

        features.append(np.hstack([time_values, ext_values, BD_feature, BD_per]))

    return np.array(features)

def vibration_mix_current_features(spindle_fs, spindle_fs_target, current_data_spindle_fs_target,
                      BD_delta_F, BD_spindle_start, BD_spindle_end, current_spindle_start,
                      current_spindle_end, tolerance, fs, fs_servo, ALL_data_nidaq,
                      ALL_servo_current):

    Final_input_data = []

    vibration_features = extract_vibration_features(
        ALL_data_nidaq, spindle_fs, spindle_fs_target, BD_delta_F,
        BD_spindle_start, BD_spindle_end, fs, tolerance
    )

    current_features = extract_current_features(
        ALL_servo_current, spindle_fs, current_data_spindle_fs_target, BD_delta_F,
        current_spindle_start, current_spindle_end, fs_servo, tolerance
    )

    if vibration_features.size > 0 and current_features.size > 0:
        Final_input_data.append(
            np.hstack([vibration_features, current_features])
        )

    return np.vstack(Final_input_data) if Final_input_data else np.array([])
