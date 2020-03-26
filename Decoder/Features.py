import torch
from pywt import wavedec
import numpy as np
import Parameter


def calc_features(window, window_acc, window_mag, window_gyro, feature_names, feature_names_im, sample_frequency):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features = torch.Tensor().to(device)

    # pre-calculated factors for some features
    P = torch.sum(torch.pow(torch.rfft(window.view(window.shape[1], -1), 1, onesided=False), 2.), 2).double().to(device)
    f = (torch.arange(window.shape[0], dtype=torch.double) / (window.shape[0] - 1.) * sample_frequency).double().to(device)
    w1 = torch.ones((window.shape[0], window.shape[1]), dtype=torch.double).to(device)
    w1[:round(window.shape[0] / 4.), :] = 0.5
    w1[round(window.shape[0] * 0.75):, :] = 0.5

    for feature_name in feature_names:
        features = torch.cat((features, calc_feature(window, feature_name, P, f, w1, device).float()), 1).to(device)

    for feature_name_im in feature_names_im:
        if Parameter.acc is True:
            features = torch.cat((features, calc_feature(window_acc, feature_name_im, P, f, w1, device).float()), 1).to(device)
        if Parameter.mag is True:
            features = torch.cat((features, calc_feature(window_mag, feature_name_im, P, f, w1, device).float()), 1).to(device)
        if Parameter.gyro is True:
            features = torch.cat((features, calc_feature(window_gyro, feature_name_im, P, f, w1, device).float()), 1).to(device)

    features.unsqueeze_(0)  # add additional dimension to concat

    return features


def calc_feature(window, feature_name, P=0, f=0, w1=0, device=0):

    if feature_name == 'mean_value':  # for acc data
        if Parameter.database == '8':
            feature0 = torch.mean(window, 0).reshape(16, 3)
        elif Parameter.database == 'Myo':
            feature0 = torch.mean(window, 0).reshape(1, 3).repeat(8, 1)
        else:
            feature0 = torch.mean(window, 0).reshape(12, 3)

    elif feature_name == 'root_mean_square':
        feature0 = torch.sqrt(torch.mean(torch.pow(window, 2.), 0))
    elif feature_name == 'wave_length':
        feature0 = torch.sum(torch.abs(window[1:, :]-window[0:-1, :]), 0)
    elif feature_name == 'histogram':
        feature0 = torch.Tensor().to(device)
        for i in range(window.shape[1]):
            feature0 = torch.cat((feature0, torch.histc(window[:, i], 13).view(1, -1).float()), 0)
    elif feature_name == 'mean_absolute_value':
        feature0 = torch.mean(torch.abs(window), 0)
    elif feature_name == 'mean_absolute_value_1':
        feature0 = torch.mean(w1 * torch.abs(window), 0)
    elif feature_name == 'temporal_moment_3':
        feature0 = torch.abs(torch.mean(torch.pow(window, 3.0), 0))
    elif feature_name == 'temporal_moment_4':
        feature0 = torch.abs(torch.mean(torch.pow(window, 4.0), 0))
    elif feature_name == 'temporal_moment_5':
        feature0 = torch.abs(torch.mean(torch.pow(window, 5.0), 0))
    elif feature_name == 'variance':
        feature0 = torch.sum(torch.pow(window, 2.), 0) / (window.shape[0] - 1.)
    elif feature_name == 'log_detector':
        feature0 = torch.exp(torch.mean(torch.log(torch.abs(window)), 0))
    elif feature_name == 'integrated_emg':
        feature0 = torch.mean(torch.abs(window), 0)
    elif feature_name == 'kurtosis':
        feature0 = torch.mean(torch.pow((window - torch.mean(window, 0) / torch.std(window)), 4.), 0) - 3.
    elif feature_name == 'kurtosis_custom':
        feature0 = torch.mean(torch.pow((window - torch.mean(window, 0) / torch.std(window)), 3.), 0) - 3.
    elif feature_name == 'kurtosis_custom2':
        feature0 = torch.mean(torch.pow((window / torch.std(window)), 4.), 0) - 3.
    elif feature_name == 'average_amplitude_change':
        feature0 = torch.mean(torch.abs(window[1:, :]-window[0:-1, :]), 0)
    elif feature_name == 'dasdv':
        feature0 = torch.sqrt(torch.mean(torch.pow(window[1:, :]-window[0:-1, :], 2.), 0))
    elif feature_name == 'simple_square_integral':
        feature0 = torch.sum(torch.pow(window, 2.), 0)
    elif feature_name == 'skewness':
        feature0 = torch.mean(torch.pow((window - torch.mean(window, 0) / torch.std(window)), 3.), 0)
    elif feature_name == 'skewness_custom':
        feature0 = torch.mean(torch.pow((window / torch.std(window)), 3.), 0)
    elif feature_name == 'total_power':  # spectral moment 0
        feature0 = torch.sum(P, 1)
    elif feature_name == 'spectral_moment_1':
        feature0 = torch.sum(P * f, 1)
    elif feature_name == 'spectral_moment_2':
        feature0 = torch.sum(P * torch.pow(f, 2.), 1)
    elif feature_name == 'spectral_moment_3':
        feature0 = torch.sum(P * torch.pow(f, 3.), 1)
    elif feature_name == 'mean_frequency':
        feature0 = torch.sum(P * f, 1) / torch.sum(P, 1)
    elif feature_name == 'median_frequency':
        feature0 = 0.5 * torch.sum(P, 1)
    elif feature_name == 'mean_power':
        feature0 = torch.mean(P, 1)
    elif feature_name == 'peak_frequency':
        feature0 = f[torch.argmax(P, dim=1)]
    elif feature_name == 'variance_of_central_frequency':
        feature0 = (torch.sum(P * torch.pow(f, 2.), 1) / torch.sum(P, 1)) - torch.pow(torch.sum(P * f, 1) / torch.sum(P, 1), 2.)
    elif feature_name == 'zero_crossings':
        crossings = torch.sign(window)[1:, :] - torch.sign(window)[:-1, :]
        feature0 = torch.Tensor().to(device)
        for k in range(window.shape[1]):
            feature0 = torch.cat((feature0, torch.tensor([crossings[:, k].nonzero().shape[0]]).cuda().float()), 0)
    elif feature_name == 'slope_sign_changes':
        diff = window[1:, :] - window[:-1, :]
        slopes = torch.sign(diff)[1:, :] - torch.sign(diff)[:-1, :]
        feature0 = torch.Tensor().to(device)
        for m in range(window.shape[1]):
            feature0 = torch.cat((feature0, torch.tensor([slopes[:, m].nonzero().shape[0]]).cuda().float()), 0)
    elif feature_name == 'DWT':
        feature0 = torch.Tensor().to(device)
        for j in range(window.shape[1]):
            input_channel = window[:, j].float().view(1, 1, 1, -1).cpu().numpy()
            coeff = wavedec(input_channel, 'db8', level=5, mode="symmetric")
            feat = list()
            for c in coeff:
                mav = np.mean(np.abs(c[0, 0, 0, :]))
                feat.append(mav)  # MAV
                feat.append(np.sqrt(np.mean(np.square(c[0, 0, 0, :] - mav))))
                #feat.append(-np.sum(np.square(c[0, 0, 0, :])*np.log(np.square(c[0, 0, 0, :]))))  # Entropy
                feat.append(np.sum(np.square(c[0, 0, 0, :])))  # Energy

            feature0 = torch.cat((feature0, torch.from_numpy(np.array(feat)).cuda().unsqueeze(0)), 0)
    else:
        feature0 = 0
        print('Invalid feature!')

    if feature_name != 'histogram' and feature_name != 'DWT' and feature_name != 'mean_value':
        feature0 = feature0.unsqueeze(1)  # resize

    return feature0
