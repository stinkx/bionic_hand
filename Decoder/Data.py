import torch
import scipy.io as scio
from scipy import signal
from Features import calc_features
from sklearn.preprocessing import StandardScaler

import Parameter

import matplotlib.pyplot as plt
# input_size, mean, std


def process_data():
    time_frame = int(Parameter.window / (1000. / Parameter.emg_frequency))  # number of samples in window
    overlap = int(Parameter.overlap / (1000. / Parameter.emg_frequency))    # number of samples in overlap
    time_progress = time_frame - overlap
    emd = int(Parameter.emd / (1000. / Parameter.emg_frequency))            # number of samples in emd

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

    emg = torch.Tensor()
    glove = torch.Tensor()
    acc = torch.Tensor()
    mag = torch.Tensor()
    gyro = torch.Tensor()

    movement = torch.Tensor()
    repetition = torch.Tensor()

    # combine all datasets to one
    for dataset_name in Parameter.dataset:
        dataset = scio.loadmat(dataset_name)
        emg_data = dataset['emg']
        if Parameter.denoise is True:
            # 4th order Butterworth filter
            low = Parameter.low_cut_freq / (0.5 * Parameter.emg_frequency)
            high = Parameter.high_cut_freq / (0.5 * Parameter.emg_frequency)
            sos = signal.butter(4, [low, high], analog=False, btype='band', output='sos')
            emg_data = signal.sosfilt(sos, emg_data)
        if Parameter.notch is True:
            b, a = signal.iirnotch(50., 30., Parameter.emg_frequency)
            #freq, h = signal.freqz(b, a, fs=emg_frequency)
            emg_data = signal.lfilter(b, a, emg_data)
            #plt.plot(freq, h, color='blue')
            #plt.show()

        emg = torch.cat((emg, torch.from_numpy(emg_data).float()), 0)
        glove = torch.cat((glove, torch.from_numpy(dataset['glove']).float()), 0)
        if Parameter.acc is True:
            acc = torch.cat((acc, torch.from_numpy(dataset['acc']).float()), 0)
        if Parameter.mag is True:
            mag = torch.cat((mag, torch.from_numpy(dataset['mag']).float()), 0)
        if Parameter.gyro is True:
            gyro = torch.cat((gyro, torch.from_numpy(dataset['gyro']).float()), 0)

        if Parameter.split_dataset is True:
            movement = torch.cat((movement, torch.from_numpy(dataset['restimulus']).float()), 0)
            repetition = torch.cat((repetition, torch.from_numpy(dataset['rerepetition']).float()), 0)

    emg = emg[emd::, :].double().to(Parameter.device)
    glove = glove[time_frame::time_progress, :].to(Parameter.device)  # TODO: evaluate this
    #if Parameter.acc is True:
    #    acc = acc.to(Parameter.device)
    #if Parameter.im is True:
        #acc = acc.to(Parameter.device)
        #mag = mag.to(Parameter.device)
        #gyro = gyro.to(Parameter.device)

    if Parameter.split_dataset is True:
        movement = movement[time_frame::time_progress, :].to(Parameter.device)
        repetition = repetition[time_frame::time_progress, :].to(Parameter.device)

    #TODO: use repetition 5 for validation and 6 for testing

    if Parameter.normalize_in is True:  # mean 0 std 1
        #emg = emg / torch.max(emg)
        #emg = emg * 300.
        Parameter.parameter['mean_in'] = torch.mean(emg)
        Parameter.parameter['std_in'] = torch.std(emg)
        scaler.fit(emg.cpu())
        emg = torch.from_numpy(scaler.transform(emg.cpu())).float().to(Parameter.device)

    if Parameter.normalize_gt is True:
        #glove = glove + 100.
        #glove = (glove + 1000.) / 200.
        Parameter.parameter['mean_gt'] = torch.mean(glove)
        Parameter.parameter['std_gt'] = torch.std(glove)
        scaler.fit(glove.cpu())
        glove = torch.from_numpy(scaler.transform(glove.cpu())).float().to(Parameter.device)

    ####################################################################################################################
    # calculate feature set
    feature_set = torch.Tensor().to(Parameter.device)

    for i in range(int((emg.shape[0] - time_frame) / time_progress)):
        window = emg[i * time_progress:i * time_progress + time_frame:, :]
        if Parameter.acc is True:
            window_acc = acc[i * time_progress:i * time_progress + time_frame:, :].to(Parameter.device)
        else:
            window_acc = 0

        if Parameter.mag is True:
            window_mag = mag[i * time_progress:i * time_progress + time_frame:, :].to(Parameter.device)
        else:
            window_mag = 0

        if Parameter.gyro is True:
            window_gyro = gyro[i * time_progress:i * time_progress + time_frame:, :].to(Parameter.device)
        else:
            window_gyro = 0

        if Parameter.calc_feature is True:
            feature_set = torch.cat((feature_set, calc_features(window, window_acc, window_mag, window_gyro, Parameter.feature_set, Parameter.feature_set_im, Parameter.emg_frequency)), 0)  # TxCxF
        else:
            feature_set = torch.cat((feature_set, window.float().to(Parameter.device).transpose_(0, 1).unsqueeze(0)), 0)

        print('Generating Feature Set [ ', i + 1, ' / ', int((emg.shape[0] - time_frame) / time_progress), ' ]', flush=True, end="\r")

    emg.cpu()

    if Parameter.network != "CNN" and Parameter.network != "CRNN":  # if CNN leave as [timestep, channel, window length]
        feature_set = feature_set.view(feature_set.shape[0], -1)  # resize to [timestep, channel x feature]

        if Parameter.normalize_ft is True:
            Parameter.parameter['mean_ft'] = torch.mean(feature_set)
            Parameter.parameter['std_ft'] = torch.std(feature_set)
            scaler.fit(feature_set.cpu())
            feature_set = torch.from_numpy(scaler.transform(feature_set.cpu())).float().to(Parameter.device)
    else:
        if Parameter.normalize_ft is True:  # for CNN normalize for each channel
            Parameter.parameter['mean_ft'] = torch.mean(feature_set)
            Parameter.parameter['std_ft'] = torch.std(feature_set)
            for j in range(feature_set.shape[1]):
                scaler.fit(feature_set[:, j, :].cpu())
                feature_set[:, j, :] = torch.from_numpy(scaler.transform(feature_set[:, j, :].cpu())).float().to(Parameter.device)

    ####################################################################################################################
    # split dataset into training, validation and test set
    training_set = torch.Tensor().to(Parameter.device)
    validation_set = torch.Tensor().to(Parameter.device)
    testing_set = torch.Tensor().to(Parameter.device)

    training_set.input = torch.Tensor().to(Parameter.device)
    validation_set.input = torch.Tensor().to(Parameter.device)
    testing_set.input = torch.Tensor().to(Parameter.device)

    training_set.ground_truth = torch.Tensor().to(Parameter.device)
    validation_set.ground_truth = torch.Tensor().to(Parameter.device)
    testing_set.ground_truth = torch.Tensor().to(Parameter.device)

    if Parameter.split_dataset is True:
        for k in range(repetition.shape[0] - 1):
            if repetition[k, 0] == 5:
                validation_set.input = torch.cat((validation_set.input, feature_set[k].unsqueeze(0)), 0)
                validation_set.ground_truth = torch.cat((validation_set.ground_truth, glove[k].unsqueeze(0)), 0)
            elif repetition[k, 0] == 6:
                testing_set.input = torch.cat((testing_set.input, feature_set[k].unsqueeze(0)), 0)
                testing_set.ground_truth = torch.cat((testing_set.ground_truth, glove[k].unsqueeze(0)), 0)
            else:
                training_set.input = torch.cat((training_set.input, feature_set[k].unsqueeze(0)), 0)
                training_set.ground_truth = torch.cat((training_set.ground_truth, glove[k].unsqueeze(0)), 0)

        glove.cpu()

        training_set.input = training_set.input.unsqueeze(1)
        validation_set.input = validation_set.input.unsqueeze(1)
        testing_set.input = testing_set.input.unsqueeze(1)

        training_set.ground_truth = training_set.ground_truth.unsqueeze(1)
        validation_set.ground_truth = validation_set.ground_truth.unsqueeze(1)
        #testing_set.ground_truth = testing_set.ground_truth.unsqueeze(1)

    else:
        # calc indexes for dataset split
        border_idx = int(feature_set.shape[0] * Parameter.training_size)
        border_idx_valid = border_idx + int(feature_set.shape[0] * Parameter.validation_size)
        border_idx_test = border_idx_valid + int(feature_set.shape[0] * Parameter.testing_size)

        # split dataset
        training_set.ground_truth = glove[:border_idx:, :].to(Parameter.device)
        validation_set.ground_truth = glove[border_idx:border_idx_valid:, :].to(Parameter.device)
        testing_set.ground_truth = glove[border_idx_valid:border_idx_test:, :].to(Parameter.device)

        glove.cpu()

        if Parameter.network == "CNN" or Parameter.network == "CRNN":
            training_set.input = feature_set[:border_idx:, :, :].to(Parameter.device)
            validation_set.input = feature_set[border_idx:border_idx_valid:, :, :].to(Parameter.device)
            testing_set.input = feature_set[border_idx_valid:border_idx_test:, :, :].to(Parameter.device)
        else:
            training_set.input = feature_set[:border_idx:, :].to(Parameter.device)
            validation_set.input = feature_set[border_idx:border_idx_valid:, :].to(Parameter.device)
            testing_set.input = feature_set[border_idx_valid:border_idx_test:, :].to(Parameter.device)

        #if normalize_gt is True:  # TODO: also implement normalization as for input data
        #    training_set.ground_truth = torch.sigmoid(training_set.ground_truth / 170.).to(device)
        #    validation_set.ground_truth = torch.sigmoid(validation_set.ground_truth / 170.).to(device)
        #    testing_set.ground_truth = torch.sigmoid(testing_set.ground_truth / 170.).to(device)

        if Parameter.derivative_gt is True:  # validated
            training_set.ground_truth = training_set.ground_truth[1:, :] - training_set.ground_truth[:-1, :]
            validation_set.ground_truth = validation_set.ground_truth[1:, :] - validation_set.ground_truth[:-1, :]
            testing_set.ground_truth = testing_set.ground_truth[1:, :] - testing_set.ground_truth[:-1, :]

        # reshape into: steps x batch size x input/output size
        if Parameter.network == "CNN" or Parameter.network == "CRNN":
            training_set.input = training_set.input[:(training_set.input.shape[0] - (training_set.input.shape[0] % Parameter.batch_size)), :, :]  # prepare for reshape
            training_set.input = training_set.input.view(-1, Parameter.batch_size, training_set.input.shape[1], training_set.input.shape[2])  # reshape

            validation_set.input = validation_set.input[:(validation_set.input.shape[0] - (validation_set.input.shape[0] % Parameter.batch_size)), :, :]  # prepare for reshape
            validation_set.input = validation_set.input.view(-1, Parameter.batch_size, validation_set.input.shape[1], validation_set.input.shape[2])  # reshape

            testing_set.input = testing_set.input.view(-1, 1, testing_set.input.shape[1], testing_set.input.shape[2])  # reshape
        else:
            training_set.input = training_set.input[:(training_set.input.shape[0] - (training_set.input.shape[0] % Parameter.batch_size)), :]  # prepare for reshape
            training_set.input = training_set.input.view(-1, Parameter.batch_size, training_set.input.shape[1])  # reshape

            validation_set.input = validation_set.input[:(validation_set.input.shape[0] - (validation_set.input.shape[0] % Parameter.batch_size)), :]  # prepare for reshape
            validation_set.input = validation_set.input.view(-1, Parameter.batch_size, validation_set.input.shape[1])  # reshape

            testing_set.input = testing_set.input.view(-1, 1, testing_set.input.shape[1])  # reshape

        training_set.ground_truth = training_set.ground_truth[:(training_set.ground_truth.shape[0] - (training_set.ground_truth.shape[0] % Parameter.batch_size))]  # prepare for reshape
        training_set.ground_truth = training_set.ground_truth.view(-1, Parameter.batch_size, training_set.ground_truth.shape[1])  # reshape

        validation_set.ground_truth = validation_set.ground_truth[:(validation_set.ground_truth.shape[0] - (validation_set.ground_truth.shape[0] % Parameter.batch_size))]  # prepare for reshape
        validation_set.ground_truth = validation_set.ground_truth.view(-1, Parameter.batch_size, validation_set.ground_truth.shape[1])  # reshape

    print('Preprocessing finished.')

    return training_set, validation_set, testing_set
