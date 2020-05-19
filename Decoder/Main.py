# -*- coding: utf-8 -*-
import torch
import os
from statistics import mean
import torch.nn as nn
from Model import get_model
from Optimizer import get_optimizer
from Loss import get_Loss
from Data import process_data
from Initializer import weights_init
import Parameter
from tensorboardX import SummaryWriter
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import argparse

########################################################################################################################
# Parse arguments
if Parameter.parse_args is True:
    parser = argparse.ArgumentParser(description='Dataset and subject for parameter study')
    parser.add_argument('database', type=str, help='Ninapro database. Valid arguments [1, 2, 7, 8, Myo]')
    parser.add_argument('subject', type=int, help='Subject in dataset')
    parser.add_argument('mode', type=str,
                        help='Extract features, train model, test or do all. Valid arguments [feature, train, test, all]')
    args = parser.parse_args()

    if args.database not in ["1", "2", "7", "8", "Myo", 'cross-subject8', 'cross-subject2', 'cross-subject7']:
        raise ValueError("Dataset not supported. Choose one of the following [1, 2, 7, 8, Myo]")
    else:
        Parameter.database = args.database

    Parameter.subject = args.subject

    if Parameter.database == '1':
        Parameter.emg_frequency = 100.
        Parameter.acc = False
        Parameter.mag = False
        Parameter.gyro = False
        Parameter.dataset = [
            '../Ninapro/Dataset_1/s' + str(Parameter.subject) + '/S' + str(Parameter.subject) + '_A1_E1.mat',
            '../Ninapro/Dataset_1/s' + str(Parameter.subject) + '/S' + str(Parameter.subject) + '_A1_E3.mat',
            '../Ninapro/Dataset_1/s' + str(Parameter.subject) + '/S' + str(Parameter.subject) + '_A1_E2.mat']
    elif Parameter.database == '2':
        Parameter.emg_frequency = 2000.
        Parameter.mag = False
        Parameter.gyro = False
        Parameter.dataset = [
            '../Ninapro/Dataset_2/DB2_s' + str(Parameter.subject) + '/S' + str(Parameter.subject) + '_E2_A1.mat',
            '../Ninapro/Dataset_2/DB2_s' + str(Parameter.subject) + '/S' + str(Parameter.subject) + '_E1_A1.mat']
    elif Parameter.database == '7':
        Parameter.emg_frequency = 2000.
        Parameter.dataset = [
            '../Ninapro/Dataset_7/Subject_' + str(Parameter.subject) + '/S' + str(Parameter.subject) + '_E2_A1.mat',
            '../Ninapro/Dataset_7/Subject_' + str(Parameter.subject) + '/S' + str(Parameter.subject) + '_E1_A1.mat']
    elif Parameter.database == '8':
        Parameter.emg_frequency = 1111.
        Parameter.dataset = ['../Ninapro/Dataset_8/S' + str(Parameter.subject) + '_E1_A1.mat',
                             '../Ninapro/Dataset_8/S' + str(Parameter.subject) + '_E1_A2.mat',
                             '../Ninapro/Dataset_8/S' + str(Parameter.subject) + '_E1_A3.mat']
    elif Parameter.database == 'Myo':
        Parameter.emg_frequency = 50.
        Parameter.mag = False
        Parameter.dataset = ['../Ninapro/Myo/S' + str(Parameter.subject) + '_E1.mat',
                             '../Ninapro/Myo/S' + str(Parameter.subject) + '_E2.mat']
    elif Parameter.database == 'cross-subject8':
        Parameter.emg_frequency = 1111.
    #     Parameter.dataset = []
    #     for s in range(4):
    #         for a in range(3):
    #             Parameter.dataset.append('../Ninapro/Dataset_8/S' + str(s+1) + '_E1_A' + str(a+1) + '.mat')
    else:
        print('Invalid database!')

    if args.mode == "feature":
        Parameter.load_input = False
        Parameter.train = False
        Parameter.test = False
    elif args.mode == "train":
        Parameter.load_input = True
        Parameter.train = True
        Parameter.test = True
    elif args.mode == "test":
        Parameter.load_input = True
        Parameter.train = False
        Parameter.test = True
    elif args.mode == "all":
        Parameter.load_input = False
        Parameter.train = True
        Parameter.test = True
    else:
        raise ValueError("Unknown mode, choose one of the following [feature, train, test, all]")

# TODO: implement derivative of ground truth data variable true / false     Done + validated
# TODO: check weight initialization, set to xavier
# TODO: implement CNN + RNN (and CNN + LSTM and CNN + GRU)
# TODO: implement random search (or bayesian optimization)

# TODO: go through all the code and validate
# TODO: check if everything runs on GPU

save_dir = "./feature_set/DB_" + str(Parameter.database) + '/S' + str(Parameter.subject) + '/' + Parameter.comment + '_'
model_save_dir = './model/DB_' + str(Parameter.database) + '/S' + str(Parameter.subject)

if Parameter.load_input is True:
    training_set = torch.Tensor()
    validation_set = torch.Tensor()
    testing_set = torch.Tensor()

    training_set.input = torch.load(save_dir + 'training_in.pt').to(Parameter.device)
    training_set.ground_truth = torch.load(save_dir + 'training_gt.pt').to(Parameter.device)
    validation_set.input = torch.load(save_dir + 'validation_in.pt').to(Parameter.device)
    validation_set.ground_truth = torch.load(save_dir + 'validation_gt.pt').to(Parameter.device)
    testing_set.input = torch.load(save_dir + 'testing_in.pt').to(Parameter.device)
    testing_set.ground_truth = torch.load(save_dir + 'testing_gt.pt').to(Parameter.device)
else:
    os.makedirs(save_dir, exist_ok=True)
    training_set, validation_set, testing_set = process_data()

    torch.save(training_set.input, save_dir + 'training_in.pt')
    torch.save(training_set.ground_truth, save_dir + 'training_gt.pt')
    torch.save(validation_set.input, save_dir + 'validation_in.pt')
    torch.save(validation_set.ground_truth, save_dir + 'validation_gt.pt')
    torch.save(testing_set.input, save_dir + 'testing_in.pt')
    torch.save(testing_set.ground_truth, save_dir + 'testing_gt.pt')

input_size = training_set.input.shape[2]  # equals EMG channels x number of features
if Parameter.one_joint is True:
    output_size = 1
else:
    output_size = training_set.ground_truth.shape[2]

Parameter.parameter['input_size'] = input_size
os.makedirs(model_save_dir, exist_ok=True)
torch.save(Parameter.parameter, model_save_dir + '/' + str(
    Parameter.network) + '_' + Parameter.comment + '.pt')  # TODO: change this path (for online prediction)

if Parameter.network == "SVR":
    if Parameter.regression_model == 'SVR':
        clf = SVR(gamma='auto', C=Parameter.c, epsilon=Parameter.epsilon)  # 0.0124 C=2/0.0178 e=0.05/0.185
    elif Parameter.regression_model == 'LR':
        clf = LinearRegression()
    elif Parameter.regression_model == 'KRR':
        clf = KernelRidge(kernel=Parameter.kernel, alpha=Parameter.alpha, gamma=Parameter.gamma)
    # clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
    # clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
    # clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [10], "gamma": [1e-4]})

else:
    net = get_model(Parameter.network, input_size, output_size, Parameter.hidden_size, Parameter.batch_size,
                    Parameter.num_layers, Parameter.dropout, Parameter.bias)
    net.to(Parameter.device)
    net.apply(weights_init)

if Parameter.tensorboard is True:
    writer = SummaryWriter(comment=Parameter.comment)  # for tensorboardX


########################################################################################################################
# Training


def train():
    decay = 1.
    learning_rate = Parameter.learning_rate
    loss = get_Loss(Parameter.loss).to(Parameter.device)
    # optimizer = get_optimizer(Parameter.optimizer, net.parameters(), learning_rate, Parameter.weight_decay, Parameter.momentum)

    hidden_training = net.initHidden()
    hidden_validate = net.initHidden()

    net.zero_grad()
    net.train()

    mean_validation_losses = []

    for j in range(Parameter.epochs):
        # validation_losses = torch.Tensor()  # reset validation loss
        validation_losses = []
        training_losses = []

        optimizer = get_optimizer(Parameter.optimizer, net.parameters(), learning_rate, Parameter.weight_decay,
                                  Parameter.momentum)
        # if j >= 25 and j % 5 == 0:  # TODO: test the influence of this
        #    learning_rate = learning_rate / 5.

        for i in range(int(training_set.ground_truth.shape[0] / Parameter.sequence)):
            if Parameter.network == 'CNN' or Parameter.network == 'CRNN':  # dim [timestep, batch, channel, window length]
                input_training = training_set.input[i * Parameter.sequence:i * Parameter.sequence + Parameter.sequence,
                                 :, :, :]
                if i < validation_set.ground_truth.shape[0]:
                    input_validate = validation_set.input[
                                     i * Parameter.sequence:i * Parameter.sequence + Parameter.sequence, :, :, :]
            else:  # dim [timestep, batch, feature]
                input_training = training_set.input[i * Parameter.sequence:i * Parameter.sequence + Parameter.sequence,
                                 :, :]
                if i < validation_set.ground_truth.shape[0]:
                    input_validate = validation_set.input[
                                     i * Parameter.sequence:i * Parameter.sequence + Parameter.sequence, :, :]

            output_training, hidden_training = net(input_training, hidden_training)
            if i < validation_set.ground_truth.shape[0]:
                output_validate, hidden_validate = net(input_validate, hidden_validate)

            if Parameter.one_joint is True:
                loss_train = loss(output_training[:, 0], training_set.ground_truth[i, :, Parameter.joint]).to(
                    Parameter.device)
                if Parameter.tensorboard is True and Parameter.log_training_pred is True:
                    writer.add_scalars('Performance Training', {'prediction': output_training[-1].data.item(),
                                                                'ground_truth': training_set.ground_truth[
                                                                    i, -1, Parameter.joint].data.item()}, i)
                if i < validation_set.ground_truth.shape[0]:
                    loss_valid = loss(output_validate[:, 0], validation_set.ground_truth[i, :, Parameter.joint]).to(
                        Parameter.device)
                    if Parameter.tensorboard is True and Parameter.log_training_pred is True:
                        writer.add_scalars('Performance Validation',
                                           {'prediction': output_validate[-1].detach().data.item(),
                                            'ground_truth': validation_set.ground_truth[
                                                i, -1, Parameter.joint].data.item()}, i)
            else:
                loss_train = loss(output_training, training_set.ground_truth[i]).to(Parameter.device)
                if Parameter.tensorboard is True and Parameter.log_training_pred is True:
                    writer.add_scalars('Performance Training',
                                       {'prediction': output_training[-1, Parameter.joint].data.item(),
                                        'ground_truth': training_set.ground_truth[i, -1, Parameter.joint].data.item()}, i)
                if i < validation_set.ground_truth.shape[0]:
                    loss_valid = loss(output_validate, validation_set.ground_truth[i]).to(Parameter.device)
                    if Parameter.tensorboard is True and Parameter.log_training_pred is True:
                        writer.add_scalars('Performance Validation',
                                           {'prediction': output_validate[-1, Parameter.joint].detach().data.item(),
                                            'ground_truth': validation_set.ground_truth[
                                                i, -1, Parameter.joint].data.item()}, i)

            loss_train.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            output_training.detach()  # reset output for next sequence backward

            # if i < validation_set.ground_truth.shape[0]:
            #     writer.add_scalars('Loss', {'training_loss': loss_train.data.item(), 'validation_loss': loss_valid.data.item()}, i + j*training_set.input.shape[0])
            # else:
            #     writer.add_scalars('Loss', {'training_loss': loss_train.data.item()}, i + j*training_set.input.shape[0])

            print('Training Network [ Epoch: ', j + 1, ' / ', Parameter.epochs, '    ', 'Batch: ', i + 1, ' / ',
                  int(training_set.input.shape[0] / Parameter.sequence), ']', flush=True, end='\r')

            if i < validation_set.ground_truth.shape[0]:
                validation_losses.append(loss_valid.data.item())

            training_losses.append(loss_train.data.item())

            # TODO: try setting hidden to zero sometimes
            if (i + 1) % 10 == 0:  # much better performance!
                hidden_training = net.initHidden()
                hidden_validate = net.initHidden()

        hidden_training = net.initHidden()
        hidden_validate = net.initHidden()

        mean_validation_loss = mean(validation_losses)
        mean_validation_losses.append(mean_validation_loss)

        mean_training_loss = mean(training_losses)

        min_loss = min(mean_validation_losses)
        # index = mean_validation_losses.index(min_loss)

        # save the model with the lowest validation loss
        if mean_validation_loss == min_loss:
            print('save model ...')
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(net.state_dict(),
                       model_save_dir + '/' + str(Parameter.network) + '_' + Parameter.comment + '.pth')

        # write loss each epoch
        if Parameter.tensorboard is True:
            writer.add_scalars('Loss', {'training_loss': mean_training_loss, 'validation_loss': mean_validation_loss}, j)

        # # learning rate decay after five epochs stagnating
        # if len(mean_validation_losses) > 10 and len(mean_validation_losses) - index > 1:
        #     learning_rate = learning_rate / 3.
        #     optimizer = get_optimizer(Parameter.optimizer, net.parameters(), learning_rate, Parameter.weight_decay, Parameter.momentum)
        #     print('Learning rate decay to: ' + str(learning_rate))
        #     print('jo')
        #     print('hey!')
        #     if learning_rate < 1e-8:
        #         print('Learning rate smaller than 1e-8')
        #         break

        # exit criteria: no validation loss decrease for the last five epochs
        # if len(mean_validation_losses) - index > 10:
        #    print('Breaking criteria fulfilled.')
        #    break

    # save model and write to Tensorboard
    # torch.save(net.state_dict(), 'model.pth')
    if Parameter.tensorboard is True:
        addText()

    print('Training finished.')


def fit():
    print('Fitting...', flush=True, end='\r')
    clf.fit(training_set.input[:, 0, :].cpu().numpy(), training_set.ground_truth[:, 0, Parameter.joint].cpu().numpy())
    # clf.fit(training_set.input[:, 0, :], training_set.ground_truth[:, 0, Parameter.joint])
    if Parameter.tensorboard is True:
        addText()
    print('Fitting finished.')


def addText():
    writer.add_text('Parameter', 'Loss: ' + Parameter.loss, 1)
    writer.add_text('Parameter', 'Optimizer: ' + Parameter.optimizer, 2)
    writer.add_text('Parameter', 'Learning rate: ' + str(Parameter.learning_rate), 3)
    writer.add_text('Parameter', 'Weight decay: ' + str(Parameter.weight_decay), 4)
    writer.add_text('Parameter', 'Momentum: ' + str(Parameter.momentum), 5)
    writer.add_text('Parameter', 'Training size: ' + str(Parameter.training_size), 6)
    writer.add_text('Parameter', 'Validation size: ' + str(Parameter.validation_size), 7)
    writer.add_text('Parameter', 'Testing size: ' + str(Parameter.testing_size), 8)
    writer.add_text('Parameter', 'Epochs: ' + str(Parameter.epochs), 9)
    writer.add_text('Parameter', 'Sequence: ' + str(Parameter.sequence), 10)
    writer.add_text('Parameter', 'Batch size: ' + str(Parameter.batch_size), 11)
    writer.add_text('Parameter', 'Hidden size: ' + str(Parameter.hidden_size), 12)

    writer.add_text('Parameter1', 'EMG frequency: ' + str(Parameter.emg_frequency) + ' Hz', 1)
    writer.add_text('Parameter1', 'Window: ' + str(Parameter.window) + ' ms', 2)
    writer.add_text('Parameter1', 'Overlap: ' + str(Parameter.overlap) + ' ms', 3)
    writer.add_text('Parameter1', 'Electromechanical delay: ' + str(Parameter.emd) + ' ms', 4)
    writer.add_text('Parameter1', 'Low cut freq: ' + str(Parameter.low_cut_freq) + ' Hz', 5)
    writer.add_text('Parameter1', 'High cut freq: ' + str(Parameter.high_cut_freq) + ' Hz', 6)
    writer.add_text('Parameter1', 'Joint: ' + str(Parameter.joint), 7)
    writer.add_text('Parameter1', 'One joint: ' + str(Parameter.one_joint), 8)
    writer.add_text('Parameter1', 'Number of layers: ' + str(Parameter.num_layers), 9)
    writer.add_text('Parameter1', 'Dataset: ' + ', '.join(Parameter.dataset), 10)
    writer.add_text('Parameter1', 'Features: ' + ', '.join(Parameter.feature_set), 11)
    writer.add_text('Parameter1', 'Network: ' + Parameter.network, 12)

    writer.add_text('Parameter2', 'Acceleration data: ' + str(Parameter.acc), 1)
    writer.add_text('Parameter2', 'Normalize ground truth: ' + str(Parameter.normalize_gt), 2)
    writer.add_text('Parameter2', 'Denoise: ' + str(Parameter.denoise), 3)
    writer.add_text('Parameter2', 'Calculate Feature: ' + str(Parameter.calc_feature), 4)
    writer.add_text('Parameter2', 'Bias: ' + str(Parameter.bias), 5)
    writer.add_text('Parameter2', 'Dropout: ' + str(Parameter.dropout), 6)
    writer.add_text('Parameter2', 'Normalize input: ' + str(Parameter.normalize_in), 7)
    writer.add_text('Parameter2', 'Normalize feature: ' + str(Parameter.normalize_ft), 8)
    writer.add_text('Parameter2', 'Notch filter: ' + str(Parameter.notch), 9)
    writer.add_text('Parameter2', 'C (SVR): ' + str(Parameter.c), 10)
    writer.add_text('Parameter2', 'Epsilon (SVR): ' + str(Parameter.epsilon), 11)

    writer.add_text('Parameter3', 'Feature set im: ' + str(Parameter.feature_set_im), 1)
    writer.add_text('Parameter3', 'Kernel (SVR): ' + Parameter.kernel, 2)
    writer.add_text('Parameter3', 'Alpha (SVR): ' + str(Parameter.alpha), 3)
    writer.add_text('Parameter3', 'Gamma (SVR): ' + str(Parameter.gamma), 4)
    writer.add_text('Parameter3', 'Regression model (SVR): ' + Parameter.regression_model, 5)
    writer.add_text('Parameter3', 'Gyro data: ' + str(Parameter.gyro), 6)
    writer.add_text('Parameter3', 'Mag data: ' + str(Parameter.mag), 7)


def test():
    print('Testing Network ...', flush=True, end='\r')

    net.eval()
    torch.no_grad()

    hidden_training2 = net.initHidden()

    for k in range(int(training_set.ground_truth.shape[0] / Parameter.sequence)):
        predict_training, hidden_training2 = net(
            training_set.input[k * Parameter.sequence:k * Parameter.sequence + Parameter.sequence, :, :],
            hidden_training2)  # one extra 1
        predict_training.detach()  # TODO: what is detach doing?

        if Parameter.tensorboard is True:
            if Parameter.one_joint is True:
                writer.add_scalars('Prediction_Training', {'predict_training': predict_training[-1].data.item(),
                                                           'ground_truth_training': training_set.ground_truth[
                                                               k, -1, Parameter.joint].data.item()}, k)
            else:
                writer.add_scalars('Prediction_Training',
                                   {'predict_training': predict_training[-1, Parameter.joint].data.item(),
                                    'ground_truth_training': training_set.ground_truth[k, -1, Parameter.joint].data.item()},
                                   k)
        # hidden_training = net.initHidden()  # TODO: remove later

    if Parameter.train is True:
        test_net = net.eval()
    else:
        test_net = get_model(Parameter.network, input_size, output_size, Parameter.hidden_size, 1, Parameter.num_layers,
                             Parameter.dropout, Parameter.bias)
        state_dict = torch.load(model_save_dir + '/' + str(Parameter.network) + '_' + Parameter.comment + '.pth')
        test_net.load_state_dict(state_dict)
        test_net.eval()
        torch.no_grad()
        test_net.to(Parameter.device)

    hidden_testing = test_net.initHidden()

    prediction = torch.Tensor()

    for l in range(testing_set.ground_truth.shape[0]):
        if Parameter.network == "CNN" or Parameter.network == 'CRNN':
            predict_testing, hidden_testing = test_net(testing_set.input[l, :, :, :], hidden_testing)
        else:
            predict_testing, hidden_testing = test_net(testing_set.input[l].view(1, 1, -1),
                                                       hidden_testing)  # one extra 1
        if Parameter.tensorboard is True:
            if Parameter.one_joint is True:
                writer.add_scalars('Prediction_Testing', {'predict_testing': predict_testing.data.item(),
                                                          'ground_truth_training': testing_set.ground_truth[
                                                              l, Parameter.joint].data.item()}, l)
            else:
                testing_set.ground_truth.squeeze_()
                writer.add_scalars('Prediction_Testing',
                                   {'predict_testing': predict_testing[0, Parameter.joint].data.item(),
                                    'ground_truth_training': testing_set.ground_truth[l, Parameter.joint].data.item()}, l)
                # for m in range(testing_set.ground_truth.shape[1]):
                #    writer.add_scalars('Prediction_Testing', {'predict_testing': predict_testing[0, m].data.item(), 'ground_truth_training': testing_set.ground_truth[l, m].data.item()}, l)
                #    pass  # TODO: continue here

        # hidden_testing = net.initHidden()  # TODO: remove later

        prediction = torch.cat((prediction, predict_testing.detach().cpu()),
                               0)  # TODO: ValueError: y_true and y_pred have different number of output (1!=22)

    print('Testing finished.')

    os.makedirs('./scores/DB_' + str(Parameter.database), exist_ok=True)
    file_scores = open('./scores/DB_' + str(Parameter.database) + '/S' + str(Parameter.subject) + '.txt', 'a')

    if Parameter.one_joint is True:
        r2 = r2_score(testing_set.ground_truth[:, Parameter.joint].squeeze().cpu(), prediction)
        print('R2: {}'.format(r2))
    else:
        r2 = r2_score(testing_set.ground_truth.cpu(), prediction)
        print('Total R2: {}'.format(r2))

        for n in range(testing_set.ground_truth.shape[1]):
            r2_joint = r2_score(testing_set.ground_truth[:, n].squeeze().cpu(), prediction[:, n])
            print('Joint {} R2: {}'.format(n, r2_joint))
            file_scores.write('Joint {} R2: {}'.format(n, r2_joint) + '\n')

            # add to tensorboard
            if Parameter.tensorboard is True:
                if n < 12:
                    writer.add_text('R2 Score 1', 'J' + str(n) + ': ' + str(r2_joint), n)
                else:
                    writer.add_text('R2 Score 2', 'J' + str(n) + ': ' + str(r2_joint), n - 12)

    file_scores.write('R2 Score: ' + str(r2))
    file_scores.close()
    if Parameter.tensorboard is True:
        writer.add_text('Parameter2', 'R2 Score: ' + str(r2), 12)
        writer.close()


def testSVR():
    print('Testing Network ...', flush=True, end='\r')
    prediction = clf.predict(testing_set.input[:, 0, :].cpu())

    if Parameter.tensorboard is True:
        for m in range(len(prediction)):
            writer.add_scalars('Prediction_Testing', {'predict_testing': prediction[m],
                                                      'ground_truth_training': testing_set.ground_truth[
                                                          m, Parameter.joint].data.item()}, m)

    r2 = r2_score(testing_set.ground_truth[:, Parameter.joint].cpu(), prediction)
    if Parameter.tensorboard is True:
        writer.add_text('Parameter2', 'R2 Score: ' + str(r2), 12)
        writer.close()
    print('Testing finished.')
    print('R2: {}'.format(r2))


########################################################################################################################
# Execute code

# TODO: do a for loop and go through different parameter
if Parameter.train is True:
    if Parameter.network == "SVR":
        fit()
    else:
        train()

if Parameter.test is True:
    if Parameter.network == "SVR":
        testSVR()
    else:
        test()

print('Done.')
