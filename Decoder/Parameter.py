import torch
#import Main


# check for Cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ' + str(device))
torch.cuda.empty_cache()

parameter = {}

#TODO: do dropout and weight decay!  done
#TODO: do different angles and subjects
#TODO: do multijoint (compare to seperate training)
#TODO: implement quat and gyro
#TODO: do overlap and maybe batch size
#TODO: selectkbest and reduce feature set
#TODO: use other networks (also CRNN)

comment = 'cross-subject8'
#TODO: smaller learning rate worked well, regularize even more (1e-5)
#TODO: regularize with hidden size and even more dropout

parse_args = False           # parse args as arguments in command line
reduce_electrodes = False    # reduce number of used electrodes
tensorboard = True          # turn tensorboard logs on or off
log_training_pred = False   # Decide whether to log predictions of training and validation set during training process

load_input = False          # load previously generated features
train = False                # train network
test = False                 # test network
database = '7'              # [1, 2, 7, 8, Myo, cross-subject8]
subject = 22
exercise = 'ABC'

training_size = 1.0  # 0,4  # TODO: try increasing this
validation_size = 0.0  # 0,4
testing_size = 0.0  # 0,05

batch_size = 8              # 32 is good  # TODO: try different batch sizes for CNN
emg_frequency = 2000.       # [Hz]  100Hz for Ottobock in dataset 1 and 2000Hz for Delsys in dataset 2 and 1111Hz for dataset 8
window = 256.               # [ms] length of data window for computing one timestep  150 or 200
overlap = 100.              # [ms] length of overlap of data window  75 or 150/175
emd = 0.                    # [ms] electromechanical delay
low_cut_freq = 20.          # 20
high_cut_freq = 500.        # max 49 for OttoBock electrodes in dataset 1

normalize_gt = True  # False works better for CNN?
normalize_in = True  # False for Myo data
normalize_ft = True
derivative_gt = False

denoise = False
notch = False

calc_feature = True
acc = True  # needs to be off for db1
mag = True  # needs to be off for db1 and db2
gyro = True  # needs to be off for db1 and db2
split_dataset = False  # splits dataset repetition wise

if parse_args is True:
    dataset = ""
else:
    if database == '1':
        emg_frequency = 100.
        acc = False
        mag = False
        gyro = False
        dataset = [
            '../Ninapro/Dataset_1/s' + str(subject) + '/S' + str(subject) + '_A1_E1.mat',
            '../Ninapro/Dataset_1/s' + str(subject) + '/S' + str(subject) + '_A1_E3.mat',
            '../Ninapro/Dataset_1/s' + str(subject) + '/S' + str(subject) + '_A1_E2.mat']
    elif database == '2':
        emg_frequency = 2000.
        mag = False
        gyro = False
        dataset = [
            '../Ninapro/Dataset_2/DB2_s' + str(subject) + '/S' + str(subject) + '_E2_A1.mat',
            '../Ninapro/Dataset_2/DB2_s' + str(subject) + '/S' + str(subject) + '_E1_A1.mat']
    elif database == '7':
        emg_frequency = 2000.
        dataset = [
            '../Ninapro/Dataset_7/Subject_' + str(subject) + '/S' + str(subject) + '_E2_A1.mat',
            '../Ninapro/Dataset_7/Subject_' + str(subject) + '/S' + str(subject) + '_E1_A1.mat']
    elif database == '8':
        emg_frequency = 1111.
        dataset = ['../Ninapro/Dataset_8/S' + str(subject) + '_E1_A1.mat',
                             '../Ninapro/Dataset_8/S' + str(subject) + '_E1_A2.mat',
                             '../Ninapro/Dataset_8/S' + str(subject) + '_E1_A3.mat']
    elif database == 'Myo':
        emg_frequency = 50.
        mag = False
        dataset = ['../Ninapro/Myo/S' + str(subject) + '_E1.mat',
                             '../Ninapro/Myo/S' + str(subject) + '_E2.mat']
    elif database == 'cross-subject8':
        emg_frequency = 1111.
    #     dataset = []
    #     for s in range(2):
    #         for a in range(3):
    #             dataset.append('../Ninapro/Dataset_8/S' + str(s+1) + '_E1_A' + str(a+1) + '.mat')
    elif database == 'cross-subject7':
        emg_frequency = 2000.
    else:
        print('Invalid database!')

supported_features = ['root_mean_square', 'wave_length', 'histogram', 'mean_absolute_value', 'temporal_moment_3', 'variance', 'log_detector', 'integrated_emg', 'kurtosis', 'average_amplitude_change', 'dasdv', 'simple_square_integral', 'skewness', 'total_power', 'spectral_moment_3', 'mean_frequency', 'median_frequency', 'mean_power', 'peak_frequency', 'variance_of_central_frequency', 'zero_crossings', 'slope_sign_changes', 'DWT']  # full feature set
feature_set = ['zero_crossings', 'simple_square_integral', 'integrated_emg', 'log_detector', 'variance', 'temporal_moment_3', 'mean_absolute_value', 'wave_length', 'root_mean_square'] # time-domain
supported_features_im = ['mean_value']
feature_set_im = ['mean_value']

used_electrodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Optimizer parameter
optimizer = 'SGD'  # [SGD, Adam, Adagrad, Adadelta, RMSprop]  Adagrad best so far  # TODO: try Adagrad for CNN
initializer = ''  # [Xavier_uniform, Xavier_normal, Kaiming_uniform, Kaiming_normal]  Kaiming = He, empty string will cause default initilization
learning_rate = 1e-5  # 5e-4   TODO: 1e-5 brings validation loss down, do learning rate decay  done
weight_decay = 0.001  # TODO: try different values here was 0.01 before (this is best)
momentum = 0.95

loss = 'MSELoss'   # [L1Loss, MSELoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, HingeEmbeddingLoss, SmoothL1Loss, CosineEmbeddingLoss] MSE works best
network = 'LSTM'  # [RNN, LSTM, GRU, CNN, SVR]

epochs = 200  # 128 for batch 32  #TODO: 15 epochs is not enough
sequence = 1  # this has probably effect on the delay!

hidden_size = 128  # regularizes a little bit

joint = 10  # 20 is wrist
one_joint = False
num_layers = 1


bias = True
dropout = 0.0  # 0.7 did not work better

grid_search = False

regression_model = 'KRR'  # [LR, SVR, KRR]

kernel = 'rbf'
alpha = 5.
gamma = 1e-3

# for SVR only
if grid_search is True:
    c = [2. ** -5., 2. ** -3., 2. ** -1., 2. ** 1., 2. ** 3., 2. ** 5., 2. ** 7., 2. ** 9., 2. ** 11., 2. ** 13., 2. ** 15.]
    epsilon = [2. ** -15., 2. ** -13., 2. ** -11., 2. ** -9., 2. ** -7., 2. ** -5., 2. ** -3., 2. ** -1., 2. ** 1., 2. ** 3.]
else:
    c = 2.0  # 2.0
    epsilon = 0.05  # 1. / 128.  # 0.05


if network == 'SVR':
    batch_size = 1
    one_joint = True
if network == 'CNN':
    calc_feature = False

parameter['feature_set'] = feature_set
parameter['feature_set_acc'] = feature_set_im
parameter['dataset'] = dataset
parameter['acc'] = acc
parameter['calc_feature'] = calc_feature
parameter['notch'] = notch
parameter['denoise'] = denoise
parameter['normalize_gt'] = normalize_gt
parameter['normalize_in'] = normalize_in
parameter['normalize_ft'] = normalize_ft
parameter['derivative_gt'] = derivative_gt
parameter['low_cut_freq'] = low_cut_freq
parameter['high_cut_freq'] = high_cut_freq
parameter['emd'] = emd
parameter['window'] = window
parameter['overlap'] = overlap
parameter['emg_frequency'] = emg_frequency
parameter['database'] = database
parameter['network'] = network
parameter['hidden_size'] = hidden_size
parameter['joint'] = joint
parameter['one_joint'] = one_joint
parameter['num_layers'] = num_layers
parameter['bias'] = bias
