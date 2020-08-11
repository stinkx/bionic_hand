import torch
import os


database = '8'
subject = '1'
comment = 'cross-subject8'
save_dir = './Decoder/feature_set/DB_cross-subject' + database + '/S' + str(subject) + '/' + comment + '_'

dataset_in = torch.Tensor().cuda()
dataset_gt = torch.Tensor().cuda()

training_size = 8/12.  # 0,4  # TODO: try increasing this
validation_size = 2/12.  # 0,4
testing_size = 2/12.  # 0,05
batch_size = 8
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


for s in subjects:
    load_dir = "./Decoder/feature_set/DB_" + str(database) + '/S' + str(s) + '/' + comment + '_'
    f = torch.load(load_dir + 'training_in.pt')
    dataset_in = torch.cat((dataset_in, torch.load(load_dir + 'training_in.pt').cuda()), 0)
    dataset_gt = torch.cat((dataset_gt, torch.load(load_dir + 'training_gt.pt').cuda()), 0)


l = dataset_in.shape[0]
training_set_in = dataset_in[:round(training_size*l), :, :].clone()  # use clone to reduce file size
training_set_gt = dataset_gt[:round(training_size*l), :, :].clone()
validation_set_in = dataset_in[round(training_size*l):round((training_size+validation_size)*l), :, :].clone()
validation_set_gt = dataset_gt[round(training_size*l):round((training_size+validation_size)*l), :, :].clone()
testing_set_in = dataset_in[round((training_size+validation_size)*l):, :, :].clone()
testing_set_gt = dataset_gt[round((training_size+validation_size)*l):, :, :].clone()

# reshape to match batch size
mod_train = training_set_in.shape[0] % batch_size
mod_valid = validation_set_in.shape[0] % batch_size
training_set_in = training_set_in[:training_set_in.shape[0] - mod_train, :, :]
training_set_gt = training_set_gt[:training_set_gt.shape[0] - mod_train, :, :]
validation_set_in = validation_set_in[:validation_set_in.shape[0] - mod_valid, :, :]
validation_set_gt = validation_set_gt[:validation_set_gt.shape[0] - mod_valid, :, :]

training_set_in = training_set_in.view(-1, batch_size, training_set_in.shape[2])
training_set_gt = training_set_gt.view(-1, batch_size, training_set_gt.shape[2])
validation_set_in = validation_set_in.view(-1, batch_size, validation_set_in.shape[2])
validation_set_gt = validation_set_gt.view(-1, batch_size, validation_set_gt.shape[2])


os.makedirs(save_dir, exist_ok=True)

torch.save(training_set_in.detach(), save_dir + 'training_in.pt')
torch.save(training_set_gt.detach(), save_dir + 'training_gt.pt')
torch.save(validation_set_in, save_dir + 'validation_in.pt')
torch.save(validation_set_gt, save_dir + 'validation_gt.pt')
torch.save(testing_set_in, save_dir + 'testing_in.pt')
torch.save(testing_set_gt, save_dir + 'testing_gt.pt')
