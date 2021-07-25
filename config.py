import torch

class config(object):

    batch_size = 100
    imageSize = 64
    In = 100
    Out = 3
    lr = 0.0002
    betas = (0.5, 0.999)
    epoch = 50
    save_model = 1
    dataset_dir = './data/'
    results_dir = './images/'
    checkpoint_dir = './checkpoint/'
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}