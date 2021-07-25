import os
import torch
import torchvision.utils as vutils
from config import config
from dataset import get_dataloader
from model import netD, netG, weights_init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(data_loader, netD, netG, criterion, optimizerG, optimizerD):
    for epoch in range(config.epoch):
        avg_lossD = 0
        avg_lossG = 0
        FILE_LIST = os.listdir(config.dataset_dir)
        FILE_LIST.sort(key = lambda x: x[:-4])
        for i, data in enumerate(data_loader):
            mini_batch = data.shape[0]
            input = data.to(device)
            real_label = torch.ones(mini_batch).to(device)

            output = netD(input)
            D_real_loss = criterion(output, real_label)
            noise = torch.randn(mini_batch, config.In).view(-1, config.In, 1, 1).to(device)
            fake = netG(noise)
            fake_label = torch.zeros(mini_batch).to(device)
            output = netD(fake.detach())
            G_real_loss = criterion(output, fake_label)
            D_loss = D_real_loss + G_real_loss
            netD.zero_grad()
            D_loss.backward()
            avg_lossD += D_loss.item()
            optimizerD.step()

            output = netD(fake)
            G_loss = criterion(output, real_label)
            avg_lossG += G_loss.item()
            netG.zero_grad()
            G_loss.backward()
            optimizerG.step()

            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                    % (epoch + 1, config.epoch, i + 1, len(data_loader), D_loss.item(), G_loss.item()))

        avg_lossD /= i
        avg_lossG /= i
        print('epoch: ' + str(epoch) + ', G_loss: ' + str(avg_lossG) + ', D_loss: ' + str(avg_lossD))

        fixed_pred = netG(fixed_noise)
        vutils.save_image(fixed_pred.data, os.path.join(config.results_dir, 'img'+str(epoch)+'.png'), nrow=10, scale_each=True)

        if epoch%200 == 0:
            if config.save_model:
                torch.save(netD.state_dict(), os.path.join(config.checkpoint_dir, 'netD-01.pt'))
                torch.save(netG.state_dict(), os.path.join(config.checkpoint_dir, 'netG-01.pt'))
        '''
        model = netD(input)
        model.load_state_dict(torch.load('netD-01.pt'))
        model.eval()
        '''


if __name__ == '__main__':
    data_loader = get_dataloader()
    fixed_noise = torch.randn(100, config.In).view(-1, config.In, 1, 1).to(device)

    netG = netG().to(device)
    # netG.apply(weights_init)
    netD = netD().to(device)
    # netD.apply(weights_init)

    criterion = torch.nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=config.lr, betas=config.betas)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=config.lr, betas=config.betas)
    train(data_loader, netD, netG, criterion, optimizerG, optimizerD)