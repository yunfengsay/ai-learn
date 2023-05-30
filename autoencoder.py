
import torch
import visdom
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import nn, optim
 
 
 
 
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            # [b, 784] => [b, 256]
            nn.Linear(784, 256),
            nn.ReLU(),
            # [b, 256] => [b, 64]
            nn.Linear(256, 64),
            nn.ReLU(),
            # [b, 64] => [b, 20]
            nn.Linear(64, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # [b, 20] => [b, 64]
            nn.Linear(20, 64),
            nn.ReLU(),
            # [b, 64] => [b, 256]
            nn.Linear(64, 256),
            nn.ReLU(),
            # [b, 256] => [b, 784]
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
 
 
    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, -1)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batchsz, 1, 28, 28)
 
 
        return x
 
 
def main():
    mnist_train = datasets.MNIST('mnist', train=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)
 
 
    mnist_test = datasets.MNIST('mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32)
 
 
    epochs = 1000
    lr = 1e-3
    model = AE()
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)
 
 
    viz = visdom.Visdom()
    for epoch in range(epochs):
        # 不需要label，所以用一个占位符"_"代替
        for batchidx, (x, _) in enumerate(mnist_train):
            x_hat = model(x)
            loss = criteon(x_hat, x)
 
 
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(epoch, 'loss:', loss.item())
        x, _ = next(iter(mnist_test))
        with torch.no_grad():
            x_hat = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))
 
 
if __name__ == '__main__':
    main()
