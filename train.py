from dataset.Dataset import SiameseDataset
from model.model import SiameseNet, ContrastiveLoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision import datasets, transforms
from PIL import Image
import argparse
import time

def main(args):
    train_dir = "./data/faces/training"
    test_dir = "./data/faces/testing"

    train_folder = datasets.ImageFolder(root=train_dir)
    train_dataset = SiameseDataset(ImgDataset=train_folder,
                                   transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                 transforms.ToTensor()]),
                                   invert=False)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=args.worker,
                                  batch_size=args.batch_size)
    model = SiameseNet()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    max_iter = min(args.batch_size * args.iteration, len(train_dataset))
    max_iter = int(max_iter/args.batch_size)

    start_time = time.time()
    for epoch in range(args.epoch):
        data_iter = enumerate(train_dataloader)
        for iter in range(max_iter):
            index, out = next(data_iter)
            img0, img1, label = out

            optimizer.zero_grad()
            output0, output1 = model(img0, img1)
            loss = criterion(output0, output1, label).mean()
            loss.backward()
            optimizer.step()

            if iter %10 == 0:
                print("Epoch {}, iter [{}/{}]: {:.5f}  label: {}\n".format(epoch, iter, max_iter, loss.item(), label[0, 0]))
        print("Epoch {} training done!\n".format(epoch))
    elapse_time = time.time() - start_time
    print("====================================")
    print("Totoal training time: {:.3f} seconds\n".format(elapse_time))
    print("====================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--worker", type=int, default=0)

    args = parser.parse_args()

    print("Input Arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    main(args)
