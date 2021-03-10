import argparse
import random
from importlib import import_module
from pathlib import Path

import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from sampler import InfiniteSamplerWrapper
from test import style_transfer

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None               # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True      # Disable OSError: image file is truncated


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, required=True,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

    # training options
    parser.add_argument('--save_dir',required=True,
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--rand_seed', type=int, default=777, help='manual seed')
    parser.add_argument('--net_file', type=str,
                        choices=['wave_net'],
                        required=True,
                        help='net file')
    parser.add_argument('--start_iter', type=int, default=0)
    args = parser.parse_args()
    prepare_seed(args.rand_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(str(args.save_dir) + str(args.net_file) + '_' + str(args.max_iter))
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f'=> save_dir: {str(save_dir)}')
    # log_dir = Path(args.log_dir)
    # log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(save_dir))

    net = import_module(args.net_file)

    decoder = net.WaveDecoder()
    vgg = net.WaveEncoder()

    vgg.load_state_dict(torch.load(args.vgg))
    network = net.Net(vgg, decoder)

    if args.start_iter > 0:
        print("Loading state after {:d} iterations".format(args.start_iter + 0))
        states = torch.load(save_dir / 'ckpt_iter_{:d}.pth.tar'.format(args.start_iter))
        network.decoder.load_state_dict(states['decoder_state_dict'])
        network.safin4.load_state_dict(states['safin4_state_dict'])
        network.safin3.load_state_dict(states['safin3_state_dict'])

    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    assert os.path.exists(args.content_dir), args.content_dir
    assert os.path.exists(args.style_dir), args.style_dir
    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)
    print(f"length of content_dataset:{len(content_dataset)}")
    print(f"length of style_dataset: {len(style_dataset)}")

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    if hasattr(network, 'safin4'):
        params = list(network.decoder.parameters())+list(network.safin4.parameters())+\
        list(network.safin3.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
        print('=> training safin')

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        network.train()
        loss_c, loss_s = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter or i == 0:
            if hasattr(network, 'safin4'):
                states = {
                    'decoder_state_dict': network.decoder.state_dict(),
                    'safin4_state_dict': network.safin4.state_dict(),
                    'safin3_state_dict': network.safin3.state_dict()
                }
            else:
                states = {
                    'decoder_state_dict': network.decoder.state_dict(),
                }
            torch.save(states, save_dir /
                       'ckpt_iter_{:d}.pth.tar'.format(i + 1))
            with torch.no_grad():
                network.eval()
                if hasattr(network, 'safin4'):
                    safin_list = [network.safin3, network.safin4]
                    output = style_transfer(vgg, network.decoder, content_images, style_images, \
                                            1.0, safin_list)
                else : 
                    output = style_transfer(vgg, network.decoder, content_images, style_images, \
                                            1.0, None)
                styled_img_grid = make_grid(output, nrow=4, normalize=True, scale_each=True)
                reference_img_grid = make_grid(style_images, nrow=4, normalize=True, scale_each=True)
                content_img_grid = make_grid(content_images, nrow=4, normalize=True, scale_each=True)

                writer.add_image('styled_images', styled_img_grid, i)
                writer.add_image('reference_images', reference_img_grid, i)
                writer.add_image('content_images', content_img_grid, i)

    writer.close()
