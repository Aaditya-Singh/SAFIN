import argparse
from pathlib import Path
from importlib import import_module

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from function import adaptive_instance_normalization, SAFIN, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize((size, size)))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0, safin_list=None,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)[-1]  
    style_f = vgg(style)[-1] 
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        if safin_list:
            skips = {}
            transformed_f = vgg.encode_transform(safin_list[0], content, style, skips)
            base_feat = safin_list[1](transformed_f, style_f)
        else:
            base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        if safin_list:
            skips = {}
            transformed_f = vgg.encode_transform(safin_list[0], content, style, skips)
            feat = safin_list[1](transformed_f, style_f)
        else:
            feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    if safin_list: return decoder(feat, skips)
    else: return decoder(feat)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str,
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str,
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')
    parser.add_argument('--net_file', required=True, type=str, \
                        choices=['wave_net'], help='net file')
    parser.add_argument('--safin4', type=str, \
                        default='models/decoder.pth')
    parser.add_argument('--safin3', type=str, \
                        default='models/decoder.pth')

    # Additional options
    parser.add_argument('--content_size', type=int, default=512,
                        help='New (minimum) size for the content image, \
                        keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=512,
                        help='New (minimum) size for the style image, \
                        keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')

    # Advanced options
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                                 stylization. Should be between 0 and 1')
    parser.add_argument(
        '--style_interpolation_weights', type=str, default='',
        help='The weight for blending the style of multiple style images')

    args = parser.parse_args()

    do_interpolation = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)
    if args.style:
        style_paths = args.style.split(',')
        if len(style_paths) == 1:
            style_paths = [Path(args.style)]
        else:
            do_interpolation = True
            assert (args.style_interpolation_weights != ''), \
                'Please specify interpolation weights'
            weights = [int(i) for i in args.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    net = import_module(args.net_file)
    decoder = net.WaveDecoder()
    vgg = net.WaveEncoder()

    if args.net_file == 'wave_net':
        network = net.Net(vgg, decoder)
        safin4 = network.safin4; safin4.eval()
        safin3 = network.safin3; safin3.eval()
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder, map_location=device)['decoder_state_dict'])
    vgg.load_state_dict(torch.load(args.vgg, map_location=device))
    if args.net_file == 'wave_net':
        safin4.load_state_dict(torch.load(args.safin4, map_location=device)['meta_adain4_state_dict'])
        safin3.load_state_dict(torch.load(args.safin3, map_location=device)['meta_adain3_state_dict'])
        safin4.to(device); safin3.to(device)
        safin_list = [safin3, safin4]
    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)


    for content_path in content_paths:
        if do_interpolation:  # one content image, N style image
            style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
            content = content_tf(Image.open(str(content_path))) \
                .unsqueeze(0).expand_as(style)
            style = style.to(device)
            content = content.to(device)
            with torch.no_grad():
                if args.net_file == 'wave_net':
                    safin_list = [safin3, safin4]
                    output = style_transfer(vgg, decoder, content, style, args.alpha, \
                                            safin_list, interpolation_weights)
                else : 
                    output = style_transfer(vgg, decoder, content, style, args.alpha, \
                                            None, interpolation_weights)
            output = output.cpu()
            output_name = output_dir / '{:s}_interpolation{:s}'.format(
                content_path.stem, args.save_ext)
            save_image(output, str(output_name))

        else:  # process one content and one style
            for style_path in style_paths:
                content = content_tf(Image.open(str(content_path)))
                style = content_tf(Image.open(str(style_path)))
                if args.preserve_color:
                    style = coral(style, content)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    if args.net_file == 'wave_net':
                        output = style_transfer(vgg, decoder, content, style, args.alpha, \
                                                safin_list, None)
                    else : 
                        output = style_transfer(vgg, decoder, content, style, args.alpha, \
                                                None, None)
                output = output.cpu()

                output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                    content_path.stem, style_path.stem, args.save_ext)
                save_image(output, str(output_name))
