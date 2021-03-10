import torch.nn as nn
from utils import *
from function import SAFIN
from function import calc_mean_std


class WaveEncoder(nn.Module):
    def __init__(self):
        super(WaveEncoder, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from image (optional reluX_2)
    def encode(self, x, skips, level, feats = None):
        if feats == None: feats = []
        assert level in {1, 2, 3, 4}
        if level == 1:
            out = self.conv0(x)
            out = self.relu(self.conv1_1(self.pad(out)))
            feats.append(out)
            out = self.relu(self.conv1_2(self.pad(out)))
            ll, lh, hl, hh = self.pool1(out)
            skips['pool1'] = [lh, hl, hh]
            return ll

        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            feats.append(out)
            out = self.relu(self.conv2_2(self.pad(out)))
            ll, lh, hl, hh = self.pool2(out)
            skips['pool2'] = [lh, hl, hh]
            return ll

        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            feats.append(out)
            out = self.relu(self.conv3_2(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))
            ll, lh, hl, hh = self.pool3(out)
            skips['pool3'] = [lh, hl, hh]
            return ll

        else:
            out = self.relu(self.conv4_1(self.pad(x)))
            feats.append(out)
            return out

    def get_all_features(self, x):
        outs = [x]; skips = {}
        for level in [1, 2, 3, 4]:
            outs.append(self.encode(outs[-1], skips, level))
        return outs[1: ], skips

    def encode_transform(self, safin3, content, style, content_skips = None):
        if content_skips is None: content_skips = {}
        content_feat = content
        style_feats, style_skips = self.get_all_features(style)
        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if level == 4: continue 
            content_feat = stat_transform(content_feat, style_feats[level-1])
            # transform skips too
            for skip in [0, 1, 2]:
                if level == 3:
                  content_skips['pool{}'.format(level)][skip] = \
                  safin3(content_skips['pool{}'.format(level)][skip], \
                  style_skips['pool{}'.format(level)][skip])
                else:
                  content_skips['pool{}'.format(level)][skip] = \
                  stat_transform(content_skips['pool{}'.format(level)][skip], \
                  style_skips['pool{}'.format(level)][skip])                  
        return content_feat

    def forward(self, x):
        results = []
        for level in [1, 2, 3, 4]:
            x = self.encode(x, {}, level, results)
        return results


class WaveDecoder(nn.Module):
    def __init__(self):
        super(WaveDecoder, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.recon_block3 = WaveUnpool(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
    
        self.recon_block2 = WaveUnpool(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            lh, hl, hh = skips['pool3']
            out = self.recon_block3(out, lh, hl, hh)
            out = self.relu(self.conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))

        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            lh, hl, hh = skips['pool2']
            out = self.recon_block2(out, lh, hl, hh)
            return self.relu(self.conv2_2(self.pad(out)))

        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            lh, hl, hh = skips['pool1']
            out = self.recon_block1(out, lh, hl, hh)
            return self.relu(self.conv1_2(self.pad(out)))

        else:
            return self.conv1_1(self.pad(x))

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.safin4 = SAFIN(512)
        self.safin3 = SAFIN(256)
        # fix the encoder
        for param in getattr(self, 'encoder').parameters():
            param.requires_grad = False

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)    # meta adaIN
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0, output_shared=False):
        assert 0 <= alpha <= 1
        content_skips = {}
        style_feats = self.encoder(style)
        content_feat = self.encoder(content)[-1]
        mod_content_feat = self.encoder.encode_transform(self.safin3, content, style,\
        content_skips)
        # t = stat_transform(mod_content_feat, style_feats[-1])
        t = self.safin4(mod_content_feat, style_feats[-1], output_shared)
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t, content_skips)
        g_t_feats = self.encoder(g_t)

        # loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feat)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s