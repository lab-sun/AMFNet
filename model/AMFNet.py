import torch
import torch.nn as nn
from torch.nn.modules import padding 
import torchvision.models as models 
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat

class AMFNet(nn.Module):

    def __init__(self, n_class):
        super(AMFNet, self).__init__()

        resnet_raw_model1 = models.resnet50(pretrained=True)
        resnet_raw_model2 = models.resnet50(pretrained=True)
        self.inplanes = 2048

        ########  Thermal ENCODER  ########
 
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = BottleStack(dim=1024,fmap_size=(18,32),dim_out=2048,proj_factor = 4,num_layers=3,heads=4,dim_head=512)

        ########  RGB ENCODER  ########
 
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4= BottleStack(dim=1024,fmap_size=(18,32),dim_out=2048,proj_factor = 4,num_layers=3,heads=4,dim_head=512)

        ########  DECODER  ########
        self.upconv5 = upbolckV2(cin=2048,cout=1024)
        self.upconv4 = upbolckV2(cin=1024,cout=512)
        self.upconv3 = upbolckV2(cin=512,cout=256)
        self.upconv2 = upbolckV2(cin=256,cout=128)
        self.upconv1 = upbolckV2(cin=128,cout=n_class)

        ########  FUSION  ########
        self.fusion1 = Fusion_V2(in_channels=64,med_channels=32,channel=128)
        self.fusion2 = Fusion_V2(in_channels=256,med_channels=128,channel=512)
        self.fusion3 = Fusion_V2(in_channels=512,med_channels=256,channel=1024)
        self.fusion4 = Fusion_V2(in_channels=1024,med_channels=512,channel=2048)
        self.fusion5 = Fusion_V2(in_channels=2048,med_channels=1024,channel=4096)

        self.skip_tranform = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
 
    def forward(self, input):

        rgb = input[:,:3]
        thermal = input[:,3:4]

        mask = input[:,4:5]

        verbose = False

        # encoder

        ######################################################################

        if verbose: print("rgb.size() original: ", rgb.size())  
        if verbose: print("thermal.size() original: ", thermal.size()) 

        ######################################################################

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size()) 
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size()) 

        thermal = self.encoder_thermal_conv1(thermal)
        if verbose: print("thermal.size() after conv1: ", thermal.size()) 
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose: print("thermal.size() after bn1: ", thermal.size()) 
        thermal = self.encoder_thermal_relu(thermal)
        if verbose: print("thermal.size() after relu: ", thermal.size()) 

        rgb = self.fusion1(rgb,thermal,mask)
        skip1 = rgb
        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size()) 
        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose: print("thermal.size() after maxpool: ", thermal.size()) 
        ######################################################################
        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size()) 
        thermal = self.encoder_thermal_layer1(thermal)
        if verbose: print("thermal.size() after layer1: ", thermal.size()) 
        rgb = self.fusion2(rgb,thermal,mask)
        skip2 = rgb
        ######################################################################
        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size()) 
        thermal = self.encoder_thermal_layer2(thermal)
        if verbose: print("thermal.size() after layer2: ", thermal.size()) 
        rgb = self.fusion3(rgb,thermal,mask)
        skip3 = rgb
        ######################################################################
        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size()) 
        thermal = self.encoder_thermal_layer3(thermal)
        if verbose: print("thermal.size() after layer3: ", thermal.size()) 
        rgb = self.fusion4(rgb,thermal,mask)
        skip4 = rgb
        if verbose: print("rgb.size() after fusion_con3d4: ", rgb.size()) 
        ######################################################################
        rgb = self.encoder_rgb_layer4(rgb)
        if verbose: print("thermal.size() after layer4: ", rgb.size()) 
        thermal = self.encoder_thermal_layer4(thermal)
        if verbose: print("thermal.size() after layer4: ", thermal.size()) 
        fuse = self.fusion5(rgb,thermal,mask)

        ######################################################################
  
        # decoder
        fuse = self.upconv5(fuse)
        if verbose: print("fuse after deconv1: ", fuse.size()) # (30, 40)
        fuse = fuse+skip4

        fuse = self.upconv4(fuse)
        if verbose: print("fuse after deconv2: ", fuse.size()) # (60, 80)
        fuse = fuse+skip3

        fuse = self.upconv3(fuse)
        if verbose: print("fuse after deconv3: ", fuse.size()) # (120, 160)
        fuse = fuse+skip2
        fuse = self.upconv2(fuse)
        if verbose: print("fuse after deconv4: ", fuse.size()) # (240, 320)
        skip1 = self.skip_tranform(skip1)
        fuse = fuse+skip1
        fuse = self.upconv1(fuse)
        if verbose: print("fuse after deconv5: ", fuse.size()) # (480, 640)

        return fuse
  
class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)  
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class upbolckV2(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        
        self.conv1 = nn.Conv2d(cin,cin//2,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(cin//2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(cin//2,cin//2,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(cin//2)
        self.relu2 = nn.ReLU(inplace=True)       
 
        self.conv3 = nn.Conv2d(cin//2,cin//2,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(cin//2)
        self.relu3 = nn.ReLU(inplace=True)  

        self.shortcutconv = nn.Conv2d(cin,cin//2,kernel_size=1,stride=1)
        self.shortcutbn = nn.BatchNorm2d(cin//2)
        self.shortcutrelu = nn.ReLU(inplace=True)  


        self.se = SE_fz(in_channels=cin//2,med_channels=cin//4)

        self.transconv = nn.ConvTranspose2d(cin//2,cout,kernel_size=2, stride=2, padding=0, bias=False)
        self.transbn = nn.BatchNorm2d(cout)
        self.transrelu = nn.ReLU(inplace=True)

    def forward(self,x):

        fusion = self.conv1(x)
        fusion = self.bn1(fusion)
        fusion = self.relu1(fusion)

        sc0 = fusion


        fusion = self.conv2(fusion)
        fusion = self.bn2(fusion)
        fusion = self.relu2(fusion)

        fusion = sc0 + fusion

        fusion = self.conv3(fusion)
        fusion = self.bn3(fusion)
        fusion = self.relu3(fusion)

        sc = self.shortcutconv(x)
        sc = self.shortcutbn(sc)
        sc = self.shortcutrelu(sc)

        fusion = fusion+sc

        fusion = self.se(fusion)


        fusion = self.transconv(fusion)
        fusion = self.transbn(fusion)
        fusion = self.transrelu(fusion)

        return fusion

class SE_fz(nn.Module):
    def __init__(self, in_channels, med_channels):
        super(SE_fz, self).__init__()

        self.average = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels,med_channels)
        self.bn1 = nn.BatchNorm1d(med_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(med_channels,in_channels)
        self.sg = nn.Sigmoid()
    
    def forward(self,input):
        x = input
        x = self.average(input)
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = self.fc1(x)
        x= self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sg(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        out = torch.mul(input,x)
        return out


class Fusion_V2(nn.Module):
    def __init__(self, in_channels, med_channels,channel):
        super().__init__()
        self.Weight = weight(linearhidden=channel)

        self.pam = PAM(channel=in_channels)
        self.cam = SE_fz(in_channels=in_channels,med_channels=med_channels)

    def forward(self, rgb, thermal, mask):

        weights = self.Weight(rgb,thermal)

        B,C,H,W = rgb.size()

        mask = F.interpolate(mask,[H,W],mode="nearest")

        mask_rgb = torch.ones(B,1,H,W)
        if mask.is_cuda:
            mask_rgb = mask_rgb.cuda(mask.device)

        mask_thermal = torch.mul(mask.reshape(B,-1),weights[:,1].reshape((B,1))).reshape(B,1,H,W)

        mask_rgb = mask_rgb-mask_thermal

        fusion = rgb * mask_rgb + thermal * mask_thermal

        fusion = self.cam(fusion)
        fusion = self.pam(fusion)


        return fusion



class weight(nn.Module):
    def __init__(self, linearhidden):
        super().__init__()
        self.adapool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(linearhidden,linearhidden//2)
        self.bn1 = nn.BatchNorm1d(linearhidden//2)
        self.relu1 = nn.ReLU(True)

        self.fc2 = nn.Linear(linearhidden//2,linearhidden//4)
        self.bn2 = nn.BatchNorm1d(linearhidden//4)
        self.relu2 = nn.ReLU(True)

        self.fc3 = nn.Linear(linearhidden//4,2)
        self.relu3 = nn.ReLU(True)
        self.sf = nn.Softmax(dim=1)
    def forward(self,rgb,thermal):

        x = torch.cat((rgb,thermal),dim=1)
        x = self.adapool(x)
        b = x.size(0)
        x=x.reshape(b,-1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.sf(x)
        return x


class PAM(nn.Module):
    """ Position Attention Module """
    def __init__(self, channel):
        super(PAM, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W
        Returns:
            [torch.tensor]: size N*C*H*W
        """
        _, c, _, _ = x.size()
        y = self.act(self.conv(x))
        y = y.repeat(1, c, 1, 1)
        return x * y


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.pos_emb = AbsPosEmb(fmap_size, dim_head)
        

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

        return out

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.scale = scale
        self.height = nn.Parameter(torch.randn(fmap_size[0], dim_head) * scale)
        self.width = nn.Parameter(torch.randn(fmap_size[1], dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb) * self.scale
        return logits

class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:  #di yi bian de shi hou zhi xing
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attention_dim = dim_out // proj_factor



        self.net = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias = False),
            nn.BatchNorm2d(attention_dim),
            activation,
            Attention(
                dim = attention_dim,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(heads*dim_head),
            activation,
            nn.Conv2d(heads*dim_head, dim_out, 1, bias = False),
            nn.BatchNorm2d(dim_out)
        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):
        
        shortcut = self.shortcut(x)

        x = self.net(x)


        x += shortcut
        return self.activation(x)

# main bottle stack

class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample
            #layer_fmap_size = fmap_size
            layer_fmap_size = (fmap_size[0] // (2 if downsample and not is_first else 1),fmap_size[1] // (2 if downsample and not is_first else 1))
            #layer_fmap_size = fmap_size[1] // (2 if downsample and not is_first else 1)
            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size[0] and w == self.fmap_size[1], f'height and width of feature map must match the fmap_size given at init {self.fmap_size}'
        return self.net(x)



class SE_fz(nn.Module):
    def __init__(self, in_channels, med_channels):
        super(SE_fz, self).__init__()

        self.average = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels,med_channels)
        self.bn1 = nn.BatchNorm1d(med_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(med_channels,in_channels)
        self.sg = nn.Sigmoid()
    
    def forward(self,input):
        x = input
        x = self.average(input)
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = self.fc1(x)
        x= self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sg(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        out = torch.mul(input,x)
        return out

def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 288, 512).cuda(0)
    thermal = torch.randn(num_minibatch, 2, 288, 512).cuda(0)
    rtf_net = AMFNet(9).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    output = rtf_net(input)
    print("output size:",output.size())

if __name__ == '__main__':
    unit_test()