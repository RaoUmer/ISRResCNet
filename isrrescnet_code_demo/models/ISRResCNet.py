import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)

class BasicBlock(nn.Module):
    """
    Residual BasicBlock 
    """
    def __init__(self, inplanes, planes, stride=1, weightnorm=None, shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.relu1 = nn.PReLU(num_parameters=planes,init=0.1)
        self.relu2 = nn.PReLU(num_parameters=planes, init=0.1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

    def forward(self, x):
        out = self.relu1(x)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv1(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        out = self.relu2(out)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv2(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        return out

class L2Proj(nn.Module):
    """
    L2Proj layer
    source link: https://github.com/cig-skoltech/deep_demosaick/blob/master/l2proj.py
    """
    def __init__(self):
        super(L2Proj, self).__init__()

    def forward(self, x, stdn, alpha):
        if x.is_cuda:
            x_size = torch.cuda.FloatTensor(1).fill_(x.shape[1] * x.shape[2] * x.shape[3])
        else:
            x_size = torch.Tensor([x.shape[1] * x.shape[2] * x.shape[3]])
        numX = torch.sqrt(x_size-1)
        if x.is_cuda:
            epsilon = torch.cuda.FloatTensor(x.shape[0],1,1,1).fill_(1) * (torch.exp(alpha) * stdn * numX)[:,None,None,None]
        else:
            epsilon = torch.zeros(x.size(0),1,1,1).fill_(1) * (torch.exp(alpha) *  stdn * numX)[:,None,None,None]
        x_resized = x.view(x.shape[0], -1)
        x_norm = torch.norm(x_resized, 2, dim=1).reshape(x.size(0),1,1,1)
        max_norm = torch.max(x_norm, epsilon)
        result = x * (epsilon / max_norm)
        
        return result

class FeedbackBlock(nn.Module):
    '''
    Feedback Resnet block
    '''
    
    def __init__(self, in_chs, out_chs, block, weightnorm=True):
        super(FeedbackBlock, self).__init__()
        
        self.compress_in = nn.Conv2d(2*in_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=True)
        self.prelu = nn.PReLU(num_parameters=128, init=0.1)
        
        self.RBs = block
        
        if weightnorm:
            self.compress_in = weight_norm(self.compress_in)
            
        self.last_hidden = None
        self.should_reset = True
        
    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(self.prelu(x))
        out = self.RBs(x)
        self.last_hidden = out
        
        return out
    
    def reset_state(self):
        self.should_reset = True

class ResFBNet(nn.Module):
    """
    Residual Convolutional Feed-back Net
    """
    def __init__(self, depth, num_steps=4, num_features=64, color=True, weightnorm=True):
        self.inplanes = num_features
        super(ResFBNet, self).__init__()
        
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.num_steps = num_steps
        self.conv = nn.Conv2d(in_channels, num_features, kernel_size=5, stride=1, padding=0,
                               bias=True)
        if weightnorm:
            self.conv = weight_norm(self.conv)

        # inntermediate layers
        self.resblocks = self._make_layer(BasicBlock, num_features, depth)
        
        # Feedback Net
        self.fbblock = FeedbackBlock(in_chs=num_features, out_chs=num_features, block=self.resblocks)
        
        self.conv_out = nn.ConvTranspose2d(num_features, in_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)

        self.l2proj = L2Proj()
        
        self.zeromean()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        return nn.Sequential(*layers)

    def zeromean(self):
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)
    
    def _reset_state(self):
        self.fbblock.reset_state()

    def forward(self, x, stdn, alpha):
        self.zeromean()
        self._reset_state()
        out = F.pad(x,(2,2,2,2),'reflect')
        out = self.conv(out)
        for _ in range(self.num_steps):
            out = self.fbblock(out)
        out = self.conv_out(out)
        out = self.l2proj(out, stdn, alpha)
        return out

class ISRResCNet(torch.nn.Module):
    """
    Iterative SR Residual Convolutional Network (ISRResCNet).
    """
    def __init__(self, model, max_iter=10, sigma_max=2, sigma_min=0):
        super(ISRResCNet, self).__init__()
        self.model = model
        self.max_iter = max_iter
        self.alpha =  nn.Parameter(torch.Tensor(np.linspace(np.log(sigma_max),np.log(sigma_min), max_iter)))
        iterations = np.arange(self.max_iter)
        iterations[0] = 1
        iterations = np.log(iterations / (iterations+3))
        w = nn.Parameter(torch.Tensor(iterations)) # initialize as in Boyd Proximal Algorithms
        self.w = w

    def forward(self, xcur, xpre, p, k):
        if k > 0:
            wk = self.w[k]

        if k > 0:
            yk = xcur + torch.exp(wk) * (xcur-xpre) # extrapolation step
        else:
            yk = xcur

        xpre = xcur
        net_input = yk - p.energy_grad(yk)
        noise_sigma = p.L
        xcur = (net_input - self.model(net_input, noise_sigma, self.alpha[k])) # residual approach of model
        xcur = xcur.clamp(0, 255) # clamp to ensure correctness of representation
        return xcur, xpre

    def forward_all_iter(self, p, init, noise_estimation, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
        xcur = p.y
        
        if init:
            xcur = p.initialize()
        
        if noise_estimation:
            p.estimate_noise()

        xpre = 0
        for i in range(max_iter):
            xcur, xpre = self.forward(xcur, xpre, p, i)

        return xcur
    

if __name__ == "__main__":    
    input = torch.randn(2,3,50,50).type(torch.FloatTensor).cuda()
    print('input:', input.shape)
    
    stdn = torch.tensor([0.01]*2).type(torch.FloatTensor).cuda()
    alpha = torch.FloatTensor([2.]).cuda()
    stdn = 15.
    
    model = ResFBNet(depth=5)
    #model = ISRResCNet(model, max_iter=10, sigma_max=2, sigma_min=1)
    model = model.cuda()
    
    print(model)
    
    s = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Number of model params: %d' % s)
    
    output = model(input, stdn, alpha)
    print('output:', output.shape)
    