'''
Code Acknowledgement:
https://github.com/cig-skoltech/deep_demosaick
'''

import numpy as np
import torch
import torch.nn.functional as F
from modules.wmad_estimator import Wmad_estimator

def downsampling(x, size=None, scale_factor=None, mode='bilinear'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
    # create coordinates
    h = torch.arange(0,size[0]) / (size[0]-1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1]-1) * 2 - 1
    # create grid
    grid = torch.zeros(size[0],size[1],2)
    grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
    grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda: grid = grid.cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)

def bicubic_interp_nd(input_, size, scale_factor=None, endpoint=False):
    """
    Args :
    input_ : Input tensor. Its shape should be
    [batch_size, height, width, channel].
    In this implementation, the shape should be fixed for speed.
    new_size : The output size [new_height, new_width]
    ref :
    http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
    """

    shape = input_.shape
    batch_size, channel, height, width = shape

    def _hermite(A, B, C, D, t):
        a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
        b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
        c = A * (-0.5) + C * 0.5
        d = B
        return a*(t**3) + b*(t**2) + c*t + d

    def tile(a, dim, n_tile):
        " Code from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2"
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)


    def meshgrid_4d(n_i, c_i, x_i, y_i):
        r""" Return a 5d meshgrid created using the combinations of the input
        Only works for 4d tensors
        """
        # tested and it is the same
        nn = n_i[:,None,None,None].expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        cc = c_i[None,:,None,None].expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        xx = x_i.view(-1,1).expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        yy = y_i.expand((n_i.shape[0], c_i.shape[0], x_i.shape[0], y_i.shape[0]))
        return torch.cat([nn[..., None], cc[..., None], xx[..., None], yy[..., None]], dim=4)

    def get_frac_array_4d(x_d, y_d, n, c):
        # tested and it is the same
        x = x_d.shape[0]
        y = y_d.shape[0]
        x_t = x_d[None,None,:,None]
        y_t = y_d[None,None,None,:]
        # tile tensor in each dimension
        x_t = tile(x_t, 0, n)
        x_t = tile(x_t, 1, c)
        x_t = tile(x_t, 3, y)
        # x_t.transpose_(2,3)
        y_t = tile(y_t, 0, n)
        y_t = tile(y_t, 1, c)
        y_t = tile(y_t, 2, x)
        # y_t.transpose_(2,3)
        return x_t, y_t

    def roll(tensor, shift_x, shift_y):
        # calculate the shifts of the input tensor
        shift_x = shift_x.clamp(0,height-1)
        shift_y = shift_y.clamp(0,width-1)
        p_matrix = tensor[:,:,shift_x]
        p_matrix = p_matrix[..., -tensor.shape[3]:] [...,shift_y]
        return p_matrix

    if size is None: size = (int(scale_factor*height), int(scale_factor*width))
    new_height = size[0]
    new_width  = size[1]
    n_i = torch.arange(batch_size, dtype=torch.int64)
    c_i = torch.arange(channel, dtype=torch.int64)

    if endpoint:
        x_f = torch.linspace(0., height, new_height)
    else:
        x_f = torch.linspace(0., height, new_height+1)[:-1]
    x_i = torch.floor(x_f).type(torch.int64)
    x_d = x_f - torch.floor(x_f)

    if endpoint:
        y_f = torch.linspace(0., width, new_width)
    else:
        y_f = torch.linspace(0., width, new_width+1)[:-1]

    y_i = torch.floor(y_f).type(torch.int64)
    y_d = y_f - torch.floor(y_f)

    grid = meshgrid_4d(n_i, c_i, x_i, y_i)
    x_t, y_t = get_frac_array_4d(x_d, y_d, batch_size, channel)

    if input_.is_cuda:
        x_t = x_t.cuda()
        y_t = y_t.cuda()
    # calculate f-1, f0, f+1, f+2 for y axis
    p_00 = roll(input_, x_i-1, y_i-1)
    p_10 = roll(input_, x_i-1, y_i+0)
    p_20 = roll(input_, x_i-1, y_i+1)
    p_30 = roll(input_, x_i-1, y_i+2)

    p_01 = roll(input_, x_i, y_i-1)
    p_11 = roll(input_, x_i, y_i+0)
    p_21 = roll(input_, x_i, y_i+1)
    p_31 = roll(input_, x_i, y_i+2)

    p_02 = roll(input_, x_i+1, y_i-1)
    p_12 = roll(input_, x_i+1, y_i+0)
    p_22 = roll(input_, x_i+1, y_i+1)
    p_32 = roll(input_, x_i+1, y_i+2)

    p_03 = roll(input_, x_i+2, y_i-1)
    p_13 = roll(input_, x_i+2, y_i+0)
    p_23 = roll(input_, x_i+2, y_i+1)
    p_33 = roll(input_, x_i+2, y_i+2)

    col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
    col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
    col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
    col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
    value = _hermite(col0, col1, col2, col3, y_t)

    return value

class Problem:
    def __init__(self, task_name):
        self.task_name = task_name
        self.L = torch.FloatTensor(1).fill_(1)
    def task(self):
        return self.task_name

    def energy_grad(self):
        pass

    def initialize(self):
        pass

class Super_Resolution(Problem):

    def __init__(self, y, scale, k=None, estimate_noise=False, mode='bilinear', task_name='super_resolution'):
        r""" Super Resolution Problem class
        y is the observed signal
        """
        Problem.__init__(self, task_name)
        assert(mode in ['bilinear','bicubic','kernel', 'normal']), "Mode can be either 'bilinear','bicubic' or 'kernel'"
        self.y = y
        self.scale = scale
        if estimate_noise:
            self.estimate_noise()
        self.mode = mode
        self.k = k
        self.L = self.L.repeat(y.shape[0])
    def energy_grad(self, x):
        r""" Returns the gradient 1/2||y-Mx||^2
        X is given as input
        """
        if self.mode == 'bilinear':
            # downsample and upsample x
            x = downsampling(x, size=None, scale_factor=1/self.scale, mode='bilinear')
            x = downsampling(x, size=None, scale_factor=self.scale, mode='bilinear')
            # upsample y
            y = downsampling(self.y, size=None, scale_factor=self.scale, mode='bilinear')
            
        elif self.mode == 'bicubic':
            # downsample and upsample x
            x = bicubic_interp_nd(x, size=None, scale_factor=1/self.scale)
            x = bicubic_interp_nd(x, size=None, scale_factor=self.scale)
            # upsample y
            y = bicubic_interp_nd(self.y, size=None, scale_factor=self.scale)
            
        elif self.mode == 'kernel':
            init_shape = x.shape
            # downsample and upsample x
            x = F.conv2d(x, self.k, stride=self.scale, padding=self.k.shape[2]//2, groups=3)
            x = F.conv_transpose2d(x, self.k, stride=self.scale, groups=3)
            # upsample y
            y = F.conv_transpose2d(self.y, self.k, stride=self.scale, groups=3)
            # crop to enforce same dimensionality in case of partial division
            x = x[:,:, :init_shape[2], :init_shape[3]]
            y = y[:,:, :init_shape[2], :init_shape[3]]
            
        elif self.mode == 'normal':
            init_shape = x.shape
            # downsample and upsample x
            res = torch.zeros_like(x)
            res[:,:,1::self.scale,1::self.scale] = x[:,:,1::self.scale,1::self.scale] - self.y
            return res
        return x - y

    def initialize(self):
        r""" Initialize with bilinear interpolation"""
        res = downsampling(self.y, size=None, scale_factor=self.scale, mode='bilinear')
        #res = bicubic_interp_nd(self.y/255, size=None, scale_factor=self.scale)*255
        #res.clamp_(0,255)
        #res = torch.zeros_like(res)
        #res[:,:,::self.scale,::self.scale] = self.y
        return res

    def estimate_noise(self):
        y = self.y
        if self.y.max() > 1:
            y  = self.y / 255
        L = Wmad_estimator()(y)
        self.L = L
        if self.y.max() > 1:
            self.L *= 255 # scale back to uint8 representation

    def cuda_(self):
        self.y = self.y.cuda()
        self.L = self.L.cuda()
        if  self.k is not None:
            self.k = self.k.cuda()
            
    def cpu_(self):
        self.y = self.y.cpu()
        self.L = self.L.cpu()
        if  self.k is not None:
            self.k = self.k.cpu()

if __name__ == "__main__":
    for i in range(5):
        x = torch.rand(2,3,80,80)
        y = torch.rand(2,3,40,40)
        #k = torch.rand(3,1,3,3)
        p = Super_Resolution(y, scale=2, k=None, estimate_noise=True, mode='bicubic')
        eng_grad = p.energy_grad(x)
        print(eng_grad.shape)
        print(p.L)
        
