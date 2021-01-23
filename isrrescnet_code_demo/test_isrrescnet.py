import os.path as osp
import glob
import cv2
import numpy as np
import torch
from models.ISRResCNet import ResFBNet, ISRResCNet
from modules.problems import Super_Resolution
from utils import timer
from collections import OrderedDict

# test settings
model_path = 'trained_nets/isrrescnet_x4.pth'  # path of trained ISRResCNet model
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*' # testset LR images path
upscale_factor = 4 # upscaling factor
num_iter_steps = 10 # number of iterative steps


# loading Network
model = ResFBNet(depth=5)
model = ISRResCNet(model, max_iter=10, sigma_max=2, sigma_min=1)
states = torch.load(model_path)
model.load_state_dict(states['model_state_dict'])
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

test_results = OrderedDict()
test_results['time'] = []
idx = 0
for path_lr in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path_lr))[0]
    print('Img:', idx, base)
    
    # read images: LR
    img_lr = cv2.imread(path_lr, cv2.IMREAD_COLOR)
    img_LR = torch.from_numpy(np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img_LR.unsqueeze(0)
    img_LR = img_LR.to(device)
    
    # initially upscale the LR image
    p = Super_Resolution(img_LR, scale=upscale_factor, mode='bicubic')
    p.cuda_()
    
    # testing
    t = timer()
    t.tic()   
    with torch.no_grad():
        outputs = model.forward_all_iter(p, init=True, noise_estimation=True, max_iter=num_iter_steps)
        if isinstance(outputs, list):
            output_SR = outputs[-1]
        else:
            output_SR = outputs
            
    end_time = t.toc()
    output_sr = output_SR.data.squeeze().float().cpu().clamp_(0, 255).numpy()
    output_sr = np.transpose(output_sr[[2, 1, 0], :, :], (1, 2, 0)) 
    
    test_results['time'].append(end_time)
    print('{:->4d}--> {:>10s}, time: {:.4f} sec.'.format(idx, base, end_time))
    
    # save images            
    cv2.imwrite('sr_results/{:s}.png'.format(base), output_sr)
    
    del img_LR, img_lr
    del  output_SR, output_sr
    torch.cuda.empty_cache()
    #break

avg_time = sum(test_results['time']) / len(test_results['time'])
print('Avg. Time:{:.4f}'.format(avg_time))
