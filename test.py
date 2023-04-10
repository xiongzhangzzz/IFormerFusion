import argparse
import os
import torch
import time
from models.InceptionTransformerFuse import InceptionTransformerFuse as net
from utils import utils_image as util
from dataloader import Dataset
from torch.utils.data import DataLoader
from utils import utils_option as option
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./Model/Infrared_Visible_Fusion/options/opt_221222_232036.json', help='Path to option JSON file.')
    parser.add_argument('--task', type=str, default='fusion', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--model_path', type=str,
                        default='./Model/Infrared_Visible_Fusion/models/model.pth')
    parser.add_argument('--root_path', type=str, default='./testimage/21pairs',
                        help='input test image root folder')
    parser.add_argument('--save_dir', default='./result/21pairs/')
    parser.add_argument('--dataset', type=str, default='MSRS',
                        help='input test image name')
    args = parser.parse_args()

    opt = option.parse(parser.parse_args().opt, is_train=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    test_set = Dataset(opt["datasets"]["test"])
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
    with torch.no_grad():
        model = define_model(args, opt)
        model.eval()
        model = model.to(device)
        for i, test_data in enumerate(test_loader):
            imgname = test_data['A_path'][0]
            img_a = test_data['A'].to(device)
            img_b = test_data['B'].to(device)
            start = time.time()
            # inference
            _, _, h_old, w_old = img_a.size()
            left = 0
            right = 0
            top = 0
            bot = 0
            if w_old//32*32 != w_old:
                lef_right = (w_old//32 + 1)*32 - w_old
                if lef_right%2 == 0.0:
                    left = int(lef_right/2)
                    right = int(lef_right/2)
                else:
                    left = int(lef_right / 2)
                    right = int(lef_right - left)
            if h_old//32*32 != h_old:
                top_bot = (h_old//32 + 1)*32 - h_old
                if top_bot%2 == 0.0:
                    top = int(top_bot/2)
                    bot = int(top_bot/2)
                else:
                    top = int(top_bot / 2)
                    bot = int(top_bot - top)
            
                reflection_padding = [left, right, top, bot]
                reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
                img_a = reflection_pad(img_a)
                img_b = reflection_pad(img_b)
            output = model.forward(img_a, img_b)
            output = output[:,:,top:top+h_old,left:left+w_old]
            output = output.detach()[0].float().cpu()
            end = time.time()
            output = util.tensor2uint(output)
            save_name = os.path.join(save_dir, os.path.basename(imgname))
            util.imsave(output, save_name)        
            print("[{}/{}]  Saving fused image to : {}, Processing time is {:4f} s".format(i+1, len(test_loader), save_name, end - start))

def define_model(args, opt):
    model = net(img_size=opt["netG"]["img_size"],
                depths=opt["netG"]["depths"],
                embed_dims=opt["netG"]["embed_dims"],
                num_heads=opt["netG"]["num_heads"],
                attention_heads=opt["netG"]["attention_heads"],
                use_layer_scale=opt["netG"]["use_layer_scale"],
                layer_scale_init_value=opt["netG"]["layer_scale_init_value"])
    param_key_g = 'params'
    model_path = args.model_path
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
        
    return model

if __name__ == '__main__':
    main()
 