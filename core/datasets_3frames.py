import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from PIL import Image

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from torchvision.utils import save_image

from .utils import flow_viz
import cv2
from .utils.utils import coords_grid, bilinear_sampler

# from utils import frame_utils
# from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
# from torchvision.utils import save_image

# from utils import flow_viz
# import cv2
# from utils.utils import coords_grid, bilinear_sampler


import copy

import os
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

from typing import Tuple

PROJECT_ROOT_DIR = '/home/ethan/Documents/Niloofar/Projects/VideoFlow/outputRes'
        
def save_fig(
        fig_id: plt.figure,
        fig_name: str, 
        fig_dir: str = '', 
        tight_layout: bool = True
):
    # create the directory if it does not exist
    img_dir = os.path.join(PROJECT_ROOT_DIR, "images", fig_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    path = os.path.join(img_dir, fig_name + ".png")
    print("Saving figure", fig_name)
    if tight_layout:
        fig_id.tight_layout()
    fig_id.savefig(path, format='png', dpi=300)


def get_video_names(
    metadata: dict): #-> tuple[str, str] :
    video_name = metadata['video_name'].numpy().decode('UTF-8')
    print("Video: ", video_name)
    video_type = metadata['video_type'].numpy().decode('UTF-8')
    print("Video type: ", video_type)
    return video_name, video_type


def get_scale_offset(
    metadata: dict, 
    imgdata: str = 'forward_flow',
): #-> tuple[float, float] : 
    i_range = metadata[imgdata + '_range'].numpy()
    print("{0} range {1} to {2}".format(imgdata,
                                        i_range[0],
                                        i_range[1]))
    i_scale =  ( i_range[1] - i_range[0] ) / 65535.0
    i_offset = i_range[0]
    return i_scale, i_offset


def flow_quiver(
    ax : plt.Axes,
    flow : np.ndarray,
    x_mesh : np.ndarray = np.empty(1),
    y_mesh : np.ndarray = np.empty(1),        
    n_arrows: int = 32,
    img_size: int = 256
) :
    if (x_mesh.shape[0] < math.floor(img_size/n_arrows) or
            y_mesh.shape[0] < math.floor(img_size/n_arrows)) :
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, img_size, n_arrows),
                           np.linspace(0, img_size, n_arrows))

    # quiver has origin at lower left corner, offsets are x=col, y=-row
    res_quiver = ax.quiver(
        x_mesh, -y_mesh,
        flow[0: img_size: int(img_size / n_arrows),
         0: img_size: int(img_size / n_arrows),
         1],
        -flow[0: img_size: int(img_size / n_arrows),
         0: img_size: int(img_size / n_arrows),
         0],
         angles='xy')    
    return res_quiver, x_mesh, y_mesh


def find_occurences( 
        asset_ids: list,
        name: str = 'bar' 
):# -> list :
    res = list()
    for i, aid in enumerate(asset_ids):
        if name == aid.numpy().decode('UTF-8'):
            res.append(i)
    return res


def get_bar_mask(
    segmentation: np.ndarray,
    bar_ids: list
) -> np.ndarray:
    mask = np.ones(segmentation.shape)
    if len(bar_ids) != 0:
        for id in bar_ids:
            # add 1 because background id is 0
            mask = np.logical_and(mask, (segmentation != id+1))
    return mask


def get_fg_bg_mask( 
            segmentation : np.ndarray,
            mask : np.ndarray 
): #-> tuple[ np.ndarray, np.ndarray ]:
    bg_mask = np.logical_and((segmentation == 0), mask)  # background id is 0
    fg_mask = np.logical_and((segmentation > 0), mask)
    return fg_mask, bg_mask

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, oneside=False, reverse_rate=0.3):
        self.augmentor = None
        self.sparse = sparse
        self.oneside = oneside
        self.reverse_rate = reverse_rate
        print("[reverse_rate is {}]".format(self.reverse_rate))


        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        #print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img3 = frame_utils.read_gen(self.image_list[index][2])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img3 = np.array(img3).astype(np.uint8)[..., :3]
            
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
            
            return torch.stack([img1, img2, img3]), self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid1 = valid2 = None

        if self.oneside:
            if self.sparse:
                flow1, valid1 = frame_utils.readFlowKITTI(self.flow_list[index])
                flow2 = copy.deepcopy(flow1)
                valid2 = copy.deepcopy(valid1) * 0
            else:
                flow1 = frame_utils.read_gen(self.flow_list[index])
                flow2 = copy.deepcopy(flow1) * 0 + 10000

        else:
            flow1 = frame_utils.read_gen(self.flow_list[index][0])
            flow2 = frame_utils.read_gen(self.flow_list[index][1])


        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])
        
        flow1 = np.array(flow1).astype(np.float32)
        flow2 = np.array(flow2).astype(np.float32)
        
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img3 = np.array(img3).astype(np.uint8)
        
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
            img3 = np.tile(img3[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            img3 = img3[..., :3]
            
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, img3, flow1, flow2, valid1, valid2 = self.augmentor(img1, img2, img3, flow1, flow2, valid1, valid2)
            else:
                img1, img2, img3, flow1, flow2 = self.augmentor(img1, img2, img3, flow1, flow2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
        
        
        
        flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
        flow1 = torch.from_numpy(np.stack([(flow1)[1,:,:],(flow1)[0,:,:]], axis=0))
        flow2 = torch.from_numpy(-flow2).permute(2, 0, 1).float()
        flow2 = torch.from_numpy(np.stack([flow2[1,:,:],(flow2)[0,:,:]], axis=0))

        if valid1 is not None and valid2 is not None:
            valid1 = torch.from_numpy(valid1)
            valid2 = torch.from_numpy(valid2) * 0 # sparse must be oneside
        else:
            valid1 = (flow1[0].abs() < 1000) & (flow1[1].abs() < 1000)
            valid2 = (flow2[0].abs() < 1000) & (flow2[1].abs() < 1000)
        
        if np.random.rand() < self.reverse_rate:
            return torch.stack([img3, img2, img1]), torch.stack([flow2, flow1]), torch.stack([valid2.float(), valid1.float()])
        else:
            return torch.stack([img1, img2, img3]), torch.stack([flow1, flow2]), torch.stack([valid1.float(), valid2.float()])

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        self.image_list = []
        with open("./flow_datasets/flying_things_three_frames/flyingthings_"+dstype+"_png.txt") as f:
            images = f.readlines()
            for img1, img2, img3 in zip(images[0::3], images[1::3], images[2::3]):
                self.image_list.append([root+img1.strip(), root+img2.strip(), root+img3.strip()])
        self.flow_list = []
        with open("./flow_datasets/flying_things_three_frames/flyingthings_"+dstype+"_pfm.txt") as f:
            flows = f.readlines()
            for flow1, flow2 in zip(flows[0::2], flows[1::2]):
                self.flow_list.append([root+flow1.strip(), root+flow2.strip()])

class Kubric(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/ethan/Documents/Niloofar/Projects/VideoFlow/datasets/kubric', dstype='clean', reverse_rate=0.3):
        super(Kubric, self).__init__(aug_params, oneside=False, reverse_rate=reverse_rate)
        # #this dataset contains tf records for each video samples
        # if split == 'training': 
        #     foldernamefull = '/home/ethan/tensorflow_datasets/flow_data_set_builder/'
        #     foldername = 'flow_data_set_builder/' #/home/niloofarhp/tensorflow_datasets/
        # else:
        #     foldernamefull = '/home/ethan/tensorflow_datasets/flow_data_set_builder/'
        #     foldername = 'flow_data_set_builder/'   
        # for vid_sample in os.listdir(foldernamefull):
        #     if vid_sample == '.config':# or ('bar' in vid_sample) :
        #         continue
        #     flow_data_set = tfds.load(foldername+vid_sample, with_info=True)[0]
        #     m_iter = iter(flow_data_set['train'])
        #     #print('len of the videos:', len(flow_data_set['train']))
        #     for i in range(len(flow_data_set['train'])): 
        #         #extract imgs and flow files 
        #         train_data = next(m_iter)
        #         video_name = train_data['metadata']['video_name'].numpy().decode('UTF-8')
        #         video_type = train_data['metadata']['video_type'].numpy().decode('UTF-8')
        #         if 'bar' in video_type:
        #             print('skipping videos containing bar:', video_type)
        #             continue
        #         forward_flow_range = train_data['metadata']['forward_flow_range'].numpy()
        #         f_scale =  (forward_flow_range[1] - forward_flow_range[0] ) / 65535.0
        #         f_offset = forward_flow_range[0]
        #         bf_scale, bf_offset = get_scale_offset( train_data['metadata'],imgdata='backward_flow' )
        #         bar_ids = find_occurences(train_data['instances']['asset_id'])
        #         num_frames = int(train_data['metadata']['num_frames'])
        #         if not (os.path.exists(root+'/'+split+'/clean/'+video_type)):
        #             path = os.path.join(root+'/'+split+'/clean/',video_type)
        #             os.mkdir(path)
        #         if not (os.path.exists(root+'/'+split+'/flow/'+video_type)):
        #             path = os.path.join(root+'/'+split+'/flow/',video_type)
        #             os.mkdir(path)
                                    
        #         if not (os.path.exists(root+'/'+split+'/clean/'+video_type+'/'+video_name)):
        #             path = os.path.join(root+'/'+split+'/clean/',video_type+'/'+video_name)
        #             os.mkdir(path)
        #         if not (os.path.exists(root+'/'+split+'/flow/'+video_type+'/'+video_name)):
        #             os.mkdir(os.path.join(root+'/'+split+'/flow/',video_type+'/'+video_name))
                    
        #         img_path = root+'/'+split+'/clean/'+video_type+'/'+video_name
        #         flow_path = root+'/'+split+'/flow/'+video_type+'/'+video_name
        #         for i in range(num_frames - 1):
        #             img1 = train_data['video'][i, :, :, :]
        #             img2 = train_data['video'][i+1, :, :, :]
        #             forward_flow = train_data['forward_flow'][i, :, :, :].numpy()
        #             forward_flow = f_scale * forward_flow + f_offset
        #             segmentation = train_data['segmentations'][i,:,:,:].numpy()
        #             mask = get_bar_mask( segmentation, bar_ids )
        #             #forward_flow = forward_flow * mask
        #             backward_flow = train_data['backward_flow'][i, :, :, :].numpy()
        #             backward_flow = bf_scale * backward_flow + bf_offset
        #             backward_flow = backward_flow 
                    
        #             if not(os.path.exists(img_path+'/frame_{:02d}.png'.format(i))):
        #                 img = Image.fromarray(img1.numpy())
        #                 img.save(img_path+'/frame_{:02d}.png'.format(i))
        #             if not (os.path.exists(flow_path+'/frame_{:02d}.flo'.format(i))):
        #                 frame_utils.writeFlow(flow_path+'/frame_{:02d}.flo'.format(i), forward_flow)
        #             if not (os.path.exists(flow_path+'/bframe_{:02d}.flo'.format(i))):
        #                 frame_utils.writeFlow(flow_path+'/bframe_{:02d}.flo'.format(i), backward_flow)                        
        #             #if not(os.path.exists(flow_path+'/frame_{:02d}.npy'.format(i))):
        #                 #np.save(flow_path+'/frame_{:02d}.npy'.format(i), mask)
                        

        # flow_root = osp.join(root, split, 'flow')
        # image_root = osp.join(root, split, 'clean')
        # imgFile = open("/home/ethan/Documents/Niloofar/Projects/VideoFlow/datasets/kubric_images_png.txt", "a")
        # floFile = open("/home/ethan/Documents/Niloofar/Projects/VideoFlow/datasets/kubric_images_bfflo.txt", "a")
        # for scene in os.listdir(image_root):
        #     for sub_scene in os.listdir(image_root+'/'+scene):
                
        #         image_list = sorted(glob(osp.join(image_root, scene, sub_scene, '*.png')))
        #         flow_list = sorted(glob(osp.join(flow_root, scene, sub_scene, '*.flo')))
        #         bflow_list = sorted(glob(osp.join(flow_root, scene, sub_scene, 'b*.flo')))
        #         for i in range(len(image_list)-2):
        #             self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2]] ]
        #             imgFile.write(image_list[i]+','+ image_list[i+1]+','+ image_list[i+2] +'\n')
        #             self.extra_info += [ (scene+'_'+sub_scene, i) ] # scene and frame_id
        #             self.flow_list += [[flow_list[i+1],bflow_list[i+1]]]
        #             floFile.write(flow_list[i+1]+','+bflow_list[i+1]+'\n')
        #         #self.flow_list += sorted(glob(osp.join(flow_root, scene, sub_scene, '*.flo')))
        #         #self.mask_list += sorted(glob(osp.join(flow_root, scene, sub_scene, '*.npy')))
        # imgFile.close()
        # floFile.close()     
        imgFile = open("/home/ethan/Documents/Niloofar/Projects/VideoFlow/datasets/kubric_images_png.txt", "r")
        floFile = open("/home/ethan/Documents/Niloofar/Projects/VideoFlow/datasets/kubric_images_bfflo.txt", "r")
        count = 0
        for line in imgFile:
            img1, img2 ,img3 = line.split(',')
            self.image_list += [[img1,img2,img3[:-1]]]
            self.extra_info += [ (img1.split('/')[-2]+'_'+img1.split('/')[-1], count) ] # scene and frame_id
            count +=1
            #print(line.strip())
        for line in floFile:
            flo1, flo2 = line.split(',')
            self.flow_list += [[flo1, flo2[:-1]]]
            #print(line.strip())                
                
        imgFile.close()
        floFile.close()
        print('3-frame image list lenght', len(self.image_list))
        print('bidirectional flow list lenght', len(self.flow_list))                
        print('') 
        
class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean', reverse_rate=0.3):
        super(MpiSintel, self).__init__(aug_params, oneside=True, reverse_rate=reverse_rate)

        self.image_list = []
        with open("./flow_datasets/sintel_three_frames/Sintel_"+dstype+"_png.txt") as f:
            images = f.readlines()
            for img1, img2, img3 in zip(images[0::3], images[1::3], images[2::3]):
                self.image_list.append([root+img1.strip(), root+img2.strip(), root+img3.strip()])
        
        self.flow_list = []
        with open("./flow_datasets/sintel_three_frames/Sintel_"+dstype+"_flo.txt") as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root+flow.strip())
        
        assert (len(self.image_list) == len(self.flow_list))

        self.extra_info = []
        with open("./flow_datasets/sintel_three_frames/Sintel_"+dstype+"_extra_info.txt") as f:
            info = f.readlines()
            for scene, id in zip(info[0::2], info[1::2]):
                self.extra_info.append((scene.strip(), int(id.strip())))

class MpiSintel_submission(FlowDataset):
    def __init__(self, aug_params=None, split='test', root='datasets/Sintel', dstype='clean', reverse_rate=-1):
        super(MpiSintel_submission, self).__init__(aug_params, oneside=True, reverse_rate=-1)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                if i==0:
                    self.image_list += [ [image_list[i], image_list[i], image_list[i+1]] ]
                else:
                    self.image_list += [ [image_list[i-1], image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True, oneside=True)

        self.image_list = []
        with open("./flow_datasets/hd1k_three_frames/hd1k_image.txt") as f:
            images = f.readlines()
            for img1, img2, img3 in zip(images[0::3], images[1::3], images[2::3]):
                self.image_list.append([root+img1.strip(), root+img2.strip(), root+img3.strip()])
        self.flow_list = []
        with open("./flow_datasets/hd1k_three_frames/hd1k_flo.txt") as f:
            flows = f.readlines()
            for flow in flows:
                self.flow_list.append(root+flow.strip())

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True, oneside=True)
        if split == 'testing':
            self.is_test = True

        self.image_list = []
        with open("./flow_datasets/KITTI/KITTI_{}_image.txt".format(split)) as f:
            images = f.readlines()
            for img1, img2 in zip(images[0::2], images[1::2]):
                self.image_list.append([root+img1.strip().replace("_10", "_09").replace("KITTI", "KITTI-full"), root+img1.strip(), root+img2.strip()])

        self.extra_info = []
        with open("./flow_datasets/KITTI/KITTI_{}_extra_info.txt".format(split)) as f:
            info = f.readlines()
            for id in info:
                self.extra_info.append([id.strip()])

        if split == "training":
            self.flow_list = []
            with open("./flow_datasets/KITTI/KITTI_{}_flow.txt".format(split)) as f:
                flow = f.readlines()
                for flo in flow:
                    self.flow_list.append(root+flo.strip())
        
        print(self.image_list[:10])
        print(self.flow_list[:10])
        

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
   
    if args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset
    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
        train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')
    elif args.stage == 'kubric':
        aug_params = None #{'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Kubric(aug_params, split='training')        
        print(len(train_dataset))
        train_size = int(0.7 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        indices = []
        train_indices = []
        for i in range(int(len(train_dataset)/23)):
            k = i*23
            for j in range(k,k+24):
                if i % 2 == 0:
                    train_indices.append(j)
                else:
                    indices.append(j) #train_indices.append(j)
        print(len(train_indices))
        train_indices = train_indices[:]
        print(len(indices))
        indices = indices[:]
        #temp_indices = 
        test_indices = indices[len(indices)//2:]
        indices = indices[:len(indices)//2]            
        train_dataset_new = torch.utils.data.Subset(train_dataset,(train_indices))
        val_dataset = torch.utils.data.Subset(train_dataset,(indices))
        test_dataset = torch.utils.data.Subset(train_dataset,(test_indices))
        #train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size, test_size])
        print('Training with %d image pairs' % len(train_dataset_new))
        print('Validating with %d image pairs' % len(val_dataset))
        print('Testing with %d image pairs' % len(test_dataset))
        train_loader = data.DataLoader(train_dataset_new, batch_size=args.batch_size,shuffle=True, drop_last=True)#, pin_memory=True,  num_workers=2)
        
        return train_loader, train_dataset_new, val_dataset, test_dataset       

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True,drop_last=True)# num_workers=args.batch_size*2, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

import argparse
import sys
sys.path.append('/home/ethan/Documents/Niloofar/Projects/VideoFlow')
import configs
from core.utils.misc import process_cfg
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='BOFTrainKubric', help="name your experiment")
    parser.add_argument('--stage', default='kubric') 
    parser.add_argument('--validation', type=str, nargs='+')

    args = parser.parse_args()

    if args.stage == 'things':
        from configs.things import get_cfg    
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'kubric':
        from configs.kubric import get_cfg
    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)        
    fetch_dataloader(cfg)

if __name__ == "__main__":
    main()
