# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import random
from glob import glob
import os.path as osp
import cv2 as cv
from PIL import Image
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from torchvision.utils import save_image

import tensorflow_datasets as tfds
import tensorflow as tf

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
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
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
        self.mask_list = []
    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        mask = np.load(self.mask_list[index])
        mask = np.array(mask).astype(np.bool8)
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow)#.permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[:,:,0].abs() < 1000) & (flow[:,:,1].abs() < 1000)

        return img1, img2, flow, mask, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        
class KubricTfds(FlowDataset):
    def __init__(self, aug_params=None, split='training', 
                 root='/home/niloofarhp/Documents/Projects/floweval/datasets/KubricTestBar',
                   dstype='clean'):
        super(KubricTfds, self).__init__(aug_params)
        #this dataset contains tf records for each video samples
        # if split == 'training': 
        #     foldernamefull = '/home/niloofarhp/tensorflow_datasets/flow_data_set_builder/'
        #     foldername = 'flow_data_set_builder/' #/home/niloofarhp/tensorflow_datasets/
        # else:
        #     foldernamefull = '/home/niloofarhp/tensorflow_datasets/flow_data_set_builder/'
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
        #         forward_flow_range = train_data['metadata']['forward_flow_range'].numpy()
        #         f_scale =  ( forward_flow_range[1] - forward_flow_range[0] ) / 65535.0
        #         f_offset = forward_flow_range[0]
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
        #             forward_flow = forward_flow * mask
                    
        #             if not(os.path.exists(img_path+'/frame_{:02d}.png'.format(i))):
        #                 img = Image.fromarray(img1.numpy())
        #                 img.save(img_path+'/frame_{:02d}.png'.format(i))
        #                 #cv.imwrite(img_path+'/frame_{:02d}.png'.format(i),img1.numpy())
        #             if not(os.path.exists(img_path+'/frame_{:02d}.png'.format(i+1))):
        #                 img = Image.fromarray(img2.numpy())
        #                 img.save(img_path+'/frame_{:02d}.png'.format(i+1))
        #                 #cv.imwrite(img_path+'/frame_{:02d}.png'.format(i+1),img2.numpy())
        #             if not (os.path.exists(flow_path+'/frame_{:02d}.flo'.format(i))):
        #                 frame_utils.writeFlow(flow_path+'/frame_{:02d}.flo'.format(i), forward_flow)
        #             if not(os.path.exists(flow_path+'/frame_{:02d}.npy'.format(i))):
        #                 np.save(flow_path+'/frame_{:02d}.npy'.format(i), mask)
                        

        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, 'clean')

        #if split == 'test':
        #    self.is_test = True

        for scene in os.listdir(image_root):
            if ('rotation' in scene) and (split == 'training'): #('bar' in scene): #
                continue
            for sub_scene in os.listdir(image_root+'/'+scene):
                image_list = sorted(glob(osp.join(image_root, scene, sub_scene, '*.png')))
                for i in range(len(image_list)-1):
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    self.extra_info += [ (scene+'_'+sub_scene, i) ] # scene and frame_id

                self.flow_list += sorted(glob(osp.join(flow_root, scene, sub_scene, '*.flo')))
                self.mask_list += sorted(glob(osp.join(flow_root, scene, sub_scene, '*.npy')))
        print('')        

class KubricTfds2():
    def __init__(self, aug_params=None, split='training', root='datasets/KubricTestNew', dstype='clean'):
        super(KubricTfds2, self).__init__(aug_params)

        mainPath = "/home/niloofarhp/tensorflow_datasets/flow_data_set_builder"
        self.flow_data_set = tfds.load('flow_data_set_builder/rotation_static_bar',split=tfds.Split.TRAIN)#, shuffle_files=True)
        for folders in os.listdir(mainPath):
            if folders == '.config' or folders == 'rotation_static_bar' or ('bar' in folders):
                continue
            self.flow_data_set = self.flow_data_set.concatenate(tfds.load('flow_data_set_builder/'+ folders,split=tfds.Split.TRAIN))# ,shuffle_files=True))
        self.flow_data_set.shuffle(128)
        #self.image_list = [0]*23 * len(self.flow_data_set)
        #self.flow_list = [0]*23 * len(self.flow_data_set)
                
        self.batchCount = 0
        self.video_nums = len(self.flow_data_set)
        print("number of the videos: ", self.video_nums)
        self.traindata = iter(self.flow_data_set)
    def __getitem__(self, index):
        #print(index)
        #self.flow_data_set.cache()
        imgs1 = []
        imgs2 = []
        flows = []
        index = index % 23
        data = next(self.traindata)
        forward_flow_range = data['metadata']['forward_flow_range'].numpy()
        f_scale =  ( forward_flow_range[1] - forward_flow_range[0] ) / 65535.0
        f_offset = forward_flow_range[0]
        flow = data['forward_flow'][index,:,:,:]
        img1 = data['video'][index,:,:,:]
        img2 = data['video'][index+1,:,:,:]
        img1 = torch.from_numpy(img1.numpy()).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2.numpy()).permute(2, 0, 1).float()
        forward_flow = data['forward_flow'][index, :, :, :].numpy()
        flow = f_scale * forward_flow + f_offset
        flow = torch.from_numpy(flow).permute(2, 0, 1) #tf.cast(flow,tf.float32)
        mask = torch.ones_like(flow)
        valid = None
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)   
        return img1, img2, flow, mask, valid.float()
    
class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/niloofarhp/Documents/Projects/floweval/datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    val_dataset = None
    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = None# {'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}#'crop_size': args.image_size, 
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
            
        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things
    
    elif args.stage == 'kubric':
        aug_params = None
        train_dataset = KubricTfds(aug_params, split='training', dstype='final')
        test_dataset = KubricTfds(aug_params, split='testing', dstype='final')
    
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')


    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    indices = []
    train_indices = []
    for i in range(int(len(train_dataset)/23)):
        k = i*23
        for j in range(k,k+23):
            if i % 4 == 0:
                indices.append(j)
            else:
                train_indices.append(j)
    train_indices = train_indices[:3600]
    indices = indices[:1200]            
    train_dataset_new = torch.utils.data.Subset(train_dataset,(train_indices))
    val_dataset = torch.utils.data.Subset(train_dataset,(indices))
    #train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,[train_size, test_size])
    print('Training with %d image pairs' % len(train_dataset_new))
    print('Validating with %d image pairs' % len(val_dataset))
    train_loader = data.DataLoader(train_dataset_new, batch_size=args.batch_size, 
         shuffle=True, drop_last=True)#, pin_memory=True,  num_workers=2)
    
    return train_loader, train_dataset_new, val_dataset, test_dataset

