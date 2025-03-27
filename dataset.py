import os
import glob
import random
import math
import pickle
import cv2
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],0).repeat_interleave(d[1], 1)
    
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def build_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
    ])

def build_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
    ])
    
class IXIDataset(Dataset):
    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        with open('/home/lalith/Latent_SB/Preprocessed_Data/train_dict.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)

        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.image_paths_t1 = list(loaded_dict['t1'].values())
        self.image_paths_t2 = list(loaded_dict['t2'].values())
        self.mask_paths = list(loaded_dict['mask'].values())
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/bubbly/*.jpg"))

    def __len__(self):
        return len(self.image_paths_t1)

    def rescale(self, np_array):
        np_array = np.expand_dims(np_array, 0)
        np_array = (np_array * 2) - 1
        return np_array 
    
    def augment_image(self, image, gt_msk, anomaly_source_path):
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2GRAY)
        anomaly_source_img = cv2.resize(anomaly_source_img, 
                                        dsize=(self.resize_shape[1], self.resize_shape[0]), 
                                        interpolation = cv2.INTER_LINEAR)

        anomaly_img_augmented = anomaly_source_img
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        
        filter_msk = gt_msk * perlin_thr / 255.0
        img_thr = anomaly_img_augmented.astype(np.float32) * filter_msk
        beta = torch.rand(1).numpy()[0] * 0.5
        augmented_image = image * (1 - filter_msk) + (1 - beta) * img_thr + beta * image * (filter_msk)
        
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32) 
        else:
            augmented_image = augmented_image.astype(np.float32)
            has_anomaly = 1.0
            if np.sum(filter_msk) == 0:
                has_anomaly=0.0
            return augmented_image, filter_msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, gt_mask_path, anomaly_source_path):
        # read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, 
                           dsize=(self.resize_shape[1], self.resize_shape[0]), 
                           interpolation = cv2.INTER_LINEAR)

        # read gt mask
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, dsize=(self.resize_shape[1], self.resize_shape[0]), 
                             interpolation = cv2.INTER_LINEAR)
        
        image = np.array(image  / 255.0).astype(np.float32)
        gt_mask = np.array(gt_mask / 255.0).astype(np.float32) 
        
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, gt_mask, anomaly_source_path)
        
        return self.rescale(image), self.rescale(gt_mask), self.rescale(augmented_image), self.rescale(anomaly_mask), has_anomaly

    def __getitem__(self, idx):
        cls_choice = random.randint(0, 1)
        if cls_choice == 0:
            inp_img_path = self.image_paths_t1[idx]
        else:
            inp_img_path = self.image_paths_t2[idx]
            
        gt_mask_path = self.mask_paths[idx]
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, gt_mask, augmented_image, anomaly_mask, has_anomaly = self.transform_image(inp_img_path, gt_mask_path, self.anomaly_source_paths[anomaly_source_idx])

        # sample = {'image': image, "gt_mask": gt_mask, "anomaly_mask": anomaly_mask, 'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
        
        image = torch.tensor(image)
        gt_mask = torch.tensor(gt_mask)
        augmented_image = torch.tensor(augmented_image)
        anomaly_mask = torch.tensor(anomaly_mask, dtype=torch.float32)
        has_anomaly = torch.tensor(has_anomaly, dtype=torch.long)
        cls_choice = torch.tensor(cls_choice, dtype=torch.long)
        
        image = torch.cat((image,image,image), dim=0)
        gt_mask = torch.cat((gt_mask,gt_mask,gt_mask), dim=0)
        augmented_image = torch.cat((augmented_image,augmented_image,augmented_image), dim=0)
        anomaly_mask = torch.cat((anomaly_mask,anomaly_mask,anomaly_mask), dim=0)
        
        return image, gt_mask, augmented_image, anomaly_mask, has_anomaly, cls_choice