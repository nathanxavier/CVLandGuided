# Standard
import os
import sys
import numpy as np
from PIL import Image
import pickle
import zipfile
import io
import math
# Pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import sparse
from threading import Lock
import matplotlib.pyplot as plt

def transformsCompose(size_sat, size_sat_default, size_gnd, size_gnd_default):
  transform_aerial_train = [
      transforms.Resize(size_sat_default, interpolation=3),
      transforms.Pad(0, padding_mode='edge'),
      transforms.ToTensor(),
  ]

  transform_label_train = [
      transforms.Resize(size_sat_default, interpolation=1),
      transforms.Pad(0, padding_mode='edge'),
      transforms.ToTensor(),
  ]

  transform_aerial_val = [
      transforms.Resize(size_sat_default, interpolation=3),
      transforms.ToTensor(),
  ]

  transform_street_train = [
      transforms.Resize(size_gnd_default, interpolation=3),
      transforms.Pad(0, padding_mode='edge'),
      transforms.CenterCrop((832, 3328)),		# Removes 50% of to minimize distortions -> 2022 Workman
      transforms.Resize(size_gnd),				# Resize Fator = 6.5
      transforms.ToTensor(),
  ]

  transform_street_val = [
      transforms.Resize(size=size_gnd_default, interpolation=3),  # Image.BICUBIC
      transforms.Resize(size_gnd),				# Resize Fator = 6.5
      transforms.ToTensor(),
  ]

  data_transforms = {
      "train": {
        "overhead": transforms.Compose(transform_aerial_train),
        "streetview": transforms.Compose(transform_street_train),
        "labels": transforms.Compose(transform_label_train),
      },
      "val": {
        "overhead": transforms.Compose(transform_aerial_val),
        "streetview": transforms.Compose(transform_street_val),
        "labels": transforms.Compose(transform_label_train),
      }
  }

  return data_transforms

class BrooklynQueens(Dataset):
  def __init__(self,
               args,
               labels=["overhead", "streetview", "height", "landcover", "landuse", "age", "function", "depth"],
               label_heatmap=False,
               remove_none=True,
               create_heatmaps=False,
               img_dict=None) -> None:
    super().__init__()
    self.root = args.root
    # self.zip_file = zipfile.ZipFile(self.root)
    self.lock = Lock()  #zipfile doesnt work with Threads

    self.labels = labels
    self.label_heatmap = label_heatmap
    self.aerial_shape = (3, 256, 256)
    self.label_shape = (1, 256, 256)
    self.street_shape = (3, 128, 512)

    self.size_sat = args.size_sat
    self.size_sat_default = args.size_sat_default
    self.size_gnd = args.size_gnd
    self.size_gnd_default = args.size_gnd_default
      
    img_transf = transformsCompose(self.size_sat, self.size_sat_default, self.size_gnd, self.size_gnd_default)

    self.transforms_train_aerial = img_transf["train"]["overhead"]
    self.transforms_train_street = img_transf["train"]["streetview"]
    self.transforms_train_label = img_transf["train"]["labels"]
    
    self.transforms_val_aerial = img_transf["val"]["overhead"]
    self.transforms_val_street = img_transf["val"]["streetview"]
    
    # Get Images and Labels - Brooklyng
    self.dataset = args.city

    # Street View Images
    self.streetview_dir = f"{self.root}{self.dataset}/streetview/"

    # with self.zip_file.open(f"{self.streetview_dir}images.txt") as myfile:
    #   street_fnames = [x.strip() for x in io.TextIOWrapper(myfile, encoding='utf-8', newline='')]
    with open(f"{self.streetview_dir}images.txt", 'r') as myfile:
      street_fnames = [x.strip() for x in myfile]
    self.street_fnames = street_fnames

    # with self.zip_file.open(f"{self.streetview_dir}locations.txt") as myfile:
    #   street_locs = [x.strip() for x in io.TextIOWrapper(myfile, encoding='utf-8', newline='')]
    with open(f"{self.streetview_dir}locations.txt") as myfile:
      street_locs = [x.strip() for x in myfile]
    self.street_locs = street_locs

    # Aerial Images
    self.aerial_dir = f"{self.root}{self.dataset}/overhead/"
    self.label_dir = f"{self.root}{self.dataset}/labels/"

    # with self.zip_file.open(f"{self.aerial_dir}images.txt") as myfile:
    #   aerial_fnames = [x.strip() for x in io.TextIOWrapper(myfile, encoding='utf-8', newline='')]
    with open(f"{self.aerial_dir}images.txt") as myfile:
      aerial_fnames = [x.strip() for x in myfile]
    self.aerial_fnames = aerial_fnames

    # with self.zip_file.open(f"{self.aerial_dir}locations.txt") as myfile:
    #   aerial_locs = [x.strip() for x in io.TextIOWrapper(myfile, encoding='utf-8', newline='')]
    with open(f"{self.aerial_dir}locations.txt") as myfile:
      aerial_locs = [x.strip() for x in myfile]
    self.aerial_locs = aerial_locs

    # with self.zip_file.open(f"{self.aerial_dir}bboxes.txt") as myfile:
    #   aerial_bboxes = [x.strip() for x in io.TextIOWrapper(myfile, encoding='utf-8', newline='')]
    with open(f"{self.aerial_dir}bboxes.txt") as myfile:
      aerial_bboxes = [x.strip() for x in myfile]
    self.aerial_bboxes = aerial_bboxes

    """
    Dictionary linking Aerial Images and Street Images
      In Brooklyn, it takes about 50 minutes      .·´¯`(⋟︿⋞)´¯`·.
      In Queens, it takes a little less     (⌬̀⌄⌬́)
    """
    if img_dict == None:
      images_dict = {}
      for img_index, aerial_img in enumerate(self.aerial_fnames):
        street_samples = self.get_box_images(self.aerial_bboxes[img_index])
        images_dict[aerial_img] = street_samples
        sys.stdout.write("\r{}/{}".format(str(img_index), str(len(self.aerial_fnames))))
        break

      self.images_dict = images_dict

      with open(f"datasets/{self.dataset}_dict.pkl", 'wb') as fp:
        pickle.dump(images_dict, fp)
        print('Image linked dictionary saved successfully')
    else: self.images_dict = img_dict

    # # Create Heatmaps
    # if(create_heatmaps): self.create_n_save_heatmaps_labels()
    if(remove_none): self.remove_empty()

  def get_image(self, label, file_path):
    if(len(file_path) == 0):
      return None
    
    if(label=="overhead"):
      img_path = os.path.join(f"{self.aerial_dir}images/{file_path}")
      # img = Image.open(self.zip_file.open(img_path))
      img = Image.open(img_path)
      return img
      
    if(label=="streetview"):
      img_path = os.path.join(f"{self.streetview_dir}images/{file_path}")
      # img = Image.open(self.zip_file.open(img_path))
      img = Image.open(img_path)
      return img

    elif(label == "height"):
      file_path = str.replace(file_path, 'jpg', 'npz')
      img_path = os.path.join(f"{self.label_dir}{label}/{file_path}")
      # img = sparse.load_npz(self.zip_file.open(img_path)).todense()
      img = sparse.load_npz(img_path).todense()
      img = Image.fromarray(np.uint8(img))
      return img
    
    elif(label == "landcover" or \
         label == "age" or \
         label == "function" or \
         label == "landuse" or \
         label == "depth"):
      file_path = str.replace(file_path, 'jpg', 'png')
      img_path = os.path.join(f"{self.label_dir}{label}/{file_path}")
      # img = Image.open(self.zip_file.open(img_path))
      img = Image.open(img_path)
      return img
    
    return None
  
  def __getitem__(self, idx):
    with self.lock: #zipfile doesnt work with Threads

      """Aerial Images and Labels"""
      aerial_path = list(self.images_dict.keys())[idx]

      aerial_img = self.get_image("overhead", aerial_path)
      aerial_img = self.transforms_train_aerial(aerial_img)

      if("height" in self.labels):
        height_img = self.get_image("height", aerial_path)
        height_img = self.transforms_train_label(height_img)
      else: height_img = torch.zeros(1)

      if("landcover" in self.labels):
        landco_img = self.get_image("landcover", aerial_path)
        landco_img = self.transforms_train_label(landco_img)

        """ Categorize Landcover:
        0.0039, 0.0118, 0.0078, 0.0157, 
        0.0196, 0.0235, 0.0275, 0.0314
        """
        unique_values = [0.0039, 0.0118, 0.0078, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314]
        min_unique = 0.0039/2

        for i, value in enumerate(unique_values):
          landco_img[landco_img<value +min_unique] = i+1
        landco_img -= 1
      else: landco_img = torch.zeros(1)

      if("landuse" in self.labels):
        landus_img = self.get_image("landuse", aerial_path)
        landus_img = self.transforms_train_label(landus_img)

        """ Categorize Landuse:
        0.0,    0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 
        0.0235, 0.0275, 0.0314, 0.0353, 0.0392, 0.0431, 0.0471
        """
        unique_values = [0.0, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314, 0.0353, 0.0392, 0.0431, 0.0471]
        min_unique = 0.0039/2

        for i, value in enumerate(unique_values):
          landus_img[landus_img<value +min_unique] = i+1
        landus_img -= 1
      else: landus_img = torch.zeros(1)

      if("age" in self.labels):
        age_img = self.get_image("age", aerial_path)
        age_img = self.transforms_train_label(age_img)

        """ Categorize Age:
        0.,     0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 
        0.0275, 0.0314, 0.0353, 0.0392, 0.0431, 0.0471, 0.0510, 0.0549
        """
        unique_values = [0., 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314, 0.0353, 0.0392, 0.0431, 0.0471, 0.0510, 0.0549]
        min_unique = 0.0039/2

        for i, value in enumerate(unique_values):
          age_img[age_img<value +min_unique] = i+1
        age_img -= 1
      else: age_img = torch.zeros(1)

      if("function" in self.labels):
        funct_img = self.get_image("function", aerial_path)
        funct_img = self.transforms_train_label(funct_img)

        """ Categorize Function:
        0.,     0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314, 0.0353, 0.0431, 
        0.0471, 0.0510, 0.0549, 0.0588, 0.0627, 0.0667, 0.0706, 0.0745, 0.0784, 0.0824, 0.0863, 
        0.0902, 0.0941, 0.0980, 0.1020, 0.1059, 0.1098, 0.1137, 0.1176, 0.1216, 0.1255, 0.1294, 
        0.1333, 0.1373, 0.1412, 0.1451, 0.1490, 0.1529, 0.1569, 0.1608, 0.1647, 0.1686, 0.1725, 
        0.1765, 0.1804, 0.1843, 0.1882, 0.1922, 0.1961, 0.2000, 0.2039, 0.2078, 0.2118, 0.2157, 
        0.2196, 0.2235, 0.2314, 0.2353, 0.2392, 0.2431, 0.2471, 0.2510, 0.2549, 0.2588, 0.2627, 
        0.2706, 0.2745, 0.2784, 0.2824, 0.2863, 0.2902, 0.2941, 0.2980, 0.3020, 0.3059, 0.3098, 
        0.3137, 0.3176, 0.3216, 0.3255, 0.3294, 0.3333, 0.3373, 0.3412, 0.3451, 0.3490, 0.3529, 
        0.3569, 0.3608, 0.3647, 0.3686, 0.3725, 0.3765, 0.3804, 0.3843, 0.3882, 0.3922, 0.3961, 
        0.4000, 0.4039, 0.4078, 0.4118, 0.4157, 0.4196, 0.4235, 0.4275, 0.4314, 0.4353, 0.4392, 
        0.4431, 0.4471, 0.4510, 0.4549, 0.4588, 0.4627, 0.4667, 0.4706, 0.4745, 0.4784, 0.4824, 
        0.4863, 0.4902, 0.4941, 0.4980, 0.5020, 0.5059, 0.5098, 0.5137, 0.5176, 0.5216, 0.5255, 
        0.5294, 0.5333, 0.5373, 0.5412, 0.5451, 0.5490, 0.5529, 0.5569, 0.5608, 0.5647, 0.5686, 
        0.5725, 0.5804, 0.5843, 0.5882, 0.5961, 0.6000, 0.6039, 0.6078, 0.6118, 0.6157, 0.6196, 
        0.6235, 0.6275, 0.6314, 0.6392, 0.6431, 0.6471, 0.6510, 0.6549, 0.6588, 0.6627, 0.6667, 
        0.6706, 0.6745, 0.6784, 0.6824, 0.6863, 0.6902, 0.6941, 0.7020, 0.7176, 0.7216, 0.7255, 
        0.7294, 0.7333, 0.7373, 0.7412, 0.7451, 0.7490, 0.7529, 0.7569, 0.7608, 0.7647, 0.7686, 
        0.7725, 0.7765, 0.7804, 0.7843, 0.7882, 0.7922, 0.7961, 0.8000, 0.8039, 0.8078, 0.8118
        """
        unique_values = [0., 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314, 0.0353, 0.0431, 0.0471, 0.0510, 0.0549, 0.0588, 0.0627, 0.0667, 0.0706, 0.0745, 0.0784, 0.0824, 0.0863, 0.0902, 0.0941, 0.0980, 0.1020, 0.1059, 0.1098, 0.1137, 0.1176, 0.1216, 0.1255, 0.1294, 0.1333, 0.1373, 0.1412, 0.1451, 0.1490, 0.1529, 0.1569, 0.1608, 0.1647, 0.1686, 0.1725, 0.1765, 0.1804, 0.1843, 0.1882, 0.1922, 0.1961, 0.2000, 0.2039, 0.2078, 0.2118, 0.2157, 0.2196, 0.2235, 0.2314, 0.2353, 0.2392, 0.2431, 0.2471, 0.2510, 0.2549, 0.2588, 0.2627, 0.2706, 0.2745, 0.2784, 0.2824, 0.2863, 0.2902, 0.2941, 0.2980, 0.3020, 0.3059, 0.3098, 0.3137, 0.3176, 0.3216, 0.3255, 0.3294, 0.3333, 0.3373, 0.3412, 0.3451, 0.3490, 0.3529, 0.3569, 0.3608, 0.3647, 0.3686, 0.3725, 0.3765, 0.3804, 0.3843, 0.3882, 0.3922, 0.3961, 0.4000, 0.4039, 0.4078, 0.4118, 0.4157, 0.4196, 0.4235, 0.4275, 0.4314, 0.4353, 0.4392, 0.4431, 0.4471, 0.4510, 0.4549, 0.4588, 0.4627, 0.4667, 0.4706, 0.4745, 0.4784, 0.4824, 0.4863, 0.4902, 0.4941, 0.4980, 0.5020, 0.5059, 0.5098, 0.5137, 0.5176, 0.5216, 0.5255, 0.5294, 0.5333, 0.5373, 0.5412, 0.5451, 0.5490, 0.5529, 0.5569, 0.5608, 0.5647, 0.5686, 0.5725, 0.5804, 0.5843, 0.5882, 0.5961, 0.6000, 0.6039, 0.6078, 0.6118, 0.6157, 0.6196, 0.6235, 0.6275, 0.6314, 0.6392, 0.6431, 0.6471, 0.6510, 0.6549, 0.6588, 0.6627, 0.6667, 0.6706, 0.6745, 0.6784, 0.6824, 0.6863, 0.6902, 0.6941, 0.7020, 0.7176, 0.7216, 0.7255, 0.7294, 0.7333, 0.7373, 0.7412, 0.7451, 0.7490, 0.7529, 0.7569, 0.7608, 0.7647, 0.7686, 0.7725, 0.7765, 0.7804, 0.7843, 0.7882, 0.7922, 0.7961, 0.8000, 0.8039, 0.8078, 0.8118]
        min_unique = 0.0039/2

        for i, value in enumerate(unique_values):
          funct_img[funct_img<value +min_unique] = i+1
        funct_img -= 1
      else: funct_img = torch.zeros(1)

      """Street Images and Maps"""
      if((not "streetview" in self.labels) or               # No Streetview in Label
         (list(self.images_dict.values())[idx] == [])):   # No Streetview on Aerial Box
        street_img = torch.zeros(self.street_shape)
        street_path = "-1,-1"
        depth_img = self.create_heatmap(self.aerial_bboxes[self.aerial_fnames.index(aerial_path)], street_path, landco_img)
        depth_img = self.transforms_train_label(depth_img)
        
        # NaN images
        if(torch.any(torch.isnan(depth_img))):
          depth_img = torch.zeros(self.aerial_shape)
      else:
        street_path = np.random.choice(list(self.images_dict.values())[idx], 1)[0]
      
        # Get imagens inside the bounding box of aerial image
        street_img = self.get_image("streetview", street_path)
        street_img = self.transforms_train_street(street_img)
        
        if("depth" in self.labels):
          depth_img = self.get_image("depth", street_path)
          depth_img = self.transforms_train_label(depth_img)
          depth_img = depth_img[0:1]
        else: depth_img = torch.zeros(1)

      # Create Heatmap
      if(self.label_heatmap):
        heatmp_img = self.create_heatmap(self.aerial_bboxes[self.aerial_fnames.index(aerial_path)], street_path)
        heatmp_img = self.transforms_train_label(heatmp_img)
      else: heatmp_img = torch.zeros(1)

    pos_x, pos_y, box_x_min, box_x_max, box_y_min, box_y_max = mapping(self.aerial_bboxes[self.aerial_fnames.index(aerial_path)], street_path, return_var=True)

    return idx, aerial_img, street_img, height_img, landco_img, landus_img, age_img, funct_img, depth_img, heatmp_img, pos_x, pos_y, box_x_min, box_x_max, box_y_min, box_y_max

  def __len__(self):
    return len(self.images_dict.keys())

  def create_heatmap(self, aerial_box, street_loc, label):
    if(street_loc == "-1,-1"):
      heat_map = label.clone().detach()
      heat_map = heat_map.reshape(256,256)

      heat_map[heat_map<.01] = 0
      heat_map[heat_map>.023] = 0
      heat_map[heat_map>.01] = 1

      heat_map = 1-heat_map

      heat_map = (heat_map -heat_map.min())/heat_map.max()
      heat_map = Image.fromarray(np.float32(heat_map))
      return heat_map
    else:
      center = mapping(aerial_box, street_loc, size_sat=self.size_sat, return_map=True)
      if center[0]>=256: center[0] = 255
      if center[1]>=256: center[1] = 255

      heat_map = self.get_heatmap(center)
      label_map = torch.tensor(label.reshape(256,256))

      label_map[label_map<.01] = 0
      label_map[label_map>.023] = 0
      label_map[label_map>0] = -1

      # Heatmap
      for i in range(256):
        for j in range(256):
          linesight = self.connect(center, [i,j])

          first_obj = torch.where(label_map[linesight[:,1], linesight[:,0]] == -1)[0]

          if(len(first_obj)>0):
            heat_map[linesight[first_obj[0]:,1], linesight[first_obj[0]:,0]] = 0
    return heat_map

  def remove_empty(self):
    test_list = list(self.images_dict.keys())
    for aerial_test in test_list:
      if(self.images_dict[aerial_test] == []):
        # print("Removing: {}".format(aerial_test))
        self.images_dict.pop(aerial_test)

    # Saving
    # with open(f"datasets/{self.dataset}_dict.pkl", 'wb') as fp:
    #   pickle.dump(self.images_dict, fp)
    #   print('Image linked dictionary saved successfully')

    # with open('aerial_bboxes.txt', 'w') as f:
    #   for line in self.aerial_bboxes:
    #     f.write(f"{line}\n")
    # with open('aerial_locations.txt', 'w') as f:
    #   for line in self.aerial_locs:
    #     f.write(f"{line}\n")

    # with open('aerial_images.txt', 'w') as f:
    #   for line in self.aerial_dict.values():
    #     f.write(f"{line}\n")

    # with open('street_locations.txt', 'w') as f:
    #   for line in self.street_locs:
    #     f.write(f"{line}\n")

    # with open('street_images.txt', 'w') as f:
    #   for line in self.street_dict.values():
    #     f.write(f"{line}\n")

  def get_heatmap(self, loc,std_dev=64):
    x,y = torch.meshgrid(torch.arange(256).float()-loc[1],
                         torch.arange(256).float()-loc[0])
    return torch.exp(-((x ** 2 + y ** 2) / (2 * std_dev ** 2)))
  
  def connect(self,p1,p2):
    ends = np.array([p1,p2],dtype=np.int16)
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    if d0 > d1: 
      ret= np.c_[np.linspace(ends[0, 0], ends[1, 0], d0+1, dtype=np.int32),
                    np.round(np.linspace(ends[0, 1], ends[1, 1], d0+1))
                    .astype(np.int32)]
    else:
      ret = np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1+1))
                    .astype(np.int32),
                    np.linspace(ends[0, 1], ends[1, 1], d1+1, dtype=np.int32)]
    return torch.tensor(ret,dtype=torch.int)

  def create_n_save_heatmaps_labels(self):
    label="depth"
    n = len(self.images_dict)
    for i, aerial_path in (list(enumerate(self.images_dict))[:]):
      for street_path in self.images_dict[aerial_path]:
        # Create folder
        folder1, folder2, _ = str.split(street_path, '/')
        folder_path = os.path.join(f"{self.label_dir}{label}/{folder1}/{folder2}")
        if(not os.path.exists(folder_path)):
          os.makedirs(folder_path)

        file_path = str.replace(street_path, 'jpg', 'png')
        img_path = os.path.join(f"{self.label_dir}{label}/{file_path}")

        # Create Heatmap
        if(not os.path.exists(img_path)):
          aerial_box = self.aerial_bboxes[self.aerial_fnames.index(aerial_path)]
          landco_img = self.get_image("landcover", aerial_path)
          landco_img = self.transforms_train_label(landco_img)

          heatmp_img = self.create_heatmap(aerial_box, street_path, landco_img)
          plt.imsave(img_path, heatmp_img, cmap="gray")

      sys.stdout.write("\r{}/{}".format(i, n))

  def get_box_images(self, box, cleaning=False):
    _, aerial_x_min, aerial_y_min, aerial_x_max, aerial_y_max = str.split(box, ',')
    aerial_x_min = float(aerial_x_min)
    aerial_y_min = float(aerial_y_min)
    aerial_x_max = float(aerial_x_max)
    aerial_y_max = float(aerial_y_max)

    pos_list = []
    for index, street_loc in enumerate(self.street_locs):
      street_x, street_y = str.split(street_loc, ',')
      street_x = float(street_x)
      street_y = float(street_y)

      if((min(aerial_x_min, aerial_x_max) <= street_x) and (street_x <= max(aerial_x_min, aerial_x_max)) and
         (min(aerial_y_min, aerial_y_max) <= street_y) and (street_y <= max(aerial_y_min, aerial_y_max))):
        pos_list.append(self.street_fnames[index])
        if(cleaning):
          return False
    if(cleaning):
      return True
    
    return pos_list

class Sampler(object):
  def __init__(self, data_source, batchsize=8, sample_num=1) -> None:
    self.data_len = len(data_source)
    self.batchsize = batchsize
    self.sample_num = sample_num
  
  def __iter__(self):
    list = np.arange(0,self.data_len)
    np.random.shuffle(list)
    nums = np.repeat(list,self.sample_num,axis=0)
    return iter(nums)
  
  def __len__(self):
    return len(self.data_source)

def train_collate_fn(batch):
    index, aerial_img, street_img, height_img, landco_img, landus_img, age_img, funct_img, depth_img, heatmp_img, pos_x, pos_y, box_x_min, box_x_max, box_y_min, box_y_max = zip(*batch)
    return_data = [torch.tensor(index,dtype=torch.int64)]

    # if(not aerial_img == 0):
    return_data += [torch.stack(aerial_img, dim=0)]
    return_data += [torch.tensor((box_x_min, box_x_max, box_y_min, box_y_max),dtype=torch.float).T]
      
    # if(not street_img == 0):
    return_data += [torch.stack(street_img, dim=0)]
    return_data += [torch.tensor((pos_x, pos_y),dtype=torch.float).T]

    if(not height_img == 0):  return_data += [torch.stack(height_img, dim=0)]
    if(not landco_img == 0):  return_data += [torch.stack(landco_img, dim=0)]
    if(not landus_img == 0):  return_data += [torch.stack(landus_img, dim=0)]
    if(not age_img == 0):     return_data += [torch.stack(age_img, dim=0)]
    if(not funct_img == 0):   return_data += [torch.stack(funct_img, dim=0)]
    if(not depth_img == 0):   return_data += [torch.stack(depth_img, dim=0)]
    if(not heatmp_img == 0):  return_data += [torch.stack(heatmp_img, dim=0)]

    return return_data

def train_val_dataset(dataset, val_split):
	train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
	datasets = {}
	datasets["train"] = Subset(dataset, train_idx)
	datasets["val"] = Subset(dataset, val_idx)
	return datasets

def mapping(box, pos, size_sat=(0,0), return_var=False, return_map=False):
  if(isinstance(box, str)):
    _, box_y1, box_x1, box_y2, box_x2 = str.split(box, ',')
    box_x1 = float(box_x1)
    box_y1 = float(box_y1)
    box_x2 = float(box_x2)
    box_y2 = float(box_y2)
  else:
    box_x1 = float(box[0])
    box_x2 = float(box[1])
    box_y1 = float(box[2])
    box_y2 = float(box[3])

  if(isinstance(pos, str)):
    if(len(str.split(pos, ','))==2):
      pos_y, pos_x = str.split(pos, ',')
    else:
      _, _, pos_y, pos_x, _ = str.split(str.replace(str.replace(pos, '/', '_'), '.jpg', '_'), '_')
    pos_x = float(pos_x)
    pos_y = float(pos_y)
  else:
    pos_x = float(pos[0])
    pos_y = float(pos[1])
  
  box_x_min = min(box_x1, box_x2)
  box_x_max = max(box_x1, box_x2)
  box_y_min = min(box_y1, box_y2)
  box_y_max = max(box_y1, box_y2)

  if(return_var):
    return pos_x, pos_y, box_x_min, box_x_max, box_y_min, box_y_max
  
  if(return_map):
    return [(size_sat[0] *(pos_x -box_x_min)/(box_x_max -box_x_min)), (-size_sat[1] *(pos_y -box_y_min)/(box_y_max -box_y_min) +size_sat[1])]

  return [(pos_x -box_x_min)/(box_x_max -box_x_min), (pos_y -box_y_min)/(box_y_max -box_y_min)]

def make_dataset(args,
                 labels=["overhead", "streetview", "height", "landcover", "landuse", "age", "function", "depth"],
                 label_heatmap=False,
                 remove_none=True, create_heatmaps=False, img_dict=None):
  dataset = BrooklynQueens(args,
                           labels=labels,
                           label_heatmap=label_heatmap,
                           remove_none=remove_none,
                           create_heatmaps=create_heatmaps,
                           img_dict=img_dict)
  aerial_bboxes = dataset.aerial_bboxes
  
  aerial_locs = len(dataset.images_dict.keys())
  street_locs = 0
  for samples in list(dataset.images_dict.values()):
    street_locs += len(samples)

  dataset_sizes = {"overhead": aerial_locs,
                  "streetview": street_locs}
  
  if(args.val_split > 0):
    dataset = train_val_dataset(dataset, val_split=args.val_split)
    samper = {x:Sampler(dataset[x],
                        args.batch_size,
                        sample_num=args.sample) for x in ["train", "val"]}
    dataloader = {x:DataLoader(dataset[x],
                               batch_size=args.batch_size,
                               num_workers=args.workers,
                               sampler=samper[x],
                               pin_memory=True,
                               collate_fn=train_collate_fn) for x in ["train", "val"]}
  else:
    samper = Sampler(dataset, args.batch_size, sample_num=args.sample)
    dataloader =DataLoader(dataset,
                           batch_size=args.batch_size,
                           num_workers=args.workers,
                           sampler=samper,
                           pin_memory=True,
                           collate_fn=train_collate_fn)

  return dataset, dataloader, samper, aerial_bboxes, aerial_locs, street_locs, dataset_sizes
