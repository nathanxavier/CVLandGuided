import numpy as np
import os
from functools import partial

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from .backbones.vit_pytorch import Multi_scale_transformer
# from .backbones.crossformer import CrossFormer
# from .backbones.crossformer_backbone_seg import CrossFormer
# from .backbones.Deit import DistilledVisionTransformer
# from .backbones.hubconf import UpsampledBackbone
# from .backbones.sep_vit import SepViT


class Create_Model(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    if args.dataset == 'vigor':
      args.size_sat = [320, 320]
      args.size_gnd = [320, 640]
    elif args.dataset == 'cvusa':
      args.size_sat = [256, 256]
      args.size_gnd = [112, 616]
    elif args.dataset == 'cvact':
      args.size_sat = [256, 256]
      args.size_gnd = [112, 616]
    elif args.dataset == 'brooklyn&queens':
      args.size_sat = (256, 256)
      args.size_sat_default = (256, 256)
      args.size_gnd = (256, 512)
      args.size_gnd_default = (1664, 3328)

    self.total_classes = args.total_classes
    self.segments = args.segments

    self.base = args.base
    if "featup" in self.base:
      print("Loading FeatUp")
      base_transformer = featup
    elif "crossformer" in self.base:
      print("Loading CrossFormer")
      base_transformer = crossformer_b
    elif "mst" in self.base:
      print("Loading Multi Scale Transformer")
      base_transformer = mst
    self.aerial = base_transformer(img_size=args.size_sat)
    self.street = base_transformer(img_size=args.size_gnd)
    self.segmentation = "segmentation"
    self.geolocation = "geoloc"

    self.transformer = build_transformer(args)

  def forward(self, x1, x2, l_x1=None, l_x2=None):
    segments = None
    heatmap = None
    y1_cls = None
    y2_cls = None
    
    # Segmentation Only
    if not x1 is None and x2 is None:
      if "mst" in self.base:
        y1_cls, y1 = self.aerial(x1)
      else:
        with torch.no_grad():
          y1 = self.aerial(x1)
      
      # First branch prediction
      if not("Seg" in self.base):
        segments = self.transformer(self.segmentation, y1)

      return segments

    elif not x1 is None and not x2 is None:
      if "mst" in self.base:
        y1_cls, y1 = self.aerial(x1)
        y2_cls, y2 = self.street(x2)
      else:
        with torch.no_grad():
          y1 = self.aerial(x1)
          y2 = self.street(x2)
      
      # Second branch prediction
      if "Seg" in self.base:
        l_x1 = nn.functional.one_hot(l_x1.to(torch.long), self.total_classes)
        aux_label = l_x1[:, :, :, [1,2,3,6,7]].sum(dim=-1)
        l_x1 = l_x1.permute(0, 3, 1, 2)[:, self.segments, :, :] 
        l_x1[:,-1,:,:] = aux_label

        heatmap = self.transformer(self.geolocation, y2, y1=y1, seg1=l_x1, y_cls=y2_cls)
      else:
        segments, heatmap = self.transformer(self.geolocation, y2, y1=y1, seg1=None, y_cls=y2_cls)

    return segments, heatmap

class build_transformer(nn.Module):
  def __init__(self,
               args,
               return_f=False) -> None:
    super().__init__()
    self.base = args.base
    self.gpu = args.gpu
    self.return_f = return_f
    self.input_dim = args.dim
    self.batch_size = args.batch_size
    self.patch_size = args.patch_size
    self.aerial_patch_size = args.aerial_patch_size
    self.street_patch_size = args.street_patch_size
    self.embed_dim = args.embed_dim
    self.aerial_embed_dim = args.aerial_embed_dim
    self.street_embed_dim = args.street_embed_dim
    self.heats = args.heat_blocks
    self.n_classes = args.classes

    self.aerial_size = args.size_sat
    self.street_size = args.size_gnd

    self.aerial_height = self.aerial_size[0]//self.aerial_patch_size
    self.aerial_width = self.aerial_size[1]//self.aerial_patch_size
    
    self.street_height = self.street_size[0]//self.street_patch_size
    self.street_width = self.street_size[1]//self.street_patch_size

    self.aerial_resize_patch_img = transforms.Resize((self.aerial_height, self.aerial_width))
    self.street_resize_patch_img = transforms.Resize((self.street_height, self.street_width))
    
    # self.globalclassifier = GlobalClassBlock(self.embed_dim, self.input_dim, .5,)

    # self.labeling = LabelingBlock(self.embed_dim, self.input_dim, self.n_classes, self.aerial_height, self.aerial_width)
    # self.depth = DepthBlock(self.embed_dim, self.input_dim, self.street_height, self.street_width, self.aerial_height, self.aerial_width)

    if "featup" in self.base:
      """Aerial Labeling"""
      labeling = []
      labeling += [nn.Conv2d(self.aerial_embed_dim,
                             self.input_dim, (1,1))]
      labeling += [nn.LayerNorm((self.input_dim,
                                 self.aerial_size[0], self.aerial_size[1]))]
      labeling += [nn.GELU()]
      labeling += [nn.Conv2d(self.input_dim,
                             self.n_classes, (1,1))]
      labeling += [nn.Dropout()]
      self.labeling = nn.Sequential(*labeling)

      self.aerial_minmax = nn.Sigmoid()
      
      self.aerial_feat = nn.Conv2d(self.aerial_embed_dim, self.input_dim, (1,1))
      self.street_feat = nn.Conv2d(self.street_embed_dim, self.input_dim, (1,1))
      
      self.x_cls = nn.AdaptiveAvgPool2d(1)
      
      self.cos = nn.CosineSimilarity()

      depth = []
      depth += [nn.Conv2d(self.input_dim +self.n_classes+1,
                          self.input_dim//2, (1,1))]
      depth += [nn.LayerNorm((self.input_dim//2, self.aerial_size[0], self.aerial_size[1]))]
      depth += [nn.GELU()]
      depth += [nn.Conv2d(self.input_dim//2, 1, (1,1))]
      depth += [nn.Dropout()]
      self.depth = nn.Sequential(*depth)

      self.street_minmax = nn.Sigmoid()

    elif "crossformer" in self.base:
      """Aerial Labeling - UNet"""
      self.up = nn.Upsample(scale_factor=2)

      decod_0 = []
      decod_0 += [nn.Conv2d(8*self.embed_dim, 4*self.embed_dim, (1,1))]
      decod_0 += [nn.LayerNorm(4*self.embed_dim)]
      decod_0 += [nn.GELU()]
      decod_0 += [nn.Conv2d(4*self.embed_dim, 4*self.embed_dim, (1,1))]
      decod_0 += [nn.LayerNorm(4*self.embed_dim)]
      decod_0 += [nn.GELU()]
      self.decod_0 = nn.Sequential(*decod_0)

      # 1nd Block
      decod_1 = []
      decod_1 += [nn.Conv2d(8*self.embed_dim, 4*self.embed_dim, (1,1))]
      decod_1 += [nn.LayerNorm(4*self.embed_dim)]
      decod_1 += [nn.GELU()]
      decod_1 += [nn.Conv2d(4*self.embed_dim, 2*self.embed_dim, (1,1))]
      decod_1 += [nn.LayerNorm(2*self.embed_dim)]
      decod_1 += [nn.GELU()]
      self.decod_1 = nn.Sequential(*decod_1)

      # 2nd Block
      decod_2 = []
      decod_2 += [nn.Conv2d(4*self.embed_dim, 2*self.embed_dim, (1,1))]
      decod_2 += [nn.LayerNorm(2*self.embed_dim)]
      decod_2 += [nn.GELU()]
      decod_2 += [nn.Conv2d(2*self.embed_dim, self.embed_dim, (1,1))]
      decod_2 += [nn.LayerNorm(self.embed_dim)]
      decod_2 += [nn.GELU()]
      self.decod_2 = nn.Sequential(*decod_2)

      # 3nd Block
      decod_3 = []
      decod_3 += [nn.Conv2d(2*self.embed_dim, self.embed_dim, (1,1))]
      decod_3 += [nn.LayerNorm(self.embed_dim)]
      decod_3 += [nn.GELU()]
      decod_3 += [nn.Conv2d(self.embed_dim, self.embed_dim, (1,1))]
      decod_3 += [nn.LayerNorm(self.embed_dim)]
      decod_3 += [nn.GELU()]
      self.decod_3 = nn.Sequential(*decod_3)

      # 4nd Block
      decod_4 = []
      decod_4 += [nn.Conv2d(self.embed_dim, self.embed_dim, (1,1))]
      decod_4 += [nn.LayerNorm(self.embed_dim)]
      decod_4 += [nn.GELU()]
      decod_4 += [nn.Conv2d(self.embed_dim, self.embed_dim, (1,1))]
      decod_4 += [nn.LayerNorm(self.embed_dim)]
      decod_4 += [nn.GELU()]
      self.decod_4 = nn.Sequential(*decod_4)

      # 5nd Block
      decod_5 = []
      decod_5 += [nn.Conv2d(self.embed_dim, self.n_classes, (1,1))]
      decod_5 += [nn.LayerNorm(self.n_classes)]
      decod_5 += [nn.GELU()]
      self.decod_5 = nn.Sequential(*decod_5)

      self.aerial_minmax = nn.Sigmoid()
    
      """Street"""
      self.aerial_feat0 = nn.Conv2d(1*self.aerial_embed_dim, self.input_dim, (1,1))
      self.aerial_feat1 = nn.Conv2d(2*self.aerial_embed_dim, self.input_dim, (1,1))
      self.aerial_feat2 = nn.Conv2d(4*self.aerial_embed_dim, self.input_dim, (1,1))
      self.aerial_feat3 = nn.Conv2d(8*self.aerial_embed_dim, self.input_dim, (1,1))

      self.street_feat0 = nn.Conv2d(1*self.street_embed_dim, self.input_dim, (1,1))
      self.street_feat1 = nn.Conv2d(2*self.street_embed_dim, self.input_dim, (1,1))
      self.street_feat2 = nn.Conv2d(4*self.street_embed_dim, self.input_dim, (1,1))
      self.street_feat3 = nn.Conv2d(8*self.street_embed_dim, self.input_dim, (1,1))

      self.x_cls0 = nn.AdaptiveAvgPool2d(1)
      self.x_cls1 = nn.AdaptiveAvgPool2d(1)
      self.x_cls2 = nn.AdaptiveAvgPool2d(1)
      self.x_cls3 = nn.AdaptiveAvgPool2d(1)

      self.cos = nn.CosineSimilarity()

      depth_0 = []
      depth_0 += [nn.Conv2d(2, 1, (1,1))]
      depth_0 += [nn.LayerNorm(1)]
      depth_0 += [nn.GELU()]
      self.depth_0 = nn.Sequential(*depth_0)

      depth_1 = []
      depth_1 += [nn.Conv2d(2, 1, (1,1))]
      depth_1 += [nn.LayerNorm(1)]
      depth_1 += [nn.GELU()]
      self.depth_1 = nn.Sequential(*depth_1)

      depth_2 = []
      depth_2 += [nn.Conv2d(2, 1, (1,1))]
      depth_2 += [nn.LayerNorm(1)]
      depth_2 += [nn.GELU()]
      self.depth_2 = nn.Sequential(*depth_2)

      depth_3 = []
      depth_3 += [nn.Conv2d(1, 1, (1,1))]
      depth_3 += [nn.LayerNorm(1)]
      depth_3 += [nn.GELU()]
      self.depth_3 = nn.Sequential(*depth_3)

      depth = []
      depth += [nn.Conv2d(self.n_classes+1, 1, (1,1))]
      depth += [nn.LayerNorm(1)]
      depth += [nn.GELU()]
      self.depth = nn.Sequential(*depth)

      self.street_minmax = nn.Sigmoid()

    elif "mst" in self.base:
      """Aerial Labeling - UNet"""
      # self.up = nn.Upsample(scale_factor=2)
      # self.up_decod_0 = nn.ConvTranspose2d(4*self.embed_dim, 4*self.embed_dim, 2, 2)
      # self.up_decod_1 = nn.ConvTranspose2d(2*self.embed_dim, 2*self.embed_dim, 2, 2)
      # self.up_decod_2 = nn.ConvTranspose2d(1*self.embed_dim, 1*self.embed_dim, 2, 2)
      # self.up_decod_3 = nn.ConvTranspose2d(1*self.embed_dim, 1*self.embed_dim, 2, 2)
      # self.up_decod_4 = nn.ConvTranspose2d(1*self.embed_dim, 1*self.embed_dim, 2, 2)

      # self.up_depth_0 = nn.ConvTranspose2d(4*self.embed_dim, 4*self.embed_dim, 2, 2)
      # self.up_depth_1 = nn.ConvTranspose2d(2*self.embed_dim, 2*self.embed_dim, 2, 2)
      # self.up_depth_2 = nn.ConvTranspose2d(1*self.embed_dim, 1*self.embed_dim, 2, 2)
      # self.up_depth_3 = nn.ConvTranspose2d(1*self.embed_dim, 1*self.embed_dim, 2, 2)
      # self.up_depth_4 = nn.ConvTranspose2d(1*self.embed_dim, 1*self.embed_dim, 2, 2)

      decod_0 = []
      decod_0 += [nn.ConvTranspose2d(8*self.embed_dim, 4*self.embed_dim, 2, 2)]
      decod_0 += [nn.Conv2d(4*self.embed_dim, 4*self.embed_dim, (1,1))]
      decod_0 += [nn.LayerNorm((4*self.embed_dim, self.aerial_size[0]//16, self.aerial_size[1]//16))]
      decod_0 += [nn.GELU()]
      decod_0 += [nn.Conv2d(4*self.embed_dim, 4*self.embed_dim, (1,1))]
      decod_0 += [nn.Dropout()]
      self.decod_0 = nn.Sequential(*decod_0)

      # 1nd Block
      decod_1 = []
      decod_1 += [nn.ConvTranspose2d(8*self.embed_dim, 4*self.embed_dim, 2, 2)]
      decod_1 += [nn.Conv2d(4*self.embed_dim, 2*self.embed_dim, (1,1))]
      decod_1 += [nn.LayerNorm((2*self.embed_dim, self.aerial_size[0]//8, self.aerial_size[1]//8))]
      decod_1 += [nn.GELU()]
      decod_1 += [nn.Conv2d(2*self.embed_dim, 2*self.embed_dim, (1,1))]
      decod_1 += [nn.Dropout()]
      self.decod_1 = nn.Sequential(*decod_1)

      # 2nd Block
      decod_2 = []
      decod_2 += [nn.ConvTranspose2d(4*self.embed_dim, 2*self.embed_dim, 2, 2)]
      decod_2 += [nn.Conv2d(2*self.embed_dim, self.embed_dim, (1,1))]
      decod_2 += [nn.LayerNorm((self.embed_dim, self.aerial_size[0]//4, self.aerial_size[1]//4))]
      decod_2 += [nn.GELU()]
      decod_2 += [nn.Conv2d(self.embed_dim, self.embed_dim, (1,1))]
      decod_2 += [nn.Dropout()]
      self.decod_2 = nn.Sequential(*decod_2)

      # 3nd Block
      decod_3 = []
      decod_3 += [nn.ConvTranspose2d(self.embed_dim, self.embed_dim//2, 2, 2)]
      decod_3 += [nn.Conv2d(self.embed_dim//2, self.embed_dim//4, (1,1))]
      decod_3 += [nn.LayerNorm((self.embed_dim//4, self.aerial_size[0]//2, self.aerial_size[1]//2))]
      decod_3 += [nn.GELU()]
      decod_3 += [nn.Conv2d(self.embed_dim//4, self.embed_dim//4, (1,1))]
      decod_3 += [nn.Dropout()]
      self.decod_3 = nn.Sequential(*decod_3)

      # 4nd Block
      decod_4 = []
      decod_4 += [nn.ConvTranspose2d(self.embed_dim//4, self.embed_dim//8, 2, 2)]
      decod_4 += [nn.Conv2d(self.embed_dim//8, self.embed_dim//16, (1,1))]
      decod_4 += [nn.LayerNorm((self.embed_dim//16, self.aerial_size[0]//1, self.aerial_size[1]//1))]
      decod_4 += [nn.GELU()]
      decod_4 += [nn.Conv2d(self.embed_dim//16, self.embed_dim//16, (1,1))]
      decod_4 += [nn.Dropout()]
      self.decod_4 = nn.Sequential(*decod_4)

      # 5nd Block
      decod_5 = []
      # decod_5 += [nn.ConvTranspose2d(self.embed_dim//16, self.embed_dim, 2, 2)]
      decod_5 += [nn.Conv2d(self.embed_dim//16, self.n_classes, (1,1))]
      decod_5 += [nn.LayerNorm((self.n_classes, self.aerial_size[0]//1, self.aerial_size[1]//1))]
      decod_5 += [nn.GELU()]
      decod_5 += [nn.Conv2d(self.n_classes, self.n_classes, (1,1))]
      decod_5 += [nn.Dropout()]
      self.decod_5 = nn.Sequential(*decod_5)

      self.aerial_minmax = nn.Sigmoid()
    
      """Street"""
      self.aerial_feat0 = nn.Conv2d(8*self.aerial_embed_dim,
                                    8*self.aerial_embed_dim, (1,1))
      self.aerial_feat1 = nn.Conv2d(4*self.aerial_embed_dim,
                                    4*self.aerial_embed_dim, (1,1))
      self.aerial_feat2 = nn.Conv2d(2*self.aerial_embed_dim,
                                    2*self.aerial_embed_dim, (1,1))
      # self.aerial_feat3 = nn.Conv2d(1*self.aerial_embed_dim, self.input_dim, (1,1))

      # self.street_feat0 = nn.Conv2d(4*self.street_embed_dim, self.input_dim, (1,1))
      # self.street_feat1 = nn.Conv2d(2*self.street_embed_dim, self.input_dim, (1,1))
      # self.street_feat2 = nn.Conv2d(1*self.street_embed_dim, self.input_dim, (1,1))
      self.street_feat0 = nn.Conv2d(8*self.street_embed_dim,
                                    8*self.street_embed_dim, (1,1))
      self.street_feat1 = nn.Conv2d(4*self.street_embed_dim,
                                    4*self.street_embed_dim, (1,1))
      self.street_feat2 = nn.Conv2d(2*self.street_embed_dim,
                                    2*self.aerial_embed_dim, (1,1))
      # self.street_feat3 = nn.Conv2d(1*self.street_embed_dim, self.input_dim, (1,1))

      # self.x_cls0 = nn.AdaptiveAvgPool2d(1)
      self.x_cls1 = nn.AdaptiveAvgPool2d(1)
      self.x_cls2 = nn.AdaptiveAvgPool2d(1)
      # self.x_cls3 = nn.AdaptiveAvgPool2d(1)

      self.cos = nn.CosineSimilarity()

      self.up_0 = nn.ConvTranspose2d(8*self.aerial_embed_dim +1,
                                     4*self.aerial_embed_dim, 2, 2)
      self.up_1 = nn.ConvTranspose2d(6*self.aerial_embed_dim +2*self.embed_dim +1,
                                     3*self.aerial_embed_dim +self.embed_dim, 2, 2)
      self.up_2 = nn.ConvTranspose2d(3*self.aerial_embed_dim//2 +7*self.embed_dim//2 +1,
                                     3*self.aerial_embed_dim//4 +7*self.embed_dim//4, 2, 2)
      self.up_3 = nn.ConvTranspose2d(3*self.aerial_embed_dim//8 +11*self.embed_dim//8,
                                     3*self.aerial_embed_dim//16 +11*self.embed_dim//16, 2, 2)
      self.up_4 = nn.ConvTranspose2d(3*self.aerial_embed_dim//32 +15*self.embed_dim//32,
                                     3*self.aerial_embed_dim//64 +15*self.embed_dim//64, 2, 2)
      self.up_5 = nn.ConvTranspose2d(3*self.aerial_embed_dim//32 +15*self.embed_dim//32,
                                     self.street_embed_dim//4, 2, 2)

      depth_0 = []
      depth_0 += [nn.Conv2d(4*self.aerial_embed_dim +4*self.embed_dim,
                            2*self.aerial_embed_dim +2*self.embed_dim, (1,1))]
      depth_0 += [nn.LayerNorm((2*self.aerial_embed_dim +2*self.embed_dim,
                                self.aerial_size[0]//16, self.aerial_size[1]//16))]
      depth_0 += [nn.GELU()]
      depth_0 += [nn.Conv2d(2*self.aerial_embed_dim +2*self.embed_dim,
                            2*self.aerial_embed_dim +2*self.embed_dim, (1,1))]
      depth_0 += [nn.Dropout()]
      self.depth_0 = nn.Sequential(*depth_0)

      depth_1 = []
      depth_1 += [nn.Conv2d(3*self.aerial_embed_dim +3*self.embed_dim,
                            3*self.aerial_embed_dim//2 +3*self.embed_dim//2, (1,1))]
      depth_1 += [nn.LayerNorm((3*self.aerial_embed_dim//2 +3*self.embed_dim//2,
                                self.aerial_size[0]//8, self.aerial_size[1]//8))]
      depth_1 += [nn.GELU()]
      depth_1 += [nn.Conv2d(3*self.aerial_embed_dim//2 +3*self.embed_dim//2,
                            3*self.aerial_embed_dim//2 +3*self.embed_dim//2, (1,1))]
      depth_1 += [nn.Dropout()]
      self.depth_1 = nn.Sequential(*depth_1)

      depth_2 = []
      depth_2 += [nn.Conv2d(3*self.aerial_embed_dim//4 +11*self.embed_dim//4,
                            3*self.aerial_embed_dim//8 +11*self.embed_dim//8, (1,1))]
      depth_2 += [nn.LayerNorm((3*self.aerial_embed_dim//8 +11*self.embed_dim//8,
                                self.aerial_size[0]//4, self.aerial_size[1]//4))]
      depth_2 += [nn.GELU()]
      depth_2 += [nn.Conv2d(3*self.aerial_embed_dim//8 +11*self.embed_dim//8,
                            3*self.aerial_embed_dim//8 +11*self.embed_dim//8, (1,1))]
      depth_2 += [nn.Dropout()]
      self.depth_2 = nn.Sequential(*depth_2)

      depth_3 = []
      depth_3 += [nn.Conv2d(3*self.aerial_embed_dim//16 +15*self.embed_dim//16,
                            3*self.aerial_embed_dim//32 +15*self.embed_dim//32, (1,1))]
      depth_3 += [nn.LayerNorm((3*self.aerial_embed_dim//32 +15*self.embed_dim//32,
                                self.aerial_size[0]//2, self.aerial_size[1]//2))]
      depth_3 += [nn.GELU()]
      depth_3 += [nn.Conv2d(3*self.aerial_embed_dim//32 +15*self.embed_dim//32,
                            3*self.aerial_embed_dim//32 +15*self.embed_dim//32, (1,1))]
      depth_3 += [nn.Dropout()]
      self.depth_3 = nn.Sequential(*depth_3)

      depth_4 = []
      depth_4 += [nn.Conv2d(3*self.aerial_embed_dim//64 +19*self.embed_dim//64,
                            3*self.aerial_embed_dim//128 +19*self.embed_dim//128, (1,1))]
      depth_4 += [nn.LayerNorm((3*self.aerial_embed_dim//128 +19*self.embed_dim//128,
                                 self.aerial_size[0]//1,self.aerial_size[1]//1))]
      depth_4 += [nn.GELU()]
      depth_4 += [nn.Conv2d(3*self.aerial_embed_dim//128 +19*self.embed_dim//128,
                            3*self.aerial_embed_dim//128 +19*self.embed_dim//128, (1,1))]
      depth_4 += [nn.Dropout()]
      self.depth_4 = nn.Sequential(*depth_4)

      # depth_5 = []
      # depth_5 += [nn.Conv2d(1, 100, (1,1))]
      # depth_5 += [nn.LayerNorm((100, self.aerial_size[0]//1, self.aerial_size[1]//1))]
      # depth_5 += [nn.GELU()]
      # depth_5 += [nn.Conv2d(100, 1, (1,1))]
      # depth_5 += [nn.Dropout()]
      # self.depth_5 = nn.Sequential(*depth_5)

      depth = []
      depth += [nn.Conv2d(3*self.aerial_embed_dim//128 +19*self.embed_dim//128 +self.n_classes,
                          10, (1,1))]
      depth += [nn.LayerNorm((10, self.aerial_size[0]//1, self.aerial_size[1]//1))]
      depth += [nn.GELU()]
      depth += [nn.Conv2d(10, 1, (1,1))]
      depth += [nn.Dropout()]
      self.depth = nn.Sequential(*depth)

      self.street_minmax = nn.Sigmoid()

  def forward(self, id, y, y1=None, seg1=None, y_cls=None):
    if(id=="segmentation"):
      if "featup" in self.base:
        map = self.labeling(y)
        map = self.aerial_minmax(map)

      elif "crossformer" in self.base:
        # UNet
        x = self.decod_0(y[3])
        x_up = self.up(x)
        x = torch.cat([x_up, y[2]], dim=1)

        x = self.decod_1(x)
        x_up = self.up(x)
        x = torch.cat([x_up, y[1]], dim=1)

        x = self.decod_2(x)
        x_up = self.up(x)
        x = torch.cat([x_up, y[0]], dim=1)

        x = self.decod_3(x)
        x = self.up(x)
        x = self.decod_4(x)
        x = self.up(x)
        map = self.decod_5(x)

        map = self.aerial_minmax(map)
      
      elif "mst" in self.base:
        x = self.decod_0(y[2].permute(0,2,1).reshape(y[2].size(0),
                                                     y[2].size(2),
                                                     int(y[2].size(1)**.5),
                                                     int(y[2].size(1)**.5)))
        # x = self.up_decod_0(x)
        x = torch.cat([x, y[1].permute(0,2,1).reshape(y[1].size(0),
                                                      y[1].size(2),
                                                      int(y[1].size(1)**.5),
                                                      int(y[1].size(1)**.5))], dim=1)
        x = self.decod_1(x)
        # x = self.up_decod_1(x)
        x = torch.cat([x, y[0].permute(0,2,1).reshape(y[0].size(0),
                                                         y[0].size(2),
                                                         int(y[0].size(1)**.5),
                                                         int(y[0].size(1)**.5))], dim=1)
        x = self.decod_2(x)
        # x = self.up_decod_2(x)

        x = self.decod_3(x)
        # x = self.up_decod_3(x)

        map = self.decod_4(x)
        # x = self.up_decod_4(x)

        # map = self.decod_5(map)

        map = self.aerial_minmax(map)
      
      return map
        
    elif(id=="geoloc"):
      if "featup" in self.base:
        """Segmentation Decoder"""
        seg = self.labeling(y1)
        aerial_segments = self.aerial_minmax(seg)

        """Heatmap Decoder"""
        aerial_feats = self.aerial_feat(y1)
        street_feats = self.street_feat(y)
        street_cls = self.x_cls(street_feats)
        cos_sim = self.cos(aerial_feats, street_cls)

        map = torch.cat([cos_sim.unsqueeze(1), aerial_feats, seg], dim=1)
        map = self.depth(map)
        map = self.street_minmax(map)
        
      elif "crossformer" in self.base:
        aerial_feats = self.aerial_feat3(y1[3])
        street_feats = self.x_cls3(self.street_feat3(y[3]))
        cos_sim = self.cos(aerial_feats, street_feats)
        map = self.depth_3(cos_sim.unsqueeze(1))
        map = self.up(map)

        aerial_feats = self.aerial_feat2(y1[2])
        street_feats = self.x_cls2(self.street_feat2(y[2]))
        cos_sim = self.cos(aerial_feats, street_feats)
        map = torch.cat([map, cos_sim.unsqueeze(1)], dim=1)
        map = self.depth_2(map)
        map = self.up(map)

        aerial_feats = self.aerial_feat1(y1[1])
        street_feats = self.x_cls1(self.street_feat1(y[1]))
        cos_sim = self.cos(aerial_feats, street_feats)
        map = torch.cat([map, cos_sim.unsqueeze(1)], dim=1)
        map = self.depth_1(map)
        map = self.up(map)

        aerial_feats = self.aerial_feat0(y1[0])
        street_feats = self.x_cls0(self.street_feat0(y[0]))
        cos_sim = self.cos(aerial_feats, street_feats)
        map = torch.cat([map, cos_sim.unsqueeze(1)], dim=1)
        map = self.depth_0(map)
        map = self.up(map)
        map = self.up(map)

        map = torch.cat([map, seg1], dim=1)
        map = self.depth(map)

      elif "mst" in self.base:
        """Segmentation Decoder"""
        seg_0 = self.decod_0(y1[2].permute(0,2,1).reshape(y1[2].size(0),
                                                          y1[2].size(2),
                                                          int(y1[2].size(1)**.5),
                                                          int(y1[2].size(1)**.5)))
        seg_1 = torch.cat([seg_0, y1[1].permute(0,2,1).reshape(y1[1].size(0),
                                                              y1[1].size(2),
                                                              int(y1[1].size(1)**.5),
                                                              int(y1[1].size(1)**.5))], dim=1)
        seg_1 = self.decod_1(seg_1)
        seg_2 = torch.cat([seg_1, y1[0].permute(0,2,1).reshape(y1[0].size(0),
                                                              y1[0].size(2),
                                                              int(y1[0].size(1)**.5),
                                                              int(y1[0].size(1)**.5))], dim=1)
        seg_2 = self.decod_2(seg_2)
        seg_3 = self.decod_3(seg_2)
        seg_4 = self.decod_4(seg_3)
        seg_5 = self.decod_5(seg_4)
        aerial_segments = self.aerial_minmax(seg_5)

        """Heatmap Decoder"""
        # Block 0
        aerial_img = y1[2].permute(0,2,1).reshape(y1[2].size(0),
                                                  y1[2].size(2),
                                                  int(y1[2].size(1)**.5),
                                                  int(y1[2].size(1)**.5))
        aerial_feats = self.aerial_feat0(aerial_img)
        street_feats = self.street_feat0(y_cls.unsqueeze(-1).unsqueeze(-1))
        cos_sim = self.cos(aerial_feats, street_feats)
        
        map = torch.cat([cos_sim.unsqueeze(1), aerial_feats], dim=1)
        map = self.up_0(map)
        map = torch.cat([map, seg_0], dim=1)
        map = self.depth_0(map)

        # Block 1
        aerial_img = y1[1].permute(0,2,1).reshape(y1[1].size(0),
                                                  y1[1].size(2),
                                                  int(y1[1].size(1)**.5),
                                                  int(y1[1].size(1)**.5))

        aerial_feats = self.aerial_feat1(aerial_img)
        street_img = y[1].permute(0,2,1).reshape(y[1].size(0),
                                                 y[1].size(2),
                                                 self.street_height,
                                                 self.street_width)
        street_feats = self.street_feat1(street_img)
        street_cls = self.x_cls1(street_feats)
        cos_sim = self.cos(aerial_feats, street_cls)

        map = torch.cat([map, cos_sim.unsqueeze(1), aerial_feats], dim=1)
        map = self.up_1(map)
        map = torch.cat([map, seg_1], dim=1)
        map = self.depth_1(map)

        # Block 2
        aerial_img = y1[0].permute(0,2,1).reshape(y1[0].size(0),
                                                  y1[0].size(2),
                                                  int(y1[0].size(1)**.5),
                                                  int(y1[0].size(1)**.5))

        aerial_feats = self.aerial_feat2(aerial_img)
        street_img = y[0].permute(0,2,1).reshape(y[0].size(0),
                                                 y[0].size(2),
                                                 2*self.street_height,
                                                 2*self.street_width)
        street_feats = self.street_feat2(street_img)
        street_cls = self.x_cls2(street_feats)
        cos_sim = self.cos(aerial_feats, street_cls)

        map = torch.cat([map, cos_sim.unsqueeze(1), aerial_feats], dim=1)
        map = self.up_2(map)
        map = torch.cat([map, seg_2], dim=1)
        map = self.depth_2(map)

        # Block 3
        map = self.up_3(map)
        map = torch.cat([map, seg_3], dim=1)
        map = self.depth_3(map)

        # Block 4
        map = self.up_4(map)
        map = torch.cat([map, seg_4], dim=1)
        map = self.depth_4(map)

        # Block 5
        map = torch.cat([map, seg_5], dim=1)
        map = self.depth(map)

        map = self.street_minmax(map)

      return aerial_segments, map

  def get_sorted_patch(self, patch_emb, add_global=False, otherbranch=False):
    map = torch.mean(patch_emb,dim=-1)
    patch_pos = torch.argsort(map, dim=1, descending=True)
    patch_emb_sort = [patch_emb[i, patch_pos[i], :] for i in range(patch_emb.size(0))]
    patch_emb_sort = torch.stack(patch_emb_sort, dim=0)

    return patch_emb_sort, patch_pos

  def part_classifier(self, x, cls_name='classifier_lpn'):
    part = {}
    predict = {}
    for i in range(self.heats):
        part[i] = x[i].view(x.size(1), x.size(2), -1)
        # part[i] = torch.squeeze(x[:,:,i])
        name = cls_name + str(i+1)
        c = getattr(self, name)
        predict[i] = c(part[i])
    y = []
    for i in range(self.heats):
        y.append(predict[i])
    if not self.training:
        # return torch.cat(y,dim=1)
        return torch.stack(y, dim=2)
    return y

class GlobalClassBlock(nn.Module):
  def __init__(self, aerial_embed_dim, street_embed_dim, input_dim, droprate=.5, GELU=False, bnorm=True, num_bottleneck=1024, linear=True, return_f = False):
    super(GlobalClassBlock, self).__init__()
    self.return_f = return_f
    self.aerial_linear = nn.Linear(aerial_embed_dim, num_bottleneck)
    self.street_linear = nn.Linear(street_embed_dim, num_bottleneck)

    add_block = []
    # if linear:
    #   add_block += [nn.Linear(embed_dim, num_bottleneck)]
    # else:
    #   num_bottleneck = embed_dim
    if bnorm:
      add_block += [nn.BatchNorm1d(num_bottleneck)]
    if GELU:
      add_block += [nn.LeakyGELU(0.1)]
    if droprate>0:
      add_block += [nn.Dropout(p=droprate)]
    add_block = nn.Sequential(*add_block)
    add_block.apply(weights_init_kaiming)

    classifier = []
    classifier += [nn.Linear(num_bottleneck, input_dim)]
    classifier = nn.Sequential(*classifier)
    classifier.apply(weights_init_classifier)

    self.add_block = add_block
    self.classifier = classifier
  
  def forward(self, id, x):
    if(id=="aerial"):   x = self.aerial_linear(x)
    elif(id=="street"): x = self.street_linear(x)

    x = self.add_block(x)
    if self.training:
      x = self.classifier(x)
    return x

class LabelingBlock(nn.Module):
  def __init__(self, aerial_embed_dim, input_dim, n_classes, height, width):
    super(LabelingBlock, self).__init__()
    # self.linear = nn.Linear(aerial_embed_dim, n_classes, (1,1))
    # self.height = height
    # self.width = width

    # add_block = []
    # add_block += [nn.ConvTranspose2d(input_dim, embed_dim, (32,32), 32)]
    # add_block += [nn.Dropout()]
    # add_block += [nn.Conv2d(embed_dim, n_classes, 1)]
    
    # """S. Zheng et al. (2020). Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"""
    # add_block = []
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    
    # # 1st Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # 2nd Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # 3rd Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # 4rd Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # # 5rd Block
    # # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # # input_dim = input_dim//2
    # # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # # add_block += [nn.LayerNorm(input_dim)]
    # # add_block += [nn.GELU()]
    # # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # # add_block += [nn.LayerNorm(input_dim)]
    # # add_block += [nn.GELU()]

    # # Final Block
    # add_block += [nn.Conv2d(input_dim, n_classes, (1,1))]
    # add_block += [nn.LayerNorm(n_classes)]
    # add_block += [nn.GELU()]

    # add_block = nn.Sequential(*add_block)
    # add_block.apply(weights_init_kaiming)

    # self.add_block = add_block
  
  def forward(self, x):
    x = x.permute(0,2,3,1)
    x = self.linear(x)

    return x

class DepthBlock(nn.Module):
  def __init__(self, street_embed_dim, input_dim, n_classes, street_height, street_width, aerial_height, aerial_width):
    super(DepthBlock, self).__init__()
    self.linear = nn.Linear(street_embed_dim, input_dim, (1,1))

    self.street_height = street_height
    self.street_width = street_width
    self.aerial_height = aerial_height
    self.aerial_width = aerial_width

    """Workman et al. (2022). Revisiting Near/Remote Sensing with Geospatial Attention"""
    grid_block = []
    grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    grid_block += [nn.LayerNorm(input_dim)]
    grid_block += [nn.GELU()]
    grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    grid_block += [nn.LayerNorm(input_dim)]
    grid_block += [nn.GELU()]

    # # 1st Block (Rect -> Square)
    # grid_block += [nn.ConvTranspose2d(input_dim, input_dim, (2,1), (2,1))]
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]
    
    # # 2st Block (Rect -> Square)
    # grid_block += [nn.ConvTranspose2d(input_dim, input_dim, (2,1), (2,1))]
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]
    
    # # 3st Block (Expand)
    # grid_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]
    
    # # 4st Block (Expand)
    # grid_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]
    # grid_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # grid_block += [nn.LayerNorm(input_dim)]
    # grid_block += [nn.GELU()]

    # Final Block
    grid_block += [nn.Conv2d(input_dim, 1, (1,1))]
    grid_block += [nn.LayerNorm(1)]
    grid_block += [nn.GELU()]

    grid_block += [nn.Sigmoid()]

    grid_block = nn.Sequential(*grid_block)
    grid_block.apply(weights_init_kaiming)

    self.grid_block = grid_block

    # """S. Zheng et al. (2020). Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"""
    # """Workman et al. (2022). Revisiting Near/Remote Sensing with Geospatial Attention"""
    # add_block = []
    # add_block += [nn.Conv2d(2*input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    
    # # 1st Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # 2nd Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # 3rd Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # 4rd Block
    # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # input_dim = input_dim//2
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]
    # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # add_block += [nn.LayerNorm(input_dim)]
    # add_block += [nn.GELU()]

    # # # 5rd Block
    # # add_block += [nn.ConvTranspose2d(input_dim, input_dim//2, (2,2), 2)]
    # # input_dim = input_dim//2
    # # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # # add_block += [nn.LayerNorm(input_dim)]
    # # add_block += [nn.GELU()]
    # # add_block += [nn.Conv2d(input_dim, input_dim, (1,1))]
    # # add_block += [nn.LayerNorm(input_dim)]
    # # add_block += [nn.GELU()]

    # # Final Block
    # add_block += [nn.Conv2d(input_dim, 1, (1,1))]
    # add_block += [nn.LayerNorm(1)]
    # add_block += [nn.GELU()]

    # add_block += [nn.Softmax()]

    # add_block = nn.Sequential(*add_block)
    # add_block.apply(weights_init_kaiming)

    # self.add_block = add_block
  
  def forward(self, x, sup_x):
    # Street -> Aerial
    patches = self.linear(x)
    x = patches.permute(0,2,1).reshape(x.shape[0], -1, self.street_height, self.street_width)
    x = self.grid_block(x)

    # sup_x = self.linear(sup_x)
    # sup_x = sup_x.permute(0,2,1).reshape(sup_x.shape[0], -1, self.aerial_height, self.aerial_width)
    # # print(sup_x.shape)

    # x = torch.cat([x, sup_x], dim=1)

    # x = self.add_block(x)

    return patches, x

class DepthClassBlock(nn.Module):
  def __init__(self, aerial_embed_dim, street_embed_dim, input_dim, aerial_height, aerial_width, num_bottleneck=1):
    super(DepthClassBlock, self).__init__()

    self.aerial_height = aerial_height
    self.aerial_width = aerial_width

    self.aerial_linear = nn.Linear(aerial_embed_dim, input_dim)
    self.street_linear = nn.Linear(street_embed_dim, input_dim)

    self.cos = nn.CosineSimilarity(dim=-1)
    self.sigmoid = nn.Sigmoid()

  
  def forward(self, x, sup_x):
    sup_x = self.aerial_linear(sup_x)
    x = self.street_linear(x)

    score = self.cos(x.unsqueeze(1), sup_x)
    score = self.sigmoid(score)

    # Street -> Aerial
    x = score.reshape(score.shape[0], -1, self.aerial_height, self.aerial_width)
    x = self.grid_block(x)

    return x

def weights_init_kaiming(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
    nn.init.constant_(m.bias, 0.0)

  elif classname.find('Conv') != -1:
    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)
  elif classname.find('BatchNorm') != -1:
    if m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.normal_(m.weight.data, std=0.001)
    nn.init.constant_(m.bias.data, 0.0)

@register_model
def featup(**kwargs):
    # model = torch.hub.load("mhamilton723/FeatUp", 'vit')
    model = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=True)
    return model

@register_model
def detr(**kwargs):
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    return model

@register_model
def mst(img_size=224,
        patch_size=[4, 8, 16, 32],
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=[8, 8, 8, 8],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2, 4], [2, 4]], **kwargs):
        
    model = Multi_scale_transformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                                    embed_dim=embed_dim, depths=depths, num_heads=num_heads, group_size=group_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                    attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                    ape=ape, patch_norm=patch_norm, use_checkpoint=use_checkpoint, merge_size=merge_size,
                                    **kwargs)
    return model

@register_model
def deit_small_distilled_patch16_224(pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, num_classes=num_classes, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        # for key in checkpoint["model"]:
        #     print(key)

        # resize the positional embedding
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
    return model

@register_model
def crossformer_b(img_size=224,
                  patch_size=[4, 8, 16, 32],
                  in_chans=3,
                  num_classes=1000,
                  embed_dim=96,
                  depths=[2, 2, 18, 2],
                  num_heads=[3, 6, 12, 24],
                  group_size=[7, 7, 7, 7],
                  crs_interval=[8, 4, 2, 1],
                  mlp_ratio=4., 
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  drop_path_rate=0.3,
                  norm_layer=nn.LayerNorm,
                  ape=False,
                  patch_norm=True,
                  use_checkpoint=False,
                  merge_size=[[2, 4], [2, 4], [2, 4]],
                  use_cpe=False,
                  group_type='constant',
                  pad_type=0,
                  no_mask=False,
                  adaptive_interval=False,
                  use_acl=False,
                  init_cfg=None, **kwargs):
    
    model = CrossFormer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                        embed_dim=embed_dim, depths=depths, num_heads=num_heads, group_size=group_size,
                        crs_interval=crs_interval, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                        norm_layer=norm_layer, ape=ape, patch_norm=patch_norm, use_checkpoint=use_checkpoint,
                        merge_size=merge_size, group_type=group_type, use_cpe=use_cpe, pad_type=pad_type,
                        no_mask=no_mask, adaptive_interval=adaptive_interval, use_acl=use_acl, init_cfg=init_cfg)
    
    checkpoint = torch.load("models/backbones/crossformer/fpn-80k-crossformer-b.pth",
                            weights_only=True)
    for key in list(checkpoint["state_dict"].keys()):
      if("decode_head" in key or \
        "auxiliary_head" in key or \
        "neck" in key):
        del checkpoint["state_dict"][key]
      else:
        checkpoint["state_dict"][key.replace('backbone.', '')] = checkpoint["state_dict"].pop(key)
      
    model.load_state_dict(checkpoint["state_dict"])
    
    return model

