import numpy as np
import os
from functools import partial

import torch
import torch.nn as nn
from torchvision import transforms

from timm.models import register_model

from .backbones.vit_pytorch import Multi_scale_transformer


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
      args.size_gnd = (128, 512)
      args.size_gnd_default = (1664, 3328)

    self.total_classes = args.total_classes
    self.segments = args.segments

    self.base = args.base
    if "featup" in self.base:
      print("Loading FeatUp")
      base_transformer = featup
    elif "mst" in self.base:
      print("Loading Multi Scale Transformer")
      base_transformer = mst
    self.aerial = base_transformer(img_size=(256,256))
    self.aerial_name = "aerial"
    self.street = base_transformer(img_size=(128,512))
    self.street_name = "street"

    self.transformer = build_transformer(args)

  def forward(self, x1, x2, l_x1=None, l_x2=None):
    heatmap1 = None
    heatmap2 = None
    y1_cls = None
    y2_cls = None
    if not x1 is None:
      if "mst" in self.base:
        y1_cls, y1 = self.aerial(x1)
      else:
        with torch.no_grad():
            y1 = self.aerial(x1)
      
      # First branch prediction
      if not("Seg" in self.base):
        heatmap1 = self.transformer(self.aerial_name, y1)

      if not x2 is None:
        if "mst" in self.base:
          y2_cls, y2 = self.street(x2)
        else:
          with torch.no_grad():
            y2 = self.street(x2)
        
        # Second branch prediction
        if "nSeg" in self.base:
          heatmap2 = self.transformer(self.street_name, y2, y1=y1, seg1=None, y_cls=y2_cls)
        elif "Seg" in self.base:
          l_x1 = nn.functional.one_hot(l_x1.to(torch.long), self.total_classes)
          aux_label = l_x1[:, :, :, [1,2,3,6,7]].sum(dim=-1)
          l_x1 = l_x1.permute(0, 3, 1, 2)[:, self.segments, :, :] 
          l_x1[:,-1,:,:] = aux_label

          heatmap2 = self.transformer(self.street_name, y2, y1=y1, seg1=l_x1, y_cls=y2_cls)
        else:
          heatmap2 = self.transformer(self.street_name, y2, y1=y1, seg1=heatmap1, y_cls=y2_cls)

    return heatmap1, heatmap2

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
    
    if "featup" in self.base:
      """Aerial Labeling"""
      labeling = []
      labeling += [nn.Conv2d(self.aerial_embed_dim, self.input_dim, (1,1))]
      labeling += [nn.LayerNorm((self.input_dim, self.aerial_size[0], self.aerial_size[1]))]
      # labeling += [nn.ReLU()]
      labeling += [nn.GELU()]
      # labeling += [nn.Sigmoid()]
      labeling += [nn.Conv2d(self.input_dim, self.n_classes, (1,1))]
      labeling += [nn.Dropout()]
      self.labeling = nn.Sequential(*labeling)

      self.aerial_minmax = nn.Sigmoid()
      
      self.aerial_feat = nn.Conv2d(self.aerial_embed_dim, self.input_dim, (1,1))
      self.street_feat = nn.Conv2d(self.street_embed_dim, self.input_dim, (1,1))
      
      self.x_cls = nn.AdaptiveAvgPool2d(1)
      
      self.cos = nn.CosineSimilarity()

      depth = []
      if "nSeg" in self.base:
        depth += [nn.Conv2d(1, 10, (1,1))]
      else:
        depth += [nn.Conv2d(self.n_classes+1, 10, (1,1))]
        
      depth += [nn.LayerNorm((10, self.aerial_size[0], self.aerial_size[1]))]
      # depth += [nn.ReLU()]
      depth += [nn.GELU()]
      # depth += [nn.Sigmoid()]
      depth += [nn.Conv2d(10, 1, (1,1))]
      depth += [nn.Dropout()]
      self.depth = nn.Sequential(*depth)

      self.street_minmax = nn.Sigmoid()

    elif "mst" in self.base:
      """Aerial Labeling - UNet"""
      self.up = nn.Upsample(scale_factor=2)

      decod_0 = []
      decod_0 += [nn.Conv2d(8*self.embed_dim, self.input_dim, (1,1))]
      decod_0 += [nn.LayerNorm((self.input_dim, self.aerial_size[0]//32, self.aerial_size[1]//32))]
      # decod_0 += [nn.ReLU()]
      decod_0 += [nn.GELU()]
      # decod_0 += [nn.Sigmoid()]
      decod_0 += [nn.Conv2d(self.input_dim, 4*self.embed_dim, (1,1))]
      decod_0 += [nn.Dropout()]
      self.decod_0 = nn.Sequential(*decod_0)

      # 1nd Block
      decod_1 = []
      decod_1 += [nn.Conv2d(8*self.embed_dim, self.input_dim//2, (1,1))]
      decod_1 += [nn.LayerNorm((self.input_dim//2, self.aerial_size[0]//16, self.aerial_size[1]//16))]
      # decod_1 += [nn.ReLU()]
      decod_1 += [nn.GELU()]
      # decod_1 += [nn.Sigmoid()]
      decod_1 += [nn.Conv2d(self.input_dim//2, 2*self.embed_dim, (1,1))]
      decod_1 += [nn.Dropout()]
      self.decod_1 = nn.Sequential(*decod_1)

      # 2nd Block
      decod_2 = []
      decod_2 += [nn.Conv2d(4*self.embed_dim, self.input_dim//4, (1,1))]
      decod_2 += [nn.LayerNorm((self.input_dim//4, self.aerial_size[0]//8, self.aerial_size[1]//8))]
      # decod_2 += [nn.ReLU()]
      decod_2 += [nn.GELU()]
      # decod_2 += [nn.Sigmoid()]
      decod_2 += [nn.Conv2d(self.input_dim//4, self.embed_dim, (1,1))]
      decod_2 += [nn.Dropout()]
      self.decod_2 = nn.Sequential(*decod_2)

      # 3nd Block
      decod_3 = []
      decod_3 += [nn.Conv2d(self.embed_dim, self.input_dim//8, (1,1))]
      decod_3 += [nn.LayerNorm((self.input_dim//8, self.aerial_size[0]//4, self.aerial_size[1]//4))]
      # decod_3 += [nn.ReLU()]
      decod_3 += [nn.GELU()]
      # decod_3 += [nn.Sigmoid()]
      decod_3 += [nn.Conv2d(self.input_dim//8, self.embed_dim, (1,1))]
      decod_3 += [nn.Dropout()]
      self.decod_3 = nn.Sequential(*decod_3)

      # 4nd Block
      decod_4 = []
      decod_4 += [nn.Conv2d(self.embed_dim, self.input_dim//8, (1,1))]
      decod_4 += [nn.LayerNorm((self.input_dim//8, self.aerial_size[0]//2, self.aerial_size[1]//2))]
      # decod_4 += [nn.ReLU()]
      decod_4 += [nn.GELU()]
      # decod_4 += [nn.Sigmoid()]
      decod_4 += [nn.Conv2d(self.input_dim//8, self.embed_dim, (1,1))]
      decod_4 += [nn.Dropout()]
      self.decod_4 = nn.Sequential(*decod_4)

      # 5nd Block
      decod_5 = []
      decod_5 += [nn.Conv2d(self.embed_dim, self.input_dim//8, (1,1))]
      decod_5 += [nn.LayerNorm((self.input_dim//8, self.aerial_size[0]//1, self.aerial_size[1]//1))]
      # decod_5 += [nn.ReLU()]
      decod_5 += [nn.GELU()]
      # decod_5 += [nn.Sigmoid()]
      decod_5 += [nn.Conv2d(self.input_dim//8, self.n_classes, (1,1))]
      decod_5 += [nn.Dropout()]
      self.decod_5 = nn.Sequential(*decod_5)

      self.aerial_minmax = nn.Sigmoid()
    
      """Street""" 
      self.aerial_feat0 = nn.Conv2d(8*self.aerial_embed_dim, self.input_dim, (1,1))
      self.aerial_feat1 = nn.Conv2d(4*self.aerial_embed_dim, self.input_dim, (1,1))
      self.aerial_feat2 = nn.Conv2d(2*self.aerial_embed_dim, self.input_dim, (1,1))
      # self.aerial_feat3 = nn.Conv2d(1*self.aerial_embed_dim, self.input_dim, (1,1))

      self.street_feat0 = nn.Conv2d(8*self.street_embed_dim, self.input_dim, (1,1))
      self.street_feat1 = nn.Conv2d(4*self.street_embed_dim, self.input_dim, (1,1))
      self.street_feat2 = nn.Conv2d(2*self.street_embed_dim, self.input_dim, (1,1))
      # self.street_feat3 = nn.Conv2d(1*self.street_embed_dim, self.input_dim, (1,1))

      # self.x_cls0 = nn.AdaptiveAvgPool2d(1)
      self.x_cls1 = nn.AdaptiveAvgPool2d(1)
      self.x_cls2 = nn.AdaptiveAvgPool2d(1)
      # self.x_cls3 = nn.AdaptiveAvgPool2d(1)

      self.cos = nn.CosineSimilarity()

      depth_0 = []
      depth_0 += [nn.Conv2d(1, 10, (1,1))]
      depth_0 += [nn.LayerNorm((10, self.aerial_size[0]//32, self.aerial_size[1]//32))]
      # depth_0 += [nn.ReLU()]
      depth_0 += [nn.GELU()]
      # depth_0 += [nn.Sigmoid()]
      depth_0 += [nn.Conv2d(10, 1, (1,1))]
      depth_0 += [nn.Dropout()]
      self.depth_0 = nn.Sequential(*depth_0)

      depth_1 = []
      depth_1 += [nn.Conv2d(2, 10, (1,1))]
      depth_1 += [nn.LayerNorm((10, self.aerial_size[0]//16, self.aerial_size[1]//16))]
      # depth_1 += [nn.ReLU()]
      depth_1 += [nn.GELU()]
      # depth_1 += [nn.Sigmoid()]
      depth_1 += [nn.Conv2d(10, 1, (1,1))]
      depth_1 += [nn.Dropout()]
      self.depth_1 = nn.Sequential(*depth_1)

      depth_2 = []
      depth_2 += [nn.Conv2d(2, 10, (1,1))]
      depth_2 += [nn.LayerNorm((10, self.aerial_size[0]//8, self.aerial_size[1]//8))]
      # depth_2 += [nn.ReLU()]
      depth_2 += [nn.GELU()]
      # depth_2 += [nn.Sigmoid()]
      depth_2 += [nn.Conv2d(10, 1, (1,1))]
      depth_2 += [nn.Dropout()]
      self.depth_2 = nn.Sequential(*depth_2)

      depth_3 = []
      depth_3 += [nn.Conv2d(1, 10, (1,1))]
      depth_3 += [nn.LayerNorm((10, self.aerial_size[0]//4, self.aerial_size[1]//4))]
      # depth_3 += [nn.ReLU()]
      depth_3 += [nn.GELU()]
      # depth_3 += [nn.Sigmoid()]
      depth_3 += [nn.Conv2d(10, 1, (1,1))]
      depth_3 += [nn.Dropout()]
      self.depth_3 = nn.Sequential(*depth_3)

      depth_4 = []
      depth_4 += [nn.Conv2d(1, 10, (1,1))]
      depth_4 += [nn.LayerNorm((10, self.aerial_size[0]//2, self.aerial_size[1]//2))]
      # depth_4 += [nn.ReLU()]
      depth_4 += [nn.GELU()]
      # depth_4 += [nn.Sigmoid()]
      depth_4 += [nn.Conv2d(10, 1, (1,1))]
      depth_4 += [nn.Dropout()]
      self.depth_4 = nn.Sequential(*depth_4)

      depth_5 = []
      depth_5 += [nn.Conv2d(1, 10, (1,1))]
      depth_5 += [nn.LayerNorm((10, self.aerial_size[0]//1, self.aerial_size[1]//1))]
      # depth_5 += [nn.ReLU()]
      depth_5 += [nn.GELU()]
      # depth_5 += [nn.Sigmoid()]
      depth_5 += [nn.Conv2d(10, 1, (1,1))]
      depth_5 += [nn.Dropout()]
      self.depth_5 = nn.Sequential(*depth_5)

      depth = []
      if "nSeg" in self.base:
        depth += [nn.Conv2d(1, 10, (1,1))]
      else:
        depth += [nn.Conv2d(self.n_classes+1, 10, (1,1))]
      depth += [nn.LayerNorm((10, self.aerial_size[0]//1, self.aerial_size[1]//1))]
      # depth += [nn.ReLU()]
      depth += [nn.GELU()]
      # depth += [nn.Sigmoid()]
      depth += [nn.Conv2d(10, 1, (1,1))]
      depth += [nn.Dropout()]
      self.depth = nn.Sequential(*depth)

      self.street_minmax = nn.Sigmoid()

  def forward(self, id, y, y1=None, seg1=None, y_cls=None):
    if(id=="aerial"):
      if "featup" in self.base:
        map = self.labeling(y)
        map = self.aerial_minmax(map)

      elif "mst" in self.base:
        # UNet
        x = self.decod_0(y[2].permute(0,2,1).reshape(y[2].size(0),
                                                     y[2].size(2),
                                                     int(y[2].size(1)**.5),
                                                     int(y[2].size(1)**.5)))
        x = self.up(x)
        x = torch.cat([x, y[1].permute(0,2,1).reshape(y[1].size(0),
                                                      y[1].size(2),
                                                      int(y[1].size(1)**.5),
                                                      int(y[1].size(1)**.5))], dim=1)
        x = self.decod_1(x)
        x = self.up(x)
        x = torch.cat([x, y[0].permute(0,2,1).reshape(y[0].size(0),
                                                         y[0].size(2),
                                                         int(y[0].size(1)**.5),
                                                         int(y[0].size(1)**.5))], dim=1)
        x = self.decod_2(x)
        x = self.up(x)

        x = self.decod_3(x)
        x = self.up(x)

        x = self.decod_4(x)
        x = self.up(x)

        x = self.decod_5(x)

        map = self.aerial_minmax(x)
        
    elif(id=="street"):
      if "featup" in self.base:
        y1 = self.aerial_feat(y1)

        y = self.street_feat(y)
        y = self.x_cls(y)

        cosmap = self.cos(y, y1)
        map = cosmap.unsqueeze(1)

        if not("nSeg" in self.base):
          map = torch.cat([map, seg1], dim=1)
        map = self.depth(map)
        map = self.street_minmax(map)
        
      elif "mst" in self.base:
        aerial_set = self.aerial_feat0(y1[2].permute(0,2,1).reshape(y1[2].size(0),
                                                                    y1[2].size(2),
                                                                    int(y1[2].size(1)**.5),
                                                                    int(y1[2].size(1)**.5)))
        street_set = self.street_feat0(y_cls.unsqueeze(-1).unsqueeze(-1))
        cos_sim = self.cos(aerial_set, street_set)
        map = self.depth_0(cos_sim.unsqueeze(1))
        map = self.up(map)

        aerial_set = self.aerial_feat1(y1[1].permute(0,2,1).reshape(y1[1].size(0),
                                                                    y1[1].size(2),
                                                                    int(y1[1].size(1)**.5),
                                                                    int(y1[1].size(1)**.5)))
        street_set = self.x_cls1(self.street_feat1(y[1].permute(0,2,1).reshape(y[1].size(0),
                                                                               y[1].size(2),
                                                                               self.street_height,
                                                                               self.street_width)))
        cos_sim = self.cos(aerial_set, street_set)
        map = torch.cat([map, cos_sim.unsqueeze(1)], dim=1)
        map = self.depth_1(map)
        map = self.up(map)

        aerial_set = self.aerial_feat2(y1[0].permute(0,2,1).reshape(y1[0].size(0),
                                                                    y1[0].size(2),
                                                                    int(y1[0].size(1)**.5),
                                                                    int(y1[0].size(1)**.5)))
        street_set = self.x_cls2(self.street_feat2(y[0].permute(0,2,1).reshape(y[0].size(0),
                                                                               y[0].size(2),
                                                                               2*self.street_height,
                                                                               2*self.street_width)))
        cos_sim = self.cos(aerial_set, street_set)
        map = torch.cat([map, cos_sim.unsqueeze(1)], dim=1)
        map = self.depth_2(map)
        map = self.up(map)

        map = self.depth_3(map)
        map = self.up(map)

        map = self.depth_4(map)
        map = self.up(map)

        if not("nSeg" in self.base):
          map = torch.cat([map, seg1], dim=1)
        map = self.depth(map)

        map = self.street_minmax(map)

    return map

@register_model
def featup(**kwargs):
    # model = torch.hub.load("mhamilton723/FeatUp", 'vit')
    model = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=True)
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
