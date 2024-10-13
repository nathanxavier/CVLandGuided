# **CVSegGuide**
Cross-view geo-localization guided by the land cover semantic segmentation map.

![Graphical_Abstract](https://github.com/nathanxavier/CVSegGuide/blob/e0a8e126908861d44dccdcadd3218cf56547b377/Figures/Graphical%20Abstract.png)


# **Dataset**
We trained the solution using the [Brooklyn and Queens Dataset](https://arxiv.org/pdf/2204.01807) and tested also using the cross test set from [VIGOR](https://github.com/Jeff-Zilence/VIGOR) dataset.


# **Requirement**
```
- Python >=, numpy, matplotlib, pillow, ptflops, timm
- PyTorch >= , torchvision >= 
```

# **Training and Evaluation**
We elaborated two different models using:
 1. A partial trained model based on the [FeatUp Backbone](https://github.com/mhamilton723/FeatUp)
 2. A full-trained model using a Multi-Scale Transformer.
