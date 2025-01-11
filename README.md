# **CVSegGuide**
[![Open CVSegGuide in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yEqk_eFxmwLafMOeTbtH2t8o2Evbziae?usp=sharing)

[Nathan A. Z. Xavier](http://lattes.cnpq.br/2088578568009855),
[Elcio H. Shiguemori](http://lattes.cnpq.br/7243145638158319),
[Marcos R. O. A. Maximo](http://lattes.cnpq.br/1610878342077626),
[Mubarak Shah](https://www.crcv.ucf.edu/person/mubarak-shah/)

<p align="center">
<img src="https://github.com/nathanxavier/CVSegGuide/blob/e0a8e126908861d44dccdcadd3218cf56547b377/Figures/Graphical%20Abstract.png">
</p>

Cross-view geo-localization guided by the land cover semantic segmentation map.

# **Dataset**
We trained the solution using the [Brooklyn and Queens Dataset](https://arxiv.org/pdf/2204.01807) and tested also using the cross test set from [VIGOR](https://github.com/Jeff-Zilence/VIGOR) dataset.


# **Requirement**
```
- Python >=3.8, numpy, pandas, matplotlib, timm, pytorch-msssim, tqdm, sparse
- PyTorch >= 12.1, torchvision torchaudio pytorch-cuda
```

# **Training**
We elaborated two different models using:
 1. A partially trained model based on the [FeatUp Backbone](https://github.com/mhamilton723/FeatUp)
 2. A full-trained model using a Multi-Scale Transformer (MST).

# **Evaluation**
We compute the performances verifying the semantic segmentation map estimated by the proposed method:
<p align="center">
<img src="https://github.com/nathanxavier/CVSegGuide/blob/main/Figures/B%26Q%20Segments.png">
</p>

and Discrete Probability Distribution (DPD) of the top-ranked regions in the image:
<p align="center">
<img src="https://github.com/nathanxavier/CVSegGuide/blob/main/Figures/B%26Q%20Regions.png">
</p>

The methodology proposed can be applied to any cross-view geo-localization dataset that uses overview and street-view images.

# **Citation**
If you find this work useful in your research, please consider citing:

[Nathan A. Z. Xavier, Elcio H. Shiguemori, Marcos R. O. A. Maximo, Mubarak Shah, *A guided approach for cross-view geolocalization estimation with land cover semantic segmentation*, Biomimetic Intelligence and Robotics, 2025](https://doi.org/10.1016/j.birob.2024.100208).

```bibtex
@article{Xavier2025,
  title = {A guided approach for cross-view geolocalization estimation with land cover semantic segmentation},
  journal = {Biomimetic Intelligence and Robotics},
  pages = {100208},
  year = {2025},
  issn = {2667-3797},
  doi = {https://doi.org/10.1016/j.birob.2024.100208},
  url = {https://www.sciencedirect.com/science/article/pii/S2667379724000664},
  author = {Nathan A.Z. Xavier and Elcio H. Shiguemori and Marcos R.O.A. Maximo and Mubarak Shah},
  keywords = {Cross-view geolocalization, Semantic segmentation, Satellite and ground image fusion, Simultaneous localization and mapping (SLAM)},
}
