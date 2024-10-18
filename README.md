# **CVSegGuide**
[![Open CVSegGuide in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yEqk_eFxmwLafMOeTbtH2t8o2Evbziae?usp=sharing)


    \author[label1,label2,label5]{Nathan A. Z. Xavier}
    \ead{nathanxavier@ufmg.br}
    \author[label2,label3,label4]{Elcio H. Shiguemori}
    \ead{elcio@ieav.cta.br}
    \author[label2]{Marcos R. O. A. Maximo}
    \ead{mmaximo@ita.br}
    \author[label5]{Mubarak Shah}
    \ead{shah@crcv.ucf.edu}

    
[Nathan A. Z. Xavier](http://lattes.cnpq.br/2088578568009855),

[Elcio H. Shiguemori](http://lattes.cnpq.br/7243145638158319),
[Marcos R. O. A. Maximo](http://lattes.cnpq.br/1610878342077626),
[Mubarak Shah]()

Cross-view geo-localization guided by the land cover semantic segmentation map.

<p align="center">
<img src="https://github.com/nathanxavier/CVSegGuide/blob/e0a8e126908861d44dccdcadd3218cf56547b377/Figures/Graphical%20Abstract.png">
</p>

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
