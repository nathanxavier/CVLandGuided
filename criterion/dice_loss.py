import torch
from torch import Tensor
from torch import nn

class Dice_Loss(nn.Module):
  def __init__(self, multiclass=True):
    super().__init__()
    self.multiclass = multiclass

  def forward(self, input, target):
    output = self.dice_loss(input, target, self.multiclass)

    return output

  def dice_coeff(self, input, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


  def multiclass_dice_coeff(self, input, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    return self.dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


  def dice_loss(self, input, target, multiclass=False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


  def clip(self, images, n_class):
    n = images.size(0)
    clip = torch.zeros(n, n_class, 256,256).cuda()
    for n_img in range(n):
      for value in range(n_class):
        pos_i, pos_j = torch.where(images[n_img] == value+1)
        clip[n_img, value, pos_i, pos_j] = 1
    return clip