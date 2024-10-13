import torch.optim as optim
from torch.optim import lr_scheduler


def make_optimizer(model):
  ignored_params = []
  
  for i in [model.aerial, model.street]:
    ignored_params += list(map(id, i.model.parameters()))

  extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
  base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
  optimizer_ft = optim.SGD([
      {'params': base_params, 'lr': 0.3 * .01},
      {'params': extra_params, 'lr': .01}
  ], weight_decay=5e-4, momentum=0.9, nesterov=True)


  exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[70,100], gamma=0.1)
  # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)
  # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
  # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=4, verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

  return optimizer_ft,exp_lr_scheduler