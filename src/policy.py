import numpy as np
import torch


def clipped_loss(new_log,old_log,advantage,clip):
    ratio=(new_log-old_log).exp()
    advantage=torch.as_tensor(advantage,device=ratio.device,dtype=ratio.dtype)
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()


def standardized(rewards):
    values=np.asarray(rewards,dtype=np.float64)
    return (values-values.mean())/max(values.std(),1e-6)


def color_fraction(image,channel):
    rgb=np.asarray(image.convert("RGB"),dtype=float)/255.
    target=rgb[:,:,channel]
    others=np.delete(rgb,channel,axis=2).max(2)
    return float(((target>others+0.12)&(target>0.24)).mean())


def roughness(image):
    a=np.asarray(image.convert("RGB"),dtype=float)/255.
    return float((np.abs(np.diff(a,axis=0)).mean()+np.abs(np.diff(a,axis=1)).mean())/2)
