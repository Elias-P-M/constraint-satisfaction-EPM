### Commonly used functions

import torch
from torch import Tensor


# =================== Hypervolume (EXACT; NOT EHVI) ===================
#region

#endregion

# =================== Statistical Functions ===================
#region

def _mean_signorm(mu:Tensor,sig:Tensor,steps=int(1e4),eps = 1e-5,x = None):
  '''
  Performs trapezoidal integration over the analytic distribution of the sigmoid of a normal variable.
  Exposed version is run through vmap. mu and sigma require the same shape. 
  Extensive testing for edge case shapes has not been preformed and there is currently minimal checking
    :param mu: mean of normal distribution
    :param sig: mean of normal distribution
    :param steps: number of steps for numerical integration
    :param eps: offset from 0 and 1 for numerical inegration to avoid infinities
    :param x: Optional parameter allowing precomputation of integration tensor
    :return: 
  '''
    if x is None:
        device = mu.device
        x = torch.linspace(eps,1-eps,steps).to(device)

    y = x * torch.exp( -((torch.logit(x)-mu)/sig)**2 /2 ) / ( (2*torch.pi)**(.5) * sig * x * (1-x) )
    print(y.shape)
    return torch.trapz(y,dx = (1 - 2 * eps)/steps,dim=-1)

mean_signorm = vmap(mu_sig_norm_trapy_pre)

#endregion
