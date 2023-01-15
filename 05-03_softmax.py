# https://wikidocs.net/60572

import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.rand(3, 5, requires_grad=True)


print((y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean())







