import torch
import numpy as np
import pickle as pkl


# A simple demo for NormRescale.
# Use model A to rescale the norm of model B.

# for pytorch checkpoints
a = torch.load('path/to/modelA.pth', map_location='cpu')['model']
a = torch.load('path/to/modelB.pth', map_location='cpu')['model']

d = {}
for i in b.keys():
    if 'online' in i:
        try:
            d[i] = b[i] / b[i].norm() * a[i].norm()
        except:
            pass

print(len(d))

torch.save({'model': d}, 'path/to/save.pth')

exit()


# for pickle checkpoints
a = pkl.load(open('path/to/modelA.pkl', 'rb'))['model']
b = pkl.load(open('path/to/modelB.pkl', 'rb'))['model']

d = {}
for i in a.keys():
    try:
        d[i] = b[i] / torch.tensor(b[i]).norm().numpy() * torch.tensor(a[i]).norm().numpy()
    except:
        print(i)

print(len(d))


res = {"model": d, "__author__": "torchvision", "matching_heuristics": True}
pkl.dump(res, open('path/to/save.pkl', 'wb'))
