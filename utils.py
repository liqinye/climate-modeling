import math, torch



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def MinMaxNorm(img, img_maxmin):
    for c in range(img.size()[1]):
        img[:, c, :, :] = (img[:, c, :, :] - img_maxmin[c][1]) / (img_maxmin[c][0] - img_maxmin[c][1])
    return img

def MinMaxNormRevert(img, img_maxmin):
    for c in range(img.size()[1]):
        img[:, c, :, :] = img[:, c, :, :] * (img_maxmin[c][0] - img_maxmin[c][1]) + img_maxmin[c][1]
    return img

def Norm(img, img_meanstd):
    for c in range(img.size()[1]):
        img[:, c, :, :] = (img[:, c, :, :] - img_meanstd[c][0]) / img_meanstd[c][1]
    return img

def NormRevert(img, img_meanstd):
    for c in range(img.size()[1]):
        img[:, c, :, :] = img[:, c, :, :] * img_meanstd[c][1] + img_meanstd[c][0]
    return img

