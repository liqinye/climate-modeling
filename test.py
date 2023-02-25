import torch
import matplotlib.pyplot as plt
from datapipe import TrainDataset
from unet import *
from diffusion import *
from elucidated_diffusion import *
from stats import *

def plot(sample, num_sample, mode, milestone=None):
    grid = int(math.sqrt(num_sample))
    if mode == "GT":
        for index in range(10):
            fig, axes = plt.subplots(figsize=(18, 10),nrows=grid, ncols=grid)
            for i in range(index*num_sample, (index+1)*num_sample):
                row = (i-index*num_sample) // grid
                col = (i-index*num_sample)  % grid
                im = axes[row][col].imshow(sample[i][0].cpu().detach().numpy(),interpolation='bilinear',cmap=plt.cm.jet,origin='lower', vmin=0, vmax=0.0001)
                fig.colorbar(im, ax=axes[row][col])
            fig.savefig(f"/home/liqiny/climate/GT-CRM_QPI-{index}.png")
            plt.close("all")
    else:
        fig, axes = plt.subplots(figsize=(18, 10),nrows=grid, ncols=grid)
        for i in range(num_sample):
            row = i // grid
            col = i % grid
            im = axes[row][col].imshow(sample[i].cpu().detach().numpy(),interpolation='bilinear',cmap=plt.cm.jet,origin='lower', vmin=0, vmax=0.001)
            fig.colorbar(im, ax=axes[row][col])
        fig.savefig(f"/home/liqiny/climate/ED_sample-1000-{milestone}-0.001.png")
    plt.close("all")

def modelSample():
    unet = Unet(
        dim = 64,
        dim_mults = (4, 8, 16, 16),
        random_fourier_features = True,
        channels = 5,
        learned_sinusoidal_dim = 64
    ).cuda()

    '''
    Elucidated Diffusion
    '''
    elucidated_diffusion = ElucidatedDiffusion(
        unet,
        image_size = (24, 32),
        num_sample_steps = 1000,
        channels = 5
    ).cuda()

    device = torch.device("cuda:0")

    train_dataset = TrainDataset("/extra/ucibdl0/shared/data/climate/h3/*.nc")
    ds_maxmin = train_dataset.getMaxMin()
    # milestones = [58, 73, 75, 81, 83, 89, 90, 91, 92, 98, 100]
    # for milestone in milestones:
    milestone = 200
    data = torch.load(str(f"/home/liqiny/climate/results/model-{milestone}.pt"), map_location=device)
    elucidated_diffusion.load_state_dict(data['model'])

    # for i in range(2):
    #     print(f"loop {i}\n")
        # sample = elucidated_diffusion.sample(batch_size=2000, ds_maxmin=ds_maxmin)
        # torch.save(sample, f"/home/liqiny/climate/sample_200_4_2.pt")
        # sample = torch.squeeze(sample)
        # plot(sample, 25, "model", milestone=milestone)
    # sample = torch.load("/home/liqiny/climate/sample.pt")

    # sample1 = torch.load(f"/home/liqiny/climate/sample_100_10k.pt")
    sample = torch.load(f"/home/liqiny/climate/sample_200_10k.pt")

    for i in range(0,5):
        s = torch.load(f"/home/liqiny/climate/sample_200_{i}_2.pt")
        sample = torch.cat((sample, s), 0)
    torch.save(sample, f"/home/liqiny/climate/sample_100_20k.pt")

    sampleStats1(sample, milestone)
    sampleStats2(sample, milestone)


def groundTruth():
    train_dataset = TrainDataset("/extra/ucibdl0/shared/data/climate/h3/*.nc")
    sample = train_dataset.CRM_QPI
    plot(sample, 25, "GT")


if __name__ == "__main__":
    # modelSample()

    unet = Unet(
        dim = 64,
        dim_mults = (4, 8, 16, 16),
        random_fourier_features = True,
        channels = 5,
        learned_sinusoidal_dim = 64,
        condition = True
    ).cuda()

    x = torch.randn([100, 5, 24, 32]).cuda()
    y = torch.randn([100, 18, 24, 32]).cuda()
    t = torch.randint(0, 1000, (100,)).cuda()
    z = unet(x, t, cond=y)
    print(z.size())
    
    # groundTruth()

    # loss = torch.load('/home/liqiny/climate/results/CRM_QC_ED_100k/model/loss.pt')
    # plt.plot(loss.cpu().detach().numpy())
    # plt.savefig('123.png')
    # plt.close('all')