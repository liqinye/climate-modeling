import matplotlib.pyplot as plt
import torch


def sampleStats1(sample, milestone):
    x = torch.range(0, sample.size()[-2] * sample.size()[-1] - 1)
    title = ['CRM_QV', 'CRM_QC', 'CRM_QI', 'CRM_QPC', 'CRM_QPI']
    for c in range(sample.size()[1]):
        var = sample[:, c, :, :]
        var = var.view(([var.size()[0], -1]))
        mean = torch.mean(var, dim=0)
        fig = plt.figure(figsize=(18,14))
        plt.plot(x.numpy(), mean.detach().cpu().numpy())
        fig.suptitle(f"{title[c]} Stats1")
        fig.savefig(f"/home/liqiny/climate/{title[c]}_{milestone}_stats1.png")
        plt.close()        





def sampleStats2(sample, milestone):
    x = torch.range(0, sample.size()[-2]-1)
    title = ['CRM_QV', 'CRM_QC', 'CRM_QI', 'CRM_QPC', 'CRM_QPI']
    for c in range(sample.size()[1]):
        var = sample[:, c, :, :]
        # var = var.view(([var.size()[0], -1]))
        var = torch.sum(var, dim=-1)
        mean = torch.mean(var, dim=0)
        variance = torch.std(var, dim=0)
        fig = plt.figure(figsize=(18,14))
        plt.plot(x.numpy(), mean.detach().cpu().numpy(), label="mean")
        plt.plot(x.numpy(), variance.detach().cpu().numpy(), label="std")
        fig.suptitle(f"{title[c]} Stats2")
        fig.legend()
        fig.savefig(f"/home/liqiny/climate/{title[c]}_{milestone}_stats2.png")
        plt.close()        

def GT(sample):
    x = torch.range(0, sample.size()[-2]-1)
    title = ['CRM_QV', 'CRM_QC', 'CRM_QI', 'CRM_QPC', 'CRM_QPI']
    for c in range(sample.size()[1]):
        var = sample[:, c, :, :]
        # var = var.view(([var.size()[0], -1]))
        var = torch.sum(var, dim=-1)
        mean = torch.mean(var, dim=0)
        variance = torch.var(var, dim=0)
        fig = plt.figure(figsize=(18,14))
        plt.plot(x.numpy(), mean.detach().cpu().numpy(), label="mean")
        plt.plot(x.numpy(), variance.detach().cpu().numpy, label="variance")
        fig.suptitle(f"{title[c]} Stats2")
        fig.savefig(f"/home/liqiny/climate/GT/{title[c]}_stats2.png")
        plt.close()       


if __name__ == "__main__":
    x = torch.randn([5, 10, 24, 32])
    # y = torch.sum(x, dim=-1)
    # z = torch.mean(y, dim=1)
    # print(z.size())
    # print(x.size())
    sampleStats2(x, 10)
    # x = x.view([2, -1])
    # print(x.size())
    # y = torch.mean(x, dim=0)
    # print(y.size())
    