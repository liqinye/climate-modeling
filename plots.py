import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
plt.rcParams['svg.fonttype'] = 'none'


def noise_plot():
    sample = np.load("data/GT.npy")
    stds = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1]

    fig, ax = plt.subplots(1, 8, figsize=(9.5, 1))
    x = sample[0]
    for i in range(8):
        x += np.random.normal(loc=0, scale=stds[i], size=x.shape)
        ax[i].imshow(x, interpolation='bicubic', cmap=plt.cm.jet, origin='lower',
                     vmin=np.percentile(x, 1), vmax=np.percentile(x, 99))
        ax[i].axis('off')
    plt.tight_layout(pad=0, w_pad=0.2)
    plt.savefig('noisy.png', dpi=150)
    plt.show()


def mean_std(alpha=0.05):
    # 'CRM_QV', 'CRM_QC', 'CRM_QI', 'CRM_QPC', 'CRM_QPI'
    sample = np.maximum(np.load('data/uncond_sample_20k.npy'), 0)
    gt = np.load('data/GT_200k.npy')

    # gt_metrics = np.load('data/gt_stats.npy')
    # n2 = 2_967_120
    # gt_mean, gt_std = gt[0], gt[1]
    # # Assume Gamma(a, theta) distribution
    # gt_theta = gt_std ** 2 / gt_mean
    # gt_a = (gt_mean / gt_std) ** 2

    n, _, h, _ = sample.shape
    n2, _, h2, _ = gt.shape
    assert h == h2

    x = np.arange(h)
    agg = sample.sum(axis=-1)
    gt_agg = gt.sum(axis=-1)
    sample_mean, sample_std = agg.mean(axis=0), agg.std(axis=0)
    gt_mean, gt_std = gt_agg.mean(axis=0), gt_agg.std(axis=0)
    As = np.linspace(0.5, alpha/2, 2)[0:]
    sample_l = [np.percentile(agg, 100 * a, axis=0) for a in As]
    sample_u = [np.percentile(agg, 100 * (1 - a), axis=0) for a in As]
    gt_l = [np.percentile(gt_agg, 100 * a, axis=0) for a in As]
    gt_u = [np.percentile(gt_agg, 100 * (1 - a), axis=0) for a in As]

    # q_n = norm.ppf(1 - alpha/2)
    # sample_ci = q_n / np.sqrt(n) * sample_std
    # gt_ci = q_n / np.sqrt(n2) * gt_std
    # gt_d = q_n * gt_std

    for c in [0, 4]:
        # gt_5, gt_95 = gamma.ppf(alpha/2, a=gt_a[c], scale=gt_theta[c]), gamma.ppf(1-alpha/2, a=gt_a[c], scale=gt_theta[c])

        fig, ax = plt.subplots(figsize=(5, 2.7))
        ax.plot(x, sample_mean[c], label="EDM", color="blue")
        ax.plot(x, gt_mean[c], label="CRM", color="red")
        # ax.fill_between(x, (sample_mean[c] - sample_ci[c]), (sample_mean[c] + sample_ci[c]), alpha=0.1)
        # ax.fill_between(x, (gt_mean[c] - gt_ci[c]), (gt_mean[c] + gt_ci[c]), alpha=0.1)
        ax.fill_between(x, sample_l[-1][c], sample_u[-1][c], alpha=0.2, color='blue')
        # ax.fill_between(x, (gt_mean[c] - gt_d[c]), (gt_mean[c] + gt_d[c]), alpha=0.2, color='red')
        ax.fill_between(x, gt_l[-1][c], gt_u[-1][c], alpha=0.2, color='red')
        # ax.plot(x, gt_l[-2][c], color="blue", lw=1, ls="--")
        # ax.plot(x, gt_l[1][c], color="blue", lw=1, ls="--")
        # ax.plot(x, gt_u[-2][c], color="red", lw=1, ls="--")
        # ax.plot(x, gt_u[1][c], color="blue", lw=1, ls="--")
        # ax.plot(x, sample_l[-2][c], color="red", lw=1, ls="--")
        # ax.plot(x, sample_l[1][c], color="red", lw=1, ls="--")
        # ax.plot(x, sample_u[-2][c], color="blue", lw=1, ls="--")
        # ax.plot(x, sample_u[1][c], color="red", lw=1, ls="--")
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel('Mixing ratio [kg/kg]')
        ax.set_xlabel('Altitude')
        plt.grid(alpha=0.4)
        plt.tight_layout(pad=0)
        plt.savefig('2_%d.svg' % c, dpi=150)
        plt.savefig('2_%d.png' % c, dpi=150)
    plt.show()


if __name__ == '__main__':
    mean_std(0.05)
