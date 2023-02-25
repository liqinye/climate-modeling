import xarray as xr
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *
from stats import *

'''
filter data by a lattitue range
'''
def _preprocess(x, latt_bound=(-30.00, -5.00)):
    return x.sel(lat=slice(*latt_bound))


class TrainDataset(Dataset):
    def __init__(self, dataset_dir: str, cond_dataset_dir: str):
        super().__init__()
        self.dataset = xr.open_mfdataset(dataset_dir, preprocess=_preprocess)
        self.CRM_QV = self.reshape(torch.squeeze(torch.tensor(self.dataset.CRM_QV.values)))
        self.CRM_QC = self.reshape(torch.squeeze(torch.tensor(self.dataset.CRM_QC.values)))
        self.CRM_QI = self.reshape(torch.squeeze(torch.tensor(self.dataset.CRM_QI.values)))
        self.CRM_QPC = self.reshape(torch.squeeze(torch.tensor(self.dataset.CRM_QPC.values)))
        self.CRM_QPI = self.reshape(torch.squeeze(torch.tensor(self.dataset.CRM_QPI.values)))
        assert self.CRM_QV.size()[0] == self.CRM_QC.size()[0] == self.CRM_QI.size()[0] == \
                self.CRM_QPC.size()[0] == self.CRM_QPI.size()[0], "Total data points number should be same"
        assert (self.CRM_QV.size()[2] == self.CRM_QC.size()[2] == self.CRM_QI.size()[2] == \
                self.CRM_QPC.size()[2] == self.CRM_QPI.size()[2]) and \
                (self.CRM_QV.size()[3] == self.CRM_QC.size()[3] == self.CRM_QI.size()[3] == \
                self.CRM_QPC.size()[3] == self.CRM_QPI.size()[3]), "Data Image shape should be same"
        self.data = torch.cat((self.CRM_QV, self.CRM_QC, self.CRM_QI, self.CRM_QPC, self.CRM_QPI), dim=1)

        self.cond_dataset = xr.open_mfdataset(cond_dataset_dir, preprocess=_preprocess)
        self.condition = [
            self.cond_dataset.TBP.values, self.cond_dataset.QBP.values, self.cond_dataset.UBP.values, self.cond_dataset.VBP.values,
            self.cond_dataset.SBP.values, self.cond_dataset.OMEGABP.values, self.cond_dataset.PMIDBP.values,
            self.cond_dataset.PMIDDRYBP.values, self.cond_dataset.PDELBP.values, self.cond_dataset.PDELDRYBP.values,
            self.cond_dataset.RPDELBP.values, self.cond_dataset.RPDELDRYBP.values, self.cond_dataset.LNPMIDBP.values,
            self.cond_dataset.LNPMIDDRYBP.values, self.cond_dataset.EXNERBP.values, self.cond_dataset.ZMBP.values,
            self.cond_dataset.CLDLIQBP.values, self.cond_dataset.CLDICEBP.values
        ]
        self.condition_data = self._process_condition(self.condition)

        # data shape: [23, 24, 32], first 18 is condition, last 5 is data
        # self.data = torch.cat((self.condition_data, self.data), dim=1)


    def __getitem__(self, index):
        return self.data[index], self.condition_data[index]

    def __len__(self):
        return self.data.size()[0]

    '''
    reshape data from [time, crm_nx, crm_nz, lat, lon]
    -> [time x lat x lon, crm_nx, crm_nz]
    '''
    def reshape(self, data):
        # [time, crm_nx, crm_nz, lat, lon] -> [time, lat, crm_nz, crm_nx, lon]
        data = torch.transpose(data, 1, 3)
        # [time, lat, crm_nz, crm_nx, lon] -> [time, lat, lon, crm_nx, crm_nz]
        data = torch.transpose(data, 2, 4)
        # [time, lat, lon, crm_nx, crm_nz]
        data_shape = data.size()
        # [time, lat, lon, crm_nx, crm_nz] -> [timexlat, lon, crm_nx, crm_nz]
        data = data.reshape(-1, data_shape[2], data_shape[3], data_shape[4])
        # [timexlat, lon, crm_nx, crm_nz] -> [timexlatxlon, crm_nx, crm_nz]
        data = data.reshape(-1, data_shape[3], data_shape[4])
        data = data[:, None, :, :]
        return data

    def _process_condition(self, condition):
        data = torch.tensor([])
        for c in condition:
            # (26,) -> (24,) & flip GCM array to match CRM order
            c = torch.flip(torch.tensor(c)[:, 2:26, :, :], dims=(1,))
            # expand (24,) -> (24, 32) to match CRM variables
            c = torch.reshape(c, [c.size()[0], c.size()[1], 1, c.size()[2], c.size()[3]])
            c = c.expand(c.size()[0], c.size()[1], 32, c.size()[3], c.size()[4])
            c = self.reshape(c)
            data = torch.cat((data, c), dim=1)
        return data

    def getDataset(self, variable = None):
        if variable == "CRM_QV":
            return self.CRM_QV
        elif variable == "CRM_QC":
            return self.CRM_QC
        elif variable == "CRM_QI":
            return self.CRM_QI
        elif variable == "CRM_QPI":
            return self.CRM_QPI
        elif variable == "CRM_QPC":
            return self.CRM_QPC
        elif variable == "condition":
            return self.condition_data
        else:
            return self.data

    def getMaxMin(self):
        channels = self.data.size()[1]
        cond_channels = self.condition_data.size()[1]
        return [(torch.max(self.data[:, c, :, :]), torch.min(self.data[:, c, :, :])) for c in range(channels)], \
                [(torch.max(self.condition_data[:, c, :, :]), torch.min(self.condition_data[:, c, :, :])) for c in range(cond_channels)]
    
    def getMeanStd(self):
        channels = self.data.size()[1]
        cond_channels = self.condition_data.size()[1]
        return [(torch.mean(self.condition_data[:, c, :, :]), torch.std(self.condition_data[:, c, :, :])) for c in range(cond_channels)], \
                [(torch.mean(self.data[:, c, :, :]), torch.std(self.data[:, c, :, :])) for c in range(channels)]

    def normData(self, normdata, cond_normdata):
        assert normdata.size() == self.data.size(), "data must be in the same shape"
        assert cond_normdata.size() == self.condition_data.size(), "data must be in the same shape"
        self.data = normdata
        self.condition_data = cond_normdata

def plot(image, num_samples):
    channels = image.size()[1]
    fig = plt.figure(constrained_layout=True, figsize=(18, 14))
    fig.suptitle(f"sample")
    subfigs = fig.subfigures(nrows=channels, ncols=1)
    title = ['CRM_QV', 'CRM_QC', 'CRM_QI', 'CRM_QPC', 'CRM_QPI']
    vmax = {'CRM_QV': 0.015, 'CRM_QC': 0.001, 'CRM_QI': 4e-6, 'CRM_QPC': 5.2e-6, 'CRM_QPI': 2e-6}
    for channel, subfig in enumerate(subfigs):
        subfig.suptitle(f'{title[channel]}')
        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=num_samples)
        for col, ax in enumerate(axs):
            im = ax.imshow(image[col][channel], interpolation='bilinear',cmap=plt.cm.jet,origin='lower') #,vmax=vmax[title[channel]])
            subfig.colorbar(im, ax=ax)
    fig.savefig('1.png')
    plt.close("all")

class ConditionDataset(Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset = xr.open_mfdataset(dataset_dir, preprocess=_preprocess)
        self.condition = [
            self.dataset.TBP.values, self.dataset.QBP.values, self.dataset.UBP.values, self.dataset.VBP.values,
            self.dataset.SBP.values, self.dataset.OMEGABP.values, self.dataset.PMIDBP.values,
            self.dataset.PMIDDRYBP.values, self.dataset.PDELBP.values, self.dataset.PDELDRYBP.values,
            self.dataset.RPDELBP.values, self.dataset.RPDELDRYBP.values, self.dataset.LNPMIDBP.values,
            self.dataset.LNPMIDDRYBP.values, self.dataset.EXNERBP.values, self.dataset.ZMBP.values,
            self.dataset.CLDLIQBP.values, self.dataset.CLDICEBP.values
            ]
        self.data = self._process_condition(self.condition)
        print(self.data.size())
        
    def _process_condition(self, condition):
        data = torch.tensor([])
        for c in condition:
            # (26,) -> (24,) & flip GCM array to match CRM order
            c = torch.flip(torch.tensor(c)[:, 2:26, :, :], dims=(1,))
            # expand (24,) -> (24, 32) to match CRM variables
            c = torch.reshape(c, [c.size()[0], c.size()[1], 1, c.size()[2], c.size()[3]])
            c = c.expand(c.size()[0], c.size()[1], 32, c.size()[3], c.size()[4])
            c = self.reshape(c)
            data = torch.cat((data, c), dim=1)
        return data

    def reshape(self, data):
        # [time, crm_nx, crm_nz, lat, lon] -> [time, lat, crm_nz, crm_nx, lon]
        data = torch.transpose(data, 1, 3)
        # [time, lat, crm_nz, crm_nx, lon] -> [time, lat, lon, crm_nx, crm_nz]
        data = torch.transpose(data, 2, 4)
        # [time, lat, lon, crm_nx, crm_nz]
        data_shape = data.size()
        # [time, lat, lon, crm_nx, crm_nz] -> [timexlat, lon, crm_nx, crm_nz]
        data = data.reshape(-1, data_shape[2], data_shape[3], data_shape[4])
        # [timexlat, lon, crm_nx, crm_nz] -> [timexlatxlon, crm_nx, crm_nz]
        data = data.reshape(-1, data_shape[3], data_shape[4])
        data = data[:, None, :, :]
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.size()[0]

if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    # # plot([], 5)
    # latt_bound = (-30.00, -5.00)
    # partial_func = partial(_preprocess, latt_bound=latt_bound)

    # train_dataset = TrainDataset("/extra/ucibdl0/shared/data/climate/h3/*.nc")
    
    # ds_meanstd = train_dataset.getMeanStd()
    # print(train_dataset.CRM_QPI[0, 0, :, :])
    # train_dataset.normData(Norm(train_dataset.CRM_QPI, ds_meanstd))
    # print(train_dataset.CRM_QPI[0, 0, :, :])

    # data = NormRevert(train_dataset.CRM_QPI, ds_meanstd)
    
    # ds_maxmin = train_dataset.getMaxMin()
    # train_dataset.normData(MinMaxNorm(train_dataset.getDataset(), ds_maxmin))
    # print(train_dataset.getDataset().size())

    # GT(train_dataset.getDataset())

    train_dataset = TrainDataset("/extra/ucibdl0/shared/data/climate/h3/*.nc", "/extra/ucibdl0/shared/data/climate/h0/*.nc")
    train_size = len(train_dataset) - 500
    valid_size = 500
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    # valid_ds = valid_dataset.getDataset('condition')
    data, cond = valid_dataset[:]


    # condition_dataset = ConditionDataset("/extra/ucibdl0/shared/data/climate/h0/*.nc")
    # condition_dataset = xr.open_mfdataset("/extra/ucibdl0/shared/data/climate/h0/*.nc", preprocess=_preprocess)
    # TBP = torch.flip(torch.tensor(condition_dataset.TBP.values)[:,2:26,:,:], dims=(1,))
    # TBP = torch.reshape(TBP, [TBP.size()[0], TBP.size()[1], 1, TBP.size()[2], TBP.size()[3]])
    # TBP = TBP.expand(TBP.size()[0], TBP.size()[1], 32, TBP.size()[3], TBP.size()[4])
    # print(TBP.size())
    # # print(TBP.size())
    # print(TBP[0,:,:,0,0])
    

    # train_dataset = xr.open_mfdataset("/extra/ucibdl0/shared/data/climate/h3/*.nc", preprocess=_preprocess)
    # CRM_QC = torch.tensor(train_dataset.CRM_QV.values)
    # print(CRM_QC.size())
    # print(CRM_QC[0,:,0,:,0,0])


    
    # train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    # train_bar = tqdm(train_loader)
    # for data in train_bar:
    #     plot(data, 5)
        # fig, axes = plt.subplots(figsize=(18, 10),nrows=5, ncols=5)
        # for i in range(data.size()[0]):
        #     row = i // 5
        #     col = i % 5
        #     im = axes[row][col].imshow(data[i],interpolation='bilinear',cmap=plt.cm.jet,origin='lower', vmax=0.02)
        #     fig.colorbar(im, ax=axes[row][col])
        # x = train_dataset.dataset.isel(time=23, lat=12).CRM_QV.squeeze().values
        # plt.imshow(data[0][0],interpolation='none',cmap=plt.cm.jet,origin='lower')  
        # many other colormaps can be seen here: http://matplotlib.org/examples/color/colormaps_reference.html
        
        # fig.savefig("2.png")
        # break


