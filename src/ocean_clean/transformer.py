"""Direct masked reconstruction: observed values + masks, no interpolation features."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class MaskedWindows(Dataset):
    """Precomputed reproducible masks; original missing values never become labels."""
    def __init__(self, arrays, mean, std, samples=512, length=128, gaps=(1,6,24,72), seed=42):
        self.arrays=arrays; self.mean=mean; self.std=std; self.length=length
        if length <= max(gaps)+16:
            raise ValueError('Window needs context around largest gap')
        rng=np.random.default_rng(seed);self.examples=[]
        # Candidate windows are restricted to contiguous observed target gaps.
        # Never join years into artificial adjacent sequences.
        for i in range(samples):
            gap=int(gaps[i%len(gaps)])
            for attempt in range(10000):
                year=int(rng.integers(len(arrays)));a=arrays[year]
                start=int(rng.integers(0,len(a)-length+1))
                offset=int(rng.integers(8,length-gap-7))
                if np.isfinite(a[start+offset:start+offset+gap,0,0]).all():
                    self.examples.append((year,start,offset,gap));break
            else:raise ValueError('No eligible target gap')
    def __len__(self):return len(self.examples)
    def __getitem__(self,i):
        year,start,offset,gap=self.examples[i]
        original=self.arrays[year][start:start+self.length].copy()
        truth=(original[:,0,0]-self.mean[0,0])/self.std[0,0]
        selected=np.zeros(self.length,dtype=bool);selected[offset:offset+gap]=True
        original[selected,0,0]=np.nan
        observed=np.isfinite(original)
        values=np.where(observed,(original-self.mean)/self.std,0).astype('float32')
        return (torch.from_numpy(values),torch.from_numpy(observed.astype('float32')),
                torch.from_numpy(np.nan_to_num(truth).astype('float32')),torch.from_numpy(selected))


class OceanTransformer(nn.Module):
    """Temporal attention per buoy, then cross-buoy attention at each timestamp."""
    def __init__(self, variables, stations=2, length=128, width=32, layers=2):
        super().__init__()
        self.input=nn.Linear(variables*2,width)
        self.position=nn.Parameter(torch.randn(1,length,1,width)*.02)
        self.station=nn.Parameter(torch.randn(1,1,stations,width)*.02)
        block=nn.TransformerEncoderLayer(width,4,width*2,dropout=.1,batch_first=True)
        self.temporal=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.spatial=nn.MultiheadAttention(width,4,dropout=.1,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,1)
    def forward(self,values,observed):
        b,t,s,_=values.shape
        x=self.input(torch.cat([values,observed],dim=-1))+self.position[:,:t]+self.station
        x=self.temporal(x.permute(0,2,1,3).reshape(b*s,t,-1))
        x=x.reshape(b,s,t,-1).permute(0,2,1,3).reshape(b*t,s,-1)
        cross,_=self.spatial(x[:,0:1],x,x,need_weights=False)
        fused=self.norm(x[:,0:1]+cross)
        return self.output(fused).reshape(b,t)


def training_stats(arrays):
    values=np.concatenate(arrays,axis=0)
    mean=np.nanmean(values,axis=0);std=np.nanstd(values,axis=0)
    mean=np.nan_to_num(mean);std=np.where(np.isfinite(std)&(std>1e-6),std,1.)
    return mean.astype('float32'),std.astype('float32')
