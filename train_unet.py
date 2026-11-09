import argparse,random
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from coffee_leaf.model import UNet
from coffee_leaf.data import MEAN,STD

class PairedMasks(Dataset):
    def __init__(self,images_dir,masks_dir,size=256):
        self.size=size; im=Path(images_dir); ma=Path(masks_dir)
        exts={'.jpg','.jpeg','.png','.bmp','.webp'}; self.pairs=[]
        for ip in sorted(im.iterdir()):
            if ip.suffix.lower() in exts:
                mp=ma/(ip.stem+'.png')
                if not mp.exists(): mp=ma/(ip.stem+'.jpg')
                if mp.exists(): self.pairs.append((ip,mp))
        if not self.pairs: raise ValueError('No matching image/mask pairs. Masks must share image basename.')
        self.tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        ip,mp=self.pairs[i]
        im=Image.open(ip).convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)
        ma=Image.open(mp).convert('L').resize((self.size,self.size),Image.Resampling.NEAREST)
        mask=torch.from_numpy((np.array(ma)>=128).astype('float32')).unsqueeze(0)
        return self.tf(im),mask

def dice_loss(z,y,eps=1e-6):
    p=torch.sigmoid(z); inter=(p*y).sum((2,3)); den=p.sum((2,3))+y.sum((2,3))
    return 1-((2*inter+eps)/(den+eps)).mean()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--images-dir',required=True); p.add_argument('--masks-dir',required=True)
    p.add_argument('--output',default='artifacts/unet_coffee.pth'); p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--lr',type=float,default=.001); a=p.parse_args()
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    ds=PairedMasks(a.images_dir,a.masks_dir); nv=max(1,int(.2*len(ds))); nt=len(ds)-nv
    if nt<1: raise ValueError('Need at least two image/mask pairs.')
    tr,va=random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(42))
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=UNet().to(dev)
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True); vl=DataLoader(va,batch_size=a.batch_size)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr); bce=nn.BCEWithLogitsLoss(); best=-1
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    for ep in range(a.epochs):
        model.train(); total=0
        for x,y in tl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
            z=model(x); loss=bce(z,y)+dice_loss(z,y); loss.backward(); opt.step(); total+=loss.item()
        model.eval(); dscores=[]; iscores=[]
        with torch.no_grad():
            for x,y in vl:
                z=torch.sigmoid(model(x.to(dev)))>=.5; y=y.to(dev).bool()
                inter=(z&y).sum().item(); union=(z|y).sum().item()
                dscores.append((2*inter+1e-6)/(z.sum().item()+y.sum().item()+1e-6))
                iscores.append((inter+1e-6)/(union+1e-6))
        d=float(np.mean(dscores)); print(f'Epoch {ep+1}/{a.epochs}: train_loss={total/max(len(tl),1):.4f} val_dice={d:.4f} val_iou={np.mean(iscores):.4f}')
        if d>best: best=d; torch.save({'model_state_dict':model.state_dict(),'image_size':256},out)
if __name__=='__main__': main()
