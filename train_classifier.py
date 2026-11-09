import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix
from coffee_leaf.model import build_classifier ,build_efficientnet_b0  
from coffee_leaf.data import folder_datasets

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True); p.add_argument('--output-dir',default='artifacts')
    p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=.001); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--split-ratio',type=float,default=.2); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--model', choices=['mobilenet_v3', 'efficientnet_b0'], default='mobilenet_v3')
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr,va,te,classes=folder_datasets(a.data_dir,a.split_ratio,a.seed)
    print('Classes:',classes,'train:',len(tr),'val:',len(va),'test:',len(te) if te else 0,'device:',dev)
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev.type=='cuda')
    vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
    if a.model == 'efficientnet_b0':
        model = build_efficientnet_b0(len(classes), not a.no_pretrained).to(dev)
    else:
        model = build_classifier(len(classes), not a.no_pretrained).to(dev)
    loss_fn=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=2,factor=.3)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); best=-1 
    model_filename=(
        'efficientnet_b0_coffee.pth'
        if a.model=='efficientnet_b0'
        else 'mobilenetv3_coffee.pth'
    )
    for ep in range(a.epochs):
        model.train(); loss_sum=correct=total=0
        for x,y in tl:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
            z=model(x); loss=loss_fn(z,y); loss.backward(); opt.step()
            loss_sum+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); total+=len(y)
        model.eval(); vc=vt=vloss=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(dev),y.to(dev); z=model(x)   
                vloss+=loss_fn(z,y).item()*len(y); vc+=(z.argmax(1)==y).sum().item(); vt+=len(y)
        vacc=vc/max(vt,1); sched.step(vacc)
        print(f'Epoch {ep+1}/{a.epochs}: train_loss={loss_sum/max(total,1):.4f} train_acc={correct/max(total,1):.4f} val_loss={vloss/max(vt,1):.4f} val_acc={vacc:.4f}')
        if vacc > best:
            best = vacc
            torch.save({'model_state_dict': model.state_dict(), 'class_names': classes, 'image_size': 224}, out / model_filename)
            (out / 'class_names.json').write_text(json.dumps(classes, indent=2))

    print('Best validation accuracy:', round(best, 4))

    if te:
        ck = torch.load(out / model_filename, map_location=dev)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        ys = []
        ps = []

        with torch.no_grad():
            for x, y in DataLoader(te, batch_size=a.batch_size):
                ps.extend(model(x.to(dev)).argmax(1).cpu().tolist())
                ys.extend(y.tolist())

        print(classification_report(ys, ps, target_names=classes, zero_division=0))
        print('Confusion matrix (rows=true, columns=predicted):\n', confusion_matrix(ys, ps))


if __name__ == '__main__':
    main()