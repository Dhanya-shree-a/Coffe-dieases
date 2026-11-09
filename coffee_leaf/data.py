import torch
from torchvision import datasets, transforms

SIZE=224
MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]

def tfm(train=False):
    ops=[transforms.Resize((SIZE,SIZE))]
    if train: ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(20),
                      transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.12)]
    ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]
    return transforms.Compose(ops)

def folder_datasets(data_dir, split_ratio=.2, seed=42):
    from pathlib import Path
    root=Path(data_dir)
    if (root/'train').is_dir():
        train_full=datasets.ImageFolder(root/'train',transform=tfm(True))
        train_eval=datasets.ImageFolder(root/'train',transform=tfm(False))
        if (root/'val').is_dir():
            val=datasets.ImageFolder(root/'val',transform=tfm(False))
        else:
            nv=max(1,int(len(train_full)*split_ratio))
            nt=len(train_full)-nv
            if nt<1: raise ValueError('Need more images to split train/validation.')
            tr,vs=torch.utils.data.random_split(train_full,[nt,nv],generator=torch.Generator().manual_seed(seed))
            val=torch.utils.data.Subset(train_eval,vs.indices)
            train_full=tr
        test=datasets.ImageFolder(root/'test',transform=tfm(False)) if (root/'test').is_dir() else None
        classes=train_full.dataset.classes if isinstance(train_full,torch.utils.data.Subset) else train_full.classes
        return train_full,val,test,classes
    full=datasets.ImageFolder(root,transform=tfm(True))
    eval_full=datasets.ImageFolder(root,transform=tfm(False))
    if len(full.classes)!=4: print('Warning: expected 4 classes; found',full.classes)
    nv=max(1,int(len(full)*split_ratio)); nt=len(full)-nv
    if nt<1: raise ValueError('Not enough images for train/validation split.')
    tr,vs=torch.utils.data.random_split(full,[nt,nv],generator=torch.Generator().manual_seed(seed))
    val=torch.utils.data.Subset(eval_full,vs.indices)
    return tr,val,None,full.classes
