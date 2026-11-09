import argparse,torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from coffee_leaf.model import build_classifier
from coffee_leaf.data import folder_datasets

p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--checkpoint',required=True)
p.add_argument('--batch-size',type=int,default=32); a=p.parse_args()
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_,val,test,_=folder_datasets(a.data_dir,.2,42); ds=test or val
ck=torch.load(a.checkpoint,map_location=dev); names=ck['class_names']
model=build_classifier(len(names),pretrained=False).to(dev); model.load_state_dict(ck['model_state_dict']); model.eval()
ys=[]; ps=[]
with torch.no_grad():
    for x,y in DataLoader(ds,batch_size=a.batch_size):
        ps.extend(model(x.to(dev)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
print('Evaluated on', 'test' if test else 'validation', 'split')
print('Accuracy:',accuracy_score(ys,ps))
print(classification_report(ys,ps,target_names=names,zero_division=0))
print('Confusion matrix (rows=true, columns=predicted):\n',confusion_matrix(ys,ps))
