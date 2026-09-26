"""Freeze matched development variants and the original MNIST-background task.

Only train=True datasets and the publisher's train member are read. The test
member in the downloaded archive is not extracted or parsed here.
"""
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from torchvision.datasets import MNIST,FashionMNIST,CIFAR10
from scripts.ndfa_strengthening_20260925.next_round import base,atomic_save


def build(root,data_dir,archive,split_seed=934005):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);inventory={}
    existing=root/'inventory.json'
    if existing.exists():return json.loads(existing.read_text())
    bg=CIFAR10(data_dir,train=True,download=False)
    gray=torch.as_tensor(bg.data[:,2:30,2:30]).float().mean(-1).flatten(1)/255.
    bgorder=torch.randperm(len(gray),generator=torch.Generator().manual_seed(split_seed+2))
    bgpools={'train':gray[bgorder[:40000]],'validation':gray[bgorder[40000:45000]]}
    bgidx={k:torch.randint(len(v),(10000 if k=='train' else 2000,),generator=torch.Generator().manual_seed(split_seed+3+(k=='validation'))) for k,v in bgpools.items()}
    def save(name,train,labels,val,vy,provenance):
        provenance.update(official_test_loaded=False,training_examples=len(train),validation_examples=len(val),split_seed=split_seed)
        provenance['hashes']={k:base.tensor_hash(v) for k,v in dict(train=train,labels=labels,validation=val,validation_labels=vy).items()}
        path=root/f'{name}.pt';atomic_save(dict(train=train,labels=labels,validation=val,validation_labels=vy,provenance=provenance,official_test_loaded=False),path)
        inventory[name]=dict(file=path.name,sha256=base.sha256(path),provenance=provenance)
    for dataset,cls in [('mnist',MNIST),('fashion',FashionMNIST)]:
        source=cls(data_dir,train=True,download=False);x=source.data.float().flatten(1)/255.;y=torch.as_tensor(source.targets,dtype=torch.long)
        order=torch.randperm(len(y),generator=torch.Generator().manual_seed(split_seed));vi,ti=order[:2000],order[2000:12000]
        for condition in ['clean','label_noise','input_nuisance']:
            tx,vx=x[ti].clone(),x[vi].clone();ty=y[ti].clone();vy=y[vi].clone()
            provenance=dict(dataset=dataset,condition=condition,training_identity_sha256=base.tensor_hash(ti),validation_identity_sha256=base.tensor_hash(vi),source_images_sha256=base.tensor_hash(source.data),source_labels_sha256=base.tensor_hash(y),recipe='pixel / 255; matched 10k/2k training-only foreground identities')
            if condition=='label_noise':
                rng=torch.Generator().manual_seed(split_seed+1);mask=torch.rand(len(ty),generator=rng)<.2
                replacements=torch.randint(1,10,(len(ty),),generator=rng);ty[mask]=(ty[mask]+replacements[mask])%10
                provenance.update(training_label_corruption=.2,realized_corruptions=int(mask.sum()),validation_labels='clean',label_noise_seed=split_seed+1)
            if condition=='input_nuisance':
                tx=.35*tx+.65*bgpools['train'][bgidx['train']];vx=.35*vx+.65*bgpools['validation'][bgidx['validation']]
                provenance.update(recipe='0.35 foreground + 0.65 independent grayscale 28x28 CIFAR-training background; no class-dependent assignment',background_identity_split_sha256=base.tensor_hash(bgorder),background_source_sha256=base.tensor_hash(torch.as_tensor(bg.data)),background_train_pool=40000,background_validation_pool=5000,background_reserved_test_pool=5000,standard_benchmark=False)
            save(f'{dataset}_{condition}',tx,ty,vx,vy,provenance)
    with zipfile.ZipFile(archive) as z:
        name='mnist_background_images_train.amat'
        with z.open(name) as f:values=np.loadtxt(f,dtype=np.float32)
    assert values.shape==(12000,785) and np.isfinite(values).all()
    x=torch.from_numpy(values[:,:784].copy());labels=torch.from_numpy(values[:,784].copy()).long()
    assert (labels>=0).all() and (labels<10).all() and torch.equal(labels.float(),torch.from_numpy(values[:,784].copy()))
    save('mnist_background_standard',x[:10000],labels[:10000],x[10000:],labels[10000:],dict(dataset='Larochelle MNIST-background-images',condition='native_input_nuisance',standard_benchmark=True,archive_sha256=base.sha256(Path(archive)),train_member=name,test_member_read=False,recipe='Publisher 12k development member: first 10k training, last 2k validation; original floating pixels; no additional augmentation'))
    existing.write_text(json.dumps(inventory,indent=2)+'\n');return inventory


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data-dir',required=True);p.add_argument('--archive',required=True);a=p.parse_args()
    print(json.dumps(build(a.root,a.data_dir,a.archive),indent=2))
