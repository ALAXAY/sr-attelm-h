# SPDX-License-Identifier: GPL-3.0-only
# Adapter, data handling and empty-graph guard: Copyright (C) 2026 Oleksii Fesenko.
# The model classes Time2Vec, FeedForward, SelfAttentionBlock, ConditionalAttentionBlock and
# AsynchronousGraphGenerator are transcribed from ChristopherLey/AsyncGraphGenerator
# (commit 83ac07dad753536caeb2c38846822a7391a32569, AGG/model.py, AGG/utils.py), GPL-3.0.
# Modified in September 2026: classes combined into one file with the data adapter of the two-clock protocol.

# Official AGG model-core comparison adapter.
# Core architecture transcribed from:
# ChristopherLey/AsyncGraphGenerator, commit 83ac07dad753536caeb2c38846822a7391a32569
# AGG/model.py + AGG/utils.py. Data adapter is custom solely to enforce the frozen
# two-clock Beijing protocol used for SR-AttELM-H.

import sys, math, time, json, copy, argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_publication_methods as B
import data as D

torch.set_num_threads(int(__import__('os').environ.get('AGG_THREADS', '4')))
DEVICE="cpu"
P=4; N=12; SLOTS=N*P; BLOCKS=2; MAXN=SLOTS*BLOCKS

# ---- exact utility/core structure from official AGG ----
class Time2Vec(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(1, embedding_dim)
    def forward(self, tau: Tensor) -> Tensor:
        time_scaling = self.linear(tau)
        if len(tau.shape) == 2:
            periodic_embedding = torch.sin(time_scaling[:, 1:])
            time_embedding = torch.cat((time_scaling[:, 0:1], periodic_embedding), dim=1)
        else:
            periodic_embedding = torch.sin(time_scaling[:, :, 1:])
            time_embedding = torch.cat((time_scaling[:, :, 0:1], periodic_embedding), dim=2)
        return time_embedding

class FeedForward(nn.Module):
    def __init__(self,input_size,hidden_dim=None,output_size=None,dropout=.5,negative_slope=.2):
        if hidden_dim is None: hidden_dim=input_size
        if output_size is None: output_size=input_size
        super().__init__()
        self.ff=nn.Sequential(nn.Linear(input_size,hidden_dim),nn.LeakyReLU(negative_slope=negative_slope),
                              nn.Dropout(dropout),nn.Linear(hidden_dim,output_size))
    def forward(self,x): return self.ff(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,attention_drop=.2,dropout=.2,batch_first=True,
                 hidden_dim=None,output_dim=None,use_mask=True):
        super().__init__(); self.use_mask=use_mask
        self.norm1=nn.LayerNorm(embed_dim)
        self.self_attention=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,
            dropout=attention_drop,kdim=embed_dim,vdim=embed_dim,batch_first=batch_first)
        self.num_heads=num_heads
        if hidden_dim is None:hidden_dim=embed_dim*num_heads
        if output_dim is None:output_dim=embed_dim
        self.feed_forward=FeedForward(embed_dim,hidden_dim,output_dim,dropout)
        self.norm2=nn.LayerNorm(embed_dim)
    def forward(self,x,attention_mask,key_padding_mask=None):
        if self.use_mask:
            if len(attention_mask.shape)==3:
                Bn,Nn,_=attention_mask.shape
                multihead_mask=(attention_mask.unsqueeze(0).repeat(self.num_heads,1,1,1)
                                .transpose(1,0).reshape(-1,Nn,Nn))
            else: multihead_mask=attention_mask
            attention,attention_weights=self.self_attention(
                x,x,x,attn_mask=multihead_mask,key_padding_mask=key_padding_mask)
        else:
            attention,attention_weights=self.self_attention(x,x,x)
        attention=torch.nan_to_num(attention);attention_weights=torch.nan_to_num(attention_weights)
        x=x+attention
        x=x+self.feed_forward(self.norm1(x))
        x=self.norm2(x)
        return self.norm2(x),attention_weights

class ConditionalAttentionBlock(nn.Module):
    def __init__(self,target_dim,source_dim,num_heads,attention_drop=.2,dropout=.2,
                 batch_first=True,hidden_dim=None):
        super().__init__()
        self.norm1_source=nn.LayerNorm(source_dim)
        self.norm1_target=nn.LayerNorm(target_dim)
        self.cross_attention=nn.MultiheadAttention(embed_dim=target_dim,num_heads=num_heads,
            dropout=attention_drop,kdim=source_dim,vdim=source_dim,batch_first=batch_first)
        self.norm2=nn.LayerNorm(target_dim)
        if hidden_dim is None:hidden_dim=target_dim*num_heads
        self.feed_forward=FeedForward(target_dim,hidden_dim,target_dim,dropout)
        self.norm3=nn.LayerNorm(target_dim)
    def forward(self,target,source,key_padding_mask=None):
        key=self.norm1_source(source)
        # Official code intentionally does not pass key_padding_mask here.
        attention,attention_weights=self.cross_attention(
            query=self.norm1_target(target),key=key,value=key)
        attention=torch.nan_to_num(attention);attention_weights=torch.nan_to_num(attention_weights)
        x=target+attention
        x=x+self.feed_forward(self.norm2(x))
        x=self.norm3(x)
        return x,attention_weights

class AsynchronousGraphGenerator(nn.Module):
    def __init__(self,input_dim,feature_dim,num_heads,time_embedding_dim,num_node_types,
                 type_embedding_dim,num_spatial_components,spatial_embedding_dim,
                 num_categories,categorical_embedding_dim,num_layers,attention_drop=.2,
                 dropout=.2,query_includes_categorical=False,categorical_input=None,
                 query_includes_type=True,transfer_learning=False,use_mask=True):
        super().__init__();self.use_mask=use_mask
        self.node_feature_dim=(feature_dim+time_embedding_dim+type_embedding_dim+
                               spatial_embedding_dim+categorical_embedding_dim)
        self.feature_projection=nn.Linear(input_dim,feature_dim)
        self.query_includes_type=query_includes_type
        self.query_includes_categorical=query_includes_categorical
        if query_includes_categorical:
            self.query_dim=(time_embedding_dim+spatial_embedding_dim+categorical_embedding_dim+
                            (type_embedding_dim if query_includes_type else 0))
        else:
            self.query_dim=time_embedding_dim+spatial_embedding_dim+(type_embedding_dim if query_includes_type else 0)
        self.time_embed=Time2Vec(time_embedding_dim)
        self.type_embed=None if type_embedding_dim==0 else nn.Embedding(num_node_types,type_embedding_dim)
        self.spatial_embed=None if spatial_embedding_dim==0 else nn.Embedding(num_spatial_components,spatial_embedding_dim)
        self.num_categorical=num_categories
        if query_includes_categorical:
            if num_categories==-1:self.categorical_embedding=nn.Linear(categorical_input,categorical_embedding_dim)
            else:self.categorical_embedding=nn.Embedding(num_categories,categorical_embedding_dim)
        else:
            # Needed for source nodes exactly as official forward expects when category_index is present.
            self.categorical_embedding=nn.Embedding(num_categories,categorical_embedding_dim)
        self.agg_layers=nn.ModuleList([
            SelfAttentionBlock(self.node_feature_dim,num_heads,dropout=attention_drop,
                               batch_first=True,use_mask=use_mask)
            for _ in range(num_layers)])
        self.cross_attention=ConditionalAttentionBlock(self.query_dim,self.node_feature_dim,
                                                       num_heads,dropout=attention_drop)
        self.head=FeedForward(self.query_dim,self.query_dim*num_heads,input_dim,dropout)
    def forward(self,graph,device="cpu"):
        if len(graph.node_features.shape)<3:
            features=self.feature_projection(graph.node_features.unsqueeze(-1).to(device))
        else:features=self.feature_projection(graph.node_features.to(device))
        time_encode=self.time_embed(graph.time.unsqueeze(-1).to(device))
        source_list=[features,time_encode]
        if self.type_embed is not None and graph.type_index is not None:
            source_list.append(self.type_embed(graph.type_index.to(device)))
        if self.spatial_embed is not None and graph.spatial_index is not None:
            source_list.append(self.spatial_embed(graph.spatial_index.to(device)))
        if graph.category_index is not None:
            source_list.append(self.categorical_embedding(graph.category_index.to(device)))
        source=torch.cat(source_list,dim=-1)
        query_list=[self.time_embed(graph.target.time.unsqueeze(-1).to(device))]
        if self.spatial_embed is not None and graph.target.spatial_index is not None:
            query_list.append(self.spatial_embed(graph.target.spatial_index.to(device)))
        if self.query_includes_type and graph.target.type_index is not None and self.type_embed is not None:
            query_list.append(self.type_embed(graph.target.type_index.to(device)))
        target=torch.cat(query_list,dim=-1)
        key_padding_mask=graph.key_padding_mask.to(device) if graph.key_padding_mask is not None else None
        attn_mask=graph.attention_mask.to(device)
        hidden=source; total_attention=[]
        for layer in self.agg_layers:
            hidden,aw=layer(hidden,attn_mask,key_padding_mask);total_attention.append(aw)
        y_hat,aw=self.cross_attention(target,hidden,key_padding_mask);total_attention.append(aw)
        y_hat=self.head(y_hat).squeeze(-1)
        return y_hat,total_attention

# ---- two-clock adapter ----
def make_views(raw,index,mean,std,seed,mult=None,fold=None):
    if mult is not None:d=B.delays_mult(raw,index,seed,mult)
    else:d=D.delays(raw,index,seed,"shifted",fold)
    qt,qo,ratio,complete=B.build_binned(raw,index,d,mean,std)
    locf,gap=B.fill_locf(qo)
    return dict(qt=qt,qo=qo,ratio=ratio,complete=complete,locf=locf,gap=gap)

class AGGTargets(Dataset):
    def __init__(self,views,lo,hi,max_samples=None,seed=0,common_support=False,include_partial_target=True):
        rec=[]
        for vi,v in enumerate(views):
            for t in range(max(1,lo),hi):
                if common_support and not np.isfinite(v["qt"][t]).all():continue
                for i in range(N):
                    for p in range(P):
                        if np.isfinite(v["qt"][t,i,p]) and v["ratio"][t,i,p]<.999:
                            rec.append((vi,t,i,p))
        if max_samples and len(rec)>max_samples:
            rng=np.random.default_rng(seed);sel=np.sort(rng.choice(len(rec),max_samples,replace=False))
            rec=[rec[k] for k in sel]
        self.views=views;self.rec=rec;self.include_partial_target=include_partial_target
        # official PM25 block_size=2 => previous/current relative times 0.5/0
        self.times=np.r_[np.full(SLOTS,.5,np.float32),np.zeros(SLOTS,np.float32)]
        self.types=np.tile(np.arange(P,dtype=np.int64),N)
        self.types=np.r_[self.types,self.types]
        self.spatial=np.repeat(np.arange(N,dtype=np.int64),P)
        self.spatial=np.r_[self.spatial,self.spatial]
        self.category=np.zeros(MAXN,np.int64)
        self.attn=(self.times[None,:] < self.times[:,None])
    def __len__(self):return len(self.rec)
    def __getitem__(self,k):
        vi,t,ti,tp=self.rec[k];v=self.views[vi]
        values=np.zeros(MAXN,np.float32); pad=np.ones(MAXN,bool)
        idx=0
        for tt in (t-1,t):
            for i in range(N):
                for p in range(P):
                    q=v["qo"][tt,i,p]
                    if np.isfinite(q):
                        values[idx]=q;pad[idx]=False
                    idx+=1
        target_slot=SLOTS+ti*P+tp
        if not self.include_partial_target:
            values[target_slot]=0;pad[target_slot]=True
        elif v["ratio"][t,ti,tp]<=0:
            values[target_slot]=0;pad[target_slot]=True
        if pad[:SLOTS].all():   # adapter guard: previous-bin nodes may attend only to previous-bin nodes (causal
            pad[0]=False        # mask); if none of them was received, one neutral (zero) node avoids an all-masked row
        y=np.float32(v["qt"][t,ti,tp])
        return (torch.from_numpy(values),torch.from_numpy(self.times.copy()),
                torch.from_numpy(self.types.copy()),torch.from_numpy(self.spatial.copy()),
                torch.from_numpy(self.category.copy()),torch.from_numpy(pad),
                torch.from_numpy(self.attn.copy()),torch.tensor(tp,dtype=torch.long),
                torch.tensor(ti,dtype=torch.long),torch.tensor(y,dtype=torch.float32))

def collate(batch):
    vals,times,types,spats,cats,pads,attns,tps,tis,ys=zip(*batch)
    target=SimpleNamespace(
        time=torch.zeros((len(batch),1),dtype=torch.float32),
        type_index=torch.stack(tps).view(-1,1),
        spatial_index=torch.stack(tis).view(-1,1),
        features=torch.stack(ys).view(-1,1),
        category_index=None)
    return SimpleNamespace(node_features=torch.stack(vals),time=torch.stack(times),
        type_index=torch.stack(types),spatial_index=torch.stack(spats),category_index=torch.stack(cats),
        key_padding_mask=torch.stack(pads),attention_mask=torch.stack(attns),target=target)

def new_model(seed):
    torch.manual_seed(seed)
    # exact PM25 config dimensions from official pm25_config.yaml; counts adapted to selected 4 channels/12 stations.
    return AsynchronousGraphGenerator(input_dim=1,feature_dim=16,num_heads=8,time_embedding_dim=16,
        num_node_types=4,type_embedding_dim=16,num_spatial_components=12,spatial_embedding_dim=16,
        num_categories=1,categorical_embedding_dim=16,num_layers=2,attention_drop=.5,dropout=.2).to(DEVICE)

def train(model,trds,vads,max_epochs=18):
    tr=DataLoader(trds,batch_size=48,shuffle=True,collate_fn=collate,num_workers=0)
    va=DataLoader(vads,batch_size=96,shuffle=False,collate_fn=collate,num_workers=0)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    sch=torch.optim.lr_scheduler.LinearLR(opt,start_factor=1.0,end_factor=.1,total_iters=100)
    best=None;bad=0;t0=time.perf_counter();hist=[]
    for ep in range(max_epochs):
        model.train();sl=sn=0
        for g in tr:
            pred,_=model(g);loss=torch.mean((pred-g.target.features)**2)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            sl+=float(loss)*len(pred);sn+=len(pred)
        model.eval();se=n=0.
        with torch.no_grad():
            for g in va:
                pred,_=model(g);e=pred-g.target.features;se+=float((e*e).sum());n+=e.numel()
        vr=math.sqrt(se/max(n,1));hist.append({"epoch":ep,"train_mse":sl/max(sn,1),"val_rmse":vr,"lr":opt.param_groups[0]["lr"]})
        if best is None or vr<best[0]-1e-4:
            best=(vr,copy.deepcopy(model.state_dict()),ep);bad=0
        else:
            bad+=1
            if bad>=4 and ep>=7:break
        sch.step()
        print("epoch",ep,"val",round(vr,5),flush=True)
    model.load_state_dict(best[1]);return best,hist,time.perf_counter()-t0

def evaluate(model,ds):
    dl=DataLoader(ds,batch_size=128,shuffle=False,collate_fn=collate,num_workers=0)
    se=ae=n=0.;pse=pae=pn=0.;t0=time.perf_counter()
    model.eval()
    with torch.no_grad():
        for g in dl:
            pred,_=model(g);y=g.target.features;e=(pred-y).squeeze(1)
            se+=float((e*e).sum());ae+=float(e.abs().sum());n+=e.numel()
            pm=(g.target.type_index.squeeze(1)==0)
            if pm.any():
                pe=e[pm];pse+=float((pe*pe).sum());pae+=float(pe.abs().sum());pn+=int(pm.sum())
    return dict(RMSE=math.sqrt(se/n),MAE=ae/n,PM25_RMSE=math.sqrt(pse/max(pn,1)),
                PM25_MAE=pae/max(pn,1),n=n,inference_s=time.perf_counter()-t0)

def run_fold(fold,partial=True):
    raw,index,days,y,ym,stations,audit=D.load_raw();spec=D.P["folds"][fold]
    train_raw=raw[index<pd.Timestamp(spec["train_end"])]
    mean=np.nanmean(train_raw,(0,1));std=np.maximum(np.nanstd(train_raw,(0,1)),1e-5)
    btr=B.bidx(index,spec["train_end"]);bva=B.bidx(index,spec["validation_end"]);bte=B.bidx(index,spec["test_end"])
    aug=[make_views(raw,index,mean,std,101+(0 if fold=="A" else 1000),mult=m) for m in (1,2,4)]
    trds=AGGTargets(aug,0,btr,max_samples=7000,seed=910,include_partial_target=partial)
    vads=AGGTargets(aug,btr,bva,max_samples=1800,seed=911,include_partial_target=partial)
    model=new_model(920+(0 if fold=="A" else 100))
    best,hist,train_s=train(model,trds,vads,max_epochs=18)
    rows=[]
    for seed in (11,23,37):
        tv=make_views(raw,index,mean,std,seed,fold=fold)
        ds=AGGTargets([tv],bva,bte,common_support=True,include_partial_target=partial)
        met=evaluate(model,ds)
        rows.append({"fold":fold,"seed":seed,"method":"OfficialAGG-core","partial_target_input":partial,
                     "train_s":train_s,"best_val_rmse":best[0],"best_epoch":best[2],**met})
    meta={"fold":fold,"official_repo_commit":"83ac07dad753536caeb2c38846822a7391a32569",
          "official_model_file":"AGG/model.py","official_config_file":"pm25_config.yaml",
          "official_model_params":{"feature_dim":16,"num_heads":8,"time_embedding_dim":16,
            "type_embedding_dim":16,"spatial_embedding_dim":16,"categorical_embedding_dim":16,
            "num_layers":2,"attention_drop":0.5,"dropout":0.2},
          "adapter":{"channels":4,"stations":12,"block_size_bins":2,"bin_hours":6,
            "time_scale_hours":12,"category":"constant zero because wind-direction category is not part of frozen 4-channel protocol",
            "training_delay_levels":[1,2,4],"common_test_support":True,
            "include_partial_target_input":partial,"train_samples":len(trds),"val_samples":len(vads)},
          "training":{"optimizer":"Adam","lr":1e-4,"linear_lr_end_factor":0.1,"linear_lr_total_iters":100,
            "gradient_clip":1.0,"best_val_rmse":best[0],"best_epoch":best[2],"history":hist},
          "note":"Official AGG model core and official PM25 architectural hyperparameters; only the data loader/training harness is replaced to enforce the identical frozen two-clock protocol."}
    return rows,meta

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--fold",choices=["A","B"],required=True)
    ap.add_argument("--strict-remove-target",action="store_true")
    a=ap.parse_args();rows,meta=run_fold(a.fold,partial=not a.strict_remove_target)
    pd.DataFrame(rows).to_csv(str(Path(__file__).resolve().parent/f"official_AGG_same_protocol_{a.fold}_{D.MODE}.csv"),index=False)
    (Path(__file__).resolve().parent/f"official_AGG_same_protocol_{a.fold}_{D.MODE}_meta.json").write_text(json.dumps(meta,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
