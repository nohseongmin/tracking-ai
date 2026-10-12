import argparse,json,lzma,random,math,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from torch import nn

p=argparse.ArgumentParser()
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--epochs',type=int,default=10)
p.add_argument('--prediction-seconds',type=float,required=True)
p.add_argument('--max-matches',type=int)
a=p.parse_args()
assert a.epochs>0
assert math.isfinite(a.prediction_seconds) and a.prediction_seconds>0
assert a.max_matches is None or a.max_matches>=10
INPUT_STEPS=6
random.seed(42); np.random.seed(42); torch.manual_seed(42)
assert torch.cuda.is_available(), 'CUDA required'
torch.set_num_threads(4)
a.output.mkdir(parents=True,exist_ok=True)
shards=a.output/'shards'; shards.mkdir(exist_ok=True)
files=sorted(a.dataset.rglob('*.json.xz'))
if a.max_matches is not None: files=files[:a.max_matches]
assert len(files)>=10, 'At least 10 match files required'
random.shuffle(files)
n=len(files); cut1=int(n*.70); cut2=cut1+int(n*.15)
splits=['train']*cut1+['validation']*(cut2-cut1)+['test']*(n-cut2)
manifest=[]; totals=Counter(); gaps=Counter(); maps=Counter(); errors=[]
train_sum=0.; train_count=0
started=time.time()
for fi,(file,split) in enumerate(zip(files,splits)):
    try:
        with lzma.open(file,'rt',encoding='utf-8') as f: d=json.load(f)
    except (OSError,EOFError,lzma.LZMAError,json.JSONDecodeError) as e:
        errors.append({'file':str(file),'error':str(e)}); continue
    rate=float(d['tickRate']); parse=int(d['parserParameters']['parseRate'])
    assert math.isfinite(rate) and rate>0 and parse>0
    prediction_steps=max(1,round(a.prediction_seconds*rate/parse))
    window_records=INPUT_STEPS+prediction_steps
    totals[split+'_playback_seconds']+=int(d.get('playbackTicks',0))/rate
    totals['playback_seconds']+=int(d.get('playbackTicks',0))/rate
    totals[split+'_playback_ticks']+=int(d.get('playbackTicks',0)); totals['matches']+=1; totals['playback_ticks']+=int(d.get('playbackTicks',0)); maps[d['mapName']]+=1
    inputs=[]; targets=[]; walking=[]
    for r in d.get('gameRounds',[]):
        totals['rounds']+=1; tracks=defaultdict(list)
        frames=r.get('frames') or []; totals['stored_frames']+=len(frames); totals[split+'_stored_frames']+=len(frames)
        for fr in frames:
            tick=int(fr['tick'])
            for side in ('t','ct'):
                for pl in (fr.get(side) or {}).get('players') or []:
                    totals['player_records']+=1
                    if not pl.get('isAlive'): continue
                    totals['alive_player_records']+=1
                    vals=[pl.get(k) for k in ('x','y','z')]
                    if any(v is None for v in vals): continue
                    tracks[str(pl['steamID'])].append((tick,*map(float,vals),float(bool(pl.get('isWalking')))))
        for track in tracks.values():
            track=sorted({row[0]:row for row in track}.values())
            for i in range(window_records-1,len(track)):
                block=np.asarray(track[i-window_records+1:i+1],dtype=np.float64)
                dt=np.diff(block[:,0]); gaps.update(map(int,dt))
                totals['candidate_windows']+=1
                if not np.all(dt==parse): totals['rejected_tick_gap']+=1; continue
                if not np.isfinite(block).all(): totals['rejected_nonfinite']+=1; continue
                # Nonmoving records are not evidence of a footstep. No interpolation through gaps.
                if np.any(np.linalg.norm(np.diff(block[:INPUT_STEPS,1:4],axis=0),axis=1)<1e-3):
                    totals['rejected_stationary_input']+=1; continue
                origin=block[INPUT_STEPS-1,1:4]
                x=np.column_stack((block[:INPUT_STEPS,1:4]-origin,block[:INPUT_STEPS,4]))
                y=block[-1,1:4]-origin
                inputs.append(x); targets.append(y); walking.append(block[INPUT_STEPS-1,4])
    count=len(inputs); totals[split+'_samples']+=count
    if count:
        x=np.asarray(inputs,dtype=np.float32); y=np.asarray(targets,dtype=np.float32)
        shard=shards/('%04d.npz'%fi)
        np.savez(shard,x=x,y=y,walking=np.asarray(walking,dtype=np.uint8))
        if split=='train': train_sum+=float(np.square(x[:,:,:3],dtype=np.float64).sum()); train_count+=x[:,:,:3].size
        manifest.append({'source':str(file.relative_to(a.dataset)),'split':split,'shard':shard.name,'samples':count,'tick_rate':rate,'parse_rate':parse,'input_window_seconds':INPUT_STEPS*parse/rate,'input_history_span_seconds':(INPUT_STEPS-1)*parse/rate,'prediction_gap_seconds':prediction_steps*parse/rate,'prediction_steps':prediction_steps})
    del d,inputs,targets
    if (fi+1)%25==0 or fi+1==n: print('PREPROCESS',fi+1,'/',n,dict(totals),flush=True)
summary={'totals':dict(totals),'maps':dict(maps),'errors':errors,'manifest':manifest,'seed':42,'input_shape':[6,4],'output_shape':[3],'assumption':'Historical player coordinates are simulated footstep locations, not real audio observations. isWalking=False after movement filtering is a simplified running proxy.','prediction_seconds_requested':a.prediction_seconds,'elapsed_preprocess_seconds':time.time()-started}
(a.output/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
assert all(totals[s+'_samples']>0 for s in ('train','validation','test')), 'Empty split'
scale=max(math.sqrt(train_sum/train_count),1.)
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(4,64,batch_first=True); self.head=nn.Linear(64,3)
    def forward(self,x):
        seq,_=self.gru(x); return self.head(seq[:,-1])
model=Model().cuda(); opt=torch.optim.Adam(model.parameters(),lr=.001)
loss_fn=nn.MSELoss(); batch=1024
history=[]; best=float('inf'); stale=0
for epoch in range(1,a.epochs+1):
    train=[m for m in manifest if m['split']=='train']; random.shuffle(train)
    model.train(); train_loss=0.; seen=0
    for m in train:
        with np.load(shards/m['shard']) as f: x=f['x']; y=f['y']
        x[:,:,:3]/=scale; y/=scale
        ids=np.random.permutation(len(x))
        for start in range(0,len(x),batch):
            ix=ids[start:start+batch]; xx=torch.from_numpy(x[ix]).cuda(); yy=torch.from_numpy(y[ix]).cuda()
            opt.zero_grad(); loss=loss_fn(model(xx),yy); loss.backward(); opt.step()
            train_loss+=loss.item()*len(ix); seen+=len(ix)
    model.eval(); val_sum=0.; val_n=0
    with torch.no_grad():
        for m in manifest:
            if m['split']!='validation': continue
            with np.load(shards/m['shard']) as f: x=f['x']; y=f['y']
            x[:,:,:3]/=scale
            for start in range(0,len(x),batch):
                pred=model(torch.from_numpy(x[start:start+batch]).cuda()).cpu().numpy()*scale
                err=np.linalg.norm(pred-y[start:start+batch],axis=1); val_sum+=err.sum(); val_n+=len(err)
    val=float(val_sum/val_n); row={'epoch':epoch,'training_mse_normalized':train_loss/seen,'validation_mean_distance':val}
    history.append(row); print('EPOCH',row,flush=True)
    (a.output/'training_history.json').write_text(json.dumps(history,indent=2))
    if val<best:
        best=val; stale=0; torch.save({'state_dict':model.state_dict(),'scale':scale,'epoch':epoch,'input_shape':[6,4],'prediction_seconds_requested':a.prediction_seconds},a.output/'best_model.pt')
    else:
        stale+=1
        if stale>=3: break
checkpoint=torch.load(a.output/'best_model.pt',weights_only=True); model.load_state_dict(checkpoint['state_dict']); model.eval()
all_errors=[]; all_states=[]
with torch.no_grad():
    for m in manifest:
        if m['split']!='test': continue
        with np.load(shards/m['shard']) as f: x=f['x']; y=f['y']; states=f['walking']
        x[:,:,:3]/=scale
        for start in range(0,len(x),batch):
            pred=model(torch.from_numpy(x[start:start+batch]).cuda()).cpu().numpy()*scale
            all_errors.append(np.linalg.norm(pred-y[start:start+batch],axis=1)); all_states.append(states[start:start+batch])
e=np.concatenate(all_errors); states=np.concatenate(all_states)
metrics={'test_samples':len(e),'mean_distance_units':float(e.mean()),'median_distance_units':float(np.median(e)),'p90_distance_units':float(np.percentile(e,90)),'walking_mean_distance_units':float(e[states==1].mean()) if (states==1).any() else None,'running_proxy_mean_distance_units':float(e[states==0].mean()) if (states==0).any() else None,'best_epoch':checkpoint['epoch'],'prediction_seconds_requested':a.prediction_seconds,'coordinate_units':'CSGO game units','total_elapsed_seconds':time.time()-started}
(a.output/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
print('COMPLETE',json.dumps(metrics),flush=True)

