                      
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from prpf.model_customer_m3 import PRPF_SetGoGAN_Generator
from prpf.customer_meteonet_dataset import MeteoNetTxtPanguDataset
from prpf.meteonet_confidence import MeteoNetConfidenceHead, brier_score, expected_calibration_error

def read_ids(path): return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,default=0); p.add_argument('--steps',type=int,default=500); p.add_argument('--output',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--radar-root',type=Path,required=True); p.add_argument('--pangu-root',type=Path,required=True); p.add_argument('--pangu-stats',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); args=p.parse_args()
 device=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
 data=args.data_root; radar=str(args.radar_root); pangu=str(args.pangu_root); stats=str(args.pangu_stats)
 train=MeteoNetTxtPanguDataset(read_ids(data/'meteonet_train_5to20.txt'),radar,pangu,pangu_stats_path=stats)
 val=MeteoNetTxtPanguDataset(read_ids(data/'meteonet_test_5to20.txt'),radar,pangu,pangu_stats_path=stats)
 tr=DataLoader(train,batch_size=8,shuffle=True,num_workers=0); va=DataLoader(val,batch_size=8,shuffle=False,num_workers=0)
 model=PRPF_SetGoGAN_Generator(model_variant='m3',router_max_residual_dbz=0.15).to(device); ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False); state=ck.get('gen',ck.get('model',ck.get('state_dict',ck))) if isinstance(ck,dict) else ck; model.load_state_dict(state,strict=True); model.eval()
 for q in model.parameters(): q.requires_grad=False
 head=MeteoNetConfidenceHead().to(device); opt=torch.optim.AdamW(head.parameters(),lr=2e-4,weight_decay=1e-4); it=iter(tr)
 for step in range(args.steps):
  try: x,o,y=next(it)
  except StopIteration: it=iter(tr); x,o,y=next(it)
  x,o,y=x.to(device).float(),o.to(device).float(),y.to(device).float()
  with torch.no_grad(): pred,_=model(x,o)
  loss=head.loss(pred,x,o,y); opt.zero_grad(); loss.backward(); opt.step()
 head.eval(); probs=[]; labels=[]
 with torch.no_grad():
  for x,o,y in va:
   x,o,y=x.to(device).float(),o.to(device).float(),y.to(device).float(); pred,_=model(x,o); probs.append(head.probabilities(pred,x,o).cpu()); labels.append(head.targets(y).cpu())
 prob=torch.cat(probs); lab=torch.cat(labels); metrics={}
 for i,t in enumerate((12,24,32)):
  metrics[str(t)]={'brier':float(brier_score(prob[:,:,i],lab[:,:,i])),'ece':float(expected_calibration_error(prob[:,:,i],lab[:,:,i])),'event_rate':float(lab[:,:,i].mean()),'mean_probability':float(prob[:,:,i].mean())}
 args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'head':head.state_dict(),'metrics':metrics,'steps':args.steps,'source_checkpoint':str(args.checkpoint)},args.output); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
