import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from .nn import Transformer
from .tensor import Tensor,cross_entropy,no_grad
from .optim import SGD,Adam,AdamW
from .attention import dense_attention,streaming_attention

def system_info():
    info={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'backend':'numpy-cpu','dtype':'float64'}
    try: info['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): info['git_revision']=None
    return info

def train(config,output):
    cfg=json.loads(Path(config).read_text()); out=Path(output); out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    (out/'system.json').write_text(json.dumps(system_info(),indent=2))
    model=Transformer(**cfg['model']); tc=cfg['training']
    opt={'sgd':SGD,'adam':Adam,'adamw':AdamW}[tc['optimizer']](model.parameters(),lr=tc['lr'])
    # Deliberately synthetic periodic data: a correctness test, not generalization evidence.
    vocab=cfg['model']['vocab']; context=cfg['model']['context']; batch=tc['batch_size']
    ids=np.stack([(np.arange(context+1)+i)%vocab for i in range(batch)])
    x,y=ids[:,:-1],ids[:,1:]
    (out/'data.json').write_text(json.dumps({'kind':'synthetic-periodic-overfit','sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'unique_training_positions':int(y.size),'validation':None},indent=2))
    start=time.perf_counter(); metrics=[]
    with (out/'metrics.jsonl').open('w') as f:
        for step in range(tc['steps']):
            model.zero_grad(); loss=cross_entropy(model(x),y); loss.backward()
            row={'step':step,'loss_before_update':float(loss.data),'tokens_seen':(step+1)*y.size,**opt.diagnostics()}
            opt.step(); row['elapsed_seconds']=time.perf_counter()-start
            f.write(json.dumps(row)+'\n'); metrics.append(row)
    model.eval()
    with no_grad(): final=float(cross_entropy(model(x),y).data)
    np.savez(out/'weights.npz',**{name:p.data for name,p in model.named_parameters()})
    summary={'initial_loss':metrics[0]['loss_before_update'],'final_loss':final,'parameters':sum(p.data.size for p in model.parameters()),'steps':tc['steps'],'tokens_seen':tc['steps']*y.size,'seconds':time.perf_counter()-start,'claim':'Synthetic overfit only; no held-out validation or scaling result.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

def benchmark(output,seq,repeats):
    rng=np.random.default_rng(42); rows=[]
    for t in seq:
        q,k,v=[rng.normal(size=(1,2,t,16)) for _ in range(3)]
        reference=dense_attention(q,k,v)
        for name,fn in [('dense',dense_attention),('streaming',streaming_attention)]:
            result=fn(q,k,v); np.testing.assert_allclose(result,reference,atol=1e-10,rtol=1e-10)
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter(); fn(q,k,v); samples.append(time.perf_counter()-start)
            rows.append({'backend':'numpy-cpu','implementation':name,'context':t,'median_seconds':float(np.median(samples)),'samples_seconds':samples,'max_abs_error':float(np.max(np.abs(result-reference))),'dense_score_array_bytes':int(2*t*t*8)})
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps({'system':system_info(),'results':rows},indent=2)); print(f'Saved {output}')

def generate(checkpoint,prompt,count):
    run=Path(checkpoint); cfg=json.loads((run/'config.json').read_text()); model=Transformer(**cfg['model'])
    with np.load(run/'weights.npz',allow_pickle=False) as weights:
        for name,p in model.named_parameters():
            if weights[name].shape!=p.shape: raise ValueError('Checkpoint shape mismatch')
            p.data[:]=weights[name]
    ids=[int(i) for i in prompt.split(',')]
    if not ids or min(ids)<0 or max(ids)>=cfg['model']['vocab']: raise ValueError('Prompt token outside vocabulary')
    model.eval()
    with no_grad():
        for _ in range(count):
            logits=model(np.array([ids[-model.context:]]))
            ids.append(int(logits.data[0,-1].argmax()))
    print(','.join(map(str,ids)))

def main():
    parser=argparse.ArgumentParser(description='ForgeML CPU research baseline')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train'); p.add_argument('config'); p.add_argument('--output',required=True)
    p=sub.add_parser('benchmark'); p.add_argument('kind',choices=['attention']); p.add_argument('--output',default='results/attention.json'); p.add_argument('--seq',nargs='+',type=int,default=[32,64,128]); p.add_argument('--repeats',type=int,default=5)
    p=sub.add_parser('generate'); p.add_argument('--checkpoint',required=True); p.add_argument('--prompt',default='0,1'); p.add_argument('--tokens',type=int,default=16)
    args=parser.parse_args()
    if args.command=='train': train(args.config,args.output)
    elif args.command=='benchmark': benchmark(args.output,args.seq,args.repeats)
    else: generate(args.checkpoint,args.prompt,args.tokens)

if __name__=='__main__': main()
