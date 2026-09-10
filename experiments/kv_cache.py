import argparse
import json
import math
import os
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.chdir(root)
p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--tokens',type=int,default=4096)
p.add_argument('--prompt',default='data/technical.txt')
p.add_argument('--stride',type=int,default=8)
p.add_argument('--types',nargs='+',choices=['f16','q8_0','q4_0'],default=['f16','q8_0','q4_0'])
a=p.parse_args()
out=Path(a.out)
if out.exists() and any(out.iterdir()):raise RuntimeError('Use a fresh directory')
out.mkdir(parents=True,exist_ok=True)
protocol=dict(tokens=a.tokens,prompt=a.prompt,stride=a.stride,generate=64,topk=32,dense_tail=2,types=a.types,quality_gates=dict(ppl_ratio_max=1.02,continuation_agreement_min=0.95),timing='One pass per configuration; synchronized scored prefill and decode, no profiling; all configurations use the same context allocation',scope='Exploratory fixed-policy comparison, no statistical equivalence or general capability claim')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
rows=[]
trials=[('dense-f16','f16',0)]+[('moba-'+t,t,1) for t in a.types]
for name,kind,sparse in trials:
    env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_VK_','BENCH_'))}
    env.update(BENCH_KV=kind,MOBA=str(sparse),MOBA_TOPK='32',MOBA_DENSE_LAYERS='2')
    cmd=['python3','experiments/benchmark.py','--out',str(out/name),'--tokens',str(a.tokens),'--prompt',a.prompt,'--generate','64','--stride',str(a.stride),'--timeout','2400']
    if rows:cmd+=['--reference',str(out/'dense-f16.tokens')]
    subprocess.run(cmd,env=env,stdout=subprocess.PIPE,text=True,check=True)
    row=json.loads((out/(name+'.json')).read_text())
    row['name']=name
    base=rows[0] if rows else row
    x=(out/'dense-f16.scores.tsv').read_text().splitlines()
    y=(out/(name+'.scores.tsv')).read_text().splitlines()
    assert len(x)==len(y)>0
    assert all(s.split()[:2]==t.split()[:2] for s,t in zip(x,y))
    assert (out/'dense-f16.prompt.tokens').read_bytes()==(out/(name+'.prompt.tokens')).read_bytes()
    row['comparison']=dict(ppl_ratio=math.exp(row['nll']-base['nll']),continuation_agreement=row['continuation_matches']/max(1,row['decode_steps']),natural_argmax_matches=sum(s.split()[2]==t.split()[2] for s,t in zip(x,y)),prefill_speedup=base['prefill_s']/row['prefill_s'],decode_tps_ratio=row['decode_tps']/base['decode_tps'],kv_reduction=base['kv_cache']['allocated_mib']/row['kv_cache']['allocated_mib'])
    row['comparison']['quality_gate_pass']=row['comparison']['ppl_ratio']<=1.02 and row['comparison']['continuation_agreement']>=0.95
    rows.append(row)
    (out/'results.json').write_text(json.dumps(dict(protocol=protocol,trials=rows),indent=2)+'\n')
    print(json.dumps(dict(name=name,**row['comparison'],kv_cache=row['kv_cache'],prefill_s=row['prefill_s'],decode_tps=row['decode_tps'])),flush=True)
