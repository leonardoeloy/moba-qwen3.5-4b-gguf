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
p.add_argument('--tokens',type=int,default=2048)
a=p.parse_args()
out=Path(a.out)
if out.exists() and any(out.iterdir()):raise RuntimeError('Use a fresh directory')
out.mkdir(parents=True,exist_ok=True)
trials=[('dense',{}),('top8-tail2',dict(MOBA='1',MOBA_TOPK='8',MOBA_DENSE_LAYERS='2')),('top16-tail2',dict(MOBA='1',MOBA_TOPK='16',MOBA_DENSE_LAYERS='2')),('top16-tail4',dict(MOBA='1',MOBA_TOPK='16',MOBA_DENSE_LAYERS='4')),('dense-repeat',{})]
protocol=dict(trials=trials,tokens=a.tokens,prompt='data/code.txt',samples='second half, every 32 tokens',continuation='32 dense greedy tokens, teacher-forced for all later runs',gate='calibration only: perplexity ratio <=1.02 and continuation agreement >=0.95',selection='fastest passing trial including dense; not an independent quality claim',model_weights='unchanged')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
results=[]
for name,options in trials:
    env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_VK_'))}
    env.update(options)
    cmd=['python3','experiments/benchmark.py','--out',str(out/name),'--tokens',str(a.tokens)]
    if results:cmd+=['--reference',str(out/'dense.tokens')]
    proc=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,text=True)
    row=json.loads((out/(name+'.json')).read_text())
    row.update(name=name,options=options)
    if not row['returncode']:
        baseline=row if not results else results[0]
        row.update(speedup=baseline['prefill_s']/row['prefill_s'],ppl_ratio=math.exp(row['nll']-baseline['nll']),agreement=row['continuation_matches']/max(row['decode_steps'],1))
        row['passes_calibration']=row['ppl_ratio']<=1.02 and row['agreement']>=.95
    results.append(row)
    (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ['stdout','command','dependencies']}),flush=True)
    if len(results)==1 and row['returncode']:raise SystemExit('Dense baseline failed')
passing=[r for r in results[1:-1] if r.get('passes_calibration')]
best=min([results[0],*passing],key=lambda r:r['prefill_s'])
summary=dict(selected=best['name'],selected_options=best['options'],trials=results)
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
