import json
import math
import os
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.chdir(root)
out=Path(sys.argv[1])
if out.exists() and any(out.iterdir()):raise RuntimeError('Use a fresh directory')
out.mkdir(parents=True,exist_ok=True)
protocol=dict(context=4096,prompt='data/technical.txt',topk=32,dense_full_attention_layers=2,selection='Frozen before running: retain the top-16/32-block terminal fraction from the 2K calibration',trials=['dense','moba','dense-repeat'],scope='One new passage, 64 sampled natural targets, 64 teacher-forced continuation tokens; not a general quality benchmark')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
rows=[]
for name in protocol['trials']:
    env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_VK_'))}
    env.update(MOBA=str(int(name=='moba')),MOBA_TOPK='32',MOBA_DENSE_LAYERS='2')
    cmd=['python3','experiments/benchmark.py','--out',str(out/name),'--tokens','4096','--prompt',protocol['prompt'],'--generate','64']
    if rows:cmd+=['--reference',str(out/'dense.tokens')]
    subprocess.run(cmd,env=env,stdout=subprocess.PIPE,text=True)
    row=json.loads((out/(name+'.json')).read_text());row['name']=name;rows.append(row)
    (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ['stdout','command','dependencies']}),flush=True)
    if row['returncode']:raise SystemExit('Validation failed')
summary=dict(dense_prefill_range=[min(rows[0]['prefill_s'],rows[2]['prefill_s']),max(rows[0]['prefill_s'],rows[2]['prefill_s'])],moba_prefill_s=rows[1]['prefill_s'],speedup_vs_fastest_dense=min(rows[0]['prefill_s'],rows[2]['prefill_s'])/rows[1]['prefill_s'],ppl_ratio=math.exp(rows[1]['nll']-rows[0]['nll']),continuation_agreement=rows[1]['continuation_matches']/rows[1]['decode_steps'])
(out/'results.json').write_text(json.dumps(dict(protocol=protocol,summary=summary,trials=rows),indent=2)+'\n')
