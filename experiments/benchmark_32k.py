import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.chdir(root)
out=Path(sys.argv[1])
if out.exists() and any(out.iterdir()):raise RuntimeError('Use a fresh directory')
out.mkdir(parents=True,exist_ok=True)
protocol=dict(context=32768,prompt='data/code.txt',topk=32,block=64,dense_full_attention_layers=2,trials=['dense','moba'],selection='Same fixed top-32 policy as the 4K validation, now 32/512 blocks at the end; more aggressive sparsity, no tuning on this run',quality='512 natural-target samples in the second half, stride 32; up to 64 dense teacher-forced continuation tokens',profiling='One extra aligned 256-token batch after timed prefill/decode, after alignment padding. Vulkan timestamps serialize operations; diagnostic batch is excluded from main prefill/decode timings.',limitations='One timed pass per mode, one previously used code passage, scored rather than ordinary prefill, no task benchmark or equivalence claim',timeout_per_run_s=2400)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
rows=[]
for name in protocol['trials']:
    env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_VK_','BENCH_'))}
    env.update(MOBA=str(int(name=='moba')),MOBA_TOPK='32',MOBA_DENSE_LAYERS='2')
    cmd=['python3','experiments/benchmark.py','--out',str(out/name),'--tokens','32768','--prompt',protocol['prompt'],'--generate','64','--timeout','2400','--profile']
    if rows:cmd+=['--reference',str(out/'dense.tokens')]
    subprocess.run(cmd,env=env,stdout=subprocess.PIPE,text=True)
    row=json.loads((out/(name+'.json')).read_text());row['name']=name
    log=(out/(name+'.log')).read_text()
    if 'profile_begin' in log:
        section=log.split('profile_begin',1)[1].split('profile_end',1)[0]
        ops={}
        for op,count,us in re.findall(r'^(.+): (\d+) x [\d.e+]+ us = ([\d.e+]+) us',section,re.M):
            entry=ops.setdefault(op,dict(calls=0,us=0.0));entry['calls']+=int(count);entry['us']+=float(us)
        row['profile']=dict(operators=ops,total_operator_us=sum(x['us'] for x in ops.values()))
        (out/(name+'.profile.json')).write_text(json.dumps(row['profile'],indent=2)+'\n')
    rows.append(row)
    (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ['stdout','command','dependencies','profile']}),flush=True)
    if row['returncode']:raise SystemExit('Run failed; preserve and inspect its records')
def scores(name):return [line.split() for line in (out/(name+'.scores.tsv')).read_text().splitlines()]
a,b=scores('dense'),scores('moba')
assert len(a)==len(b)==512
assert all(x[:2]==y[:2] for x,y in zip(a,b))
assert (out/'dense.prompt.tokens').read_bytes()==(out/'moba.prompt.tokens').read_bytes()
deltas=[float(y[3])-float(x[3]) for x,y in zip(a,b)]
rng=random.Random(7)
boot=sorted(math.exp(sum(rng.choices(deltas,k=len(deltas)))/len(deltas)) for _ in range(2000))
summary=dict(prefill_speedup=rows[0]['prefill_s']/rows[1]['prefill_s'],prefill_time_reduction=1-rows[1]['prefill_s']/rows[0]['prefill_s'],ppl_ratio=math.exp(rows[1]['nll']-rows[0]['nll']),paired_bootstrap_ppl_ratio_95=[boot[49],boot[1949]],bootstrap_scope='Paired resampling of sampled positions; does not address sequence correlation or generalization',natural_argmax_matches=sum(x[2]==y[2] for x,y in zip(a,b)),natural_samples=len(a),continuation_agreement=rows[1]['continuation_matches']/max(1,rows[1]['decode_steps']),continuation_nll_delta=rows[1]['continuation_nll']-rows[0]['continuation_nll'],decode_tps_ratio=rows[1]['decode_tps']/rows[0]['decode_tps'])
(out/'results.json').write_text(json.dumps(dict(protocol=protocol,summary=summary,trials=rows),indent=2)+'\n')
print(json.dumps(summary),flush=True)
