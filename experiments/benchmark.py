import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.chdir(root)
p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--tokens',type=int,default=2048)
p.add_argument('--prompt',default='data/code.txt')
p.add_argument('--generate',type=int,default=32)
p.add_argument('--stride',type=int,default=32)
p.add_argument('--reference',default='-')
p.add_argument('--timeout',type=int,default=300)
p.add_argument('--profile',action='store_true')
a=p.parse_args()
out=Path(a.out)
if out.with_suffix('.json').exists() or out.with_suffix('.log').exists():raise RuntimeError('Use a fresh output prefix')
out.parent.mkdir(parents=True,exist_ok=True)
lock=json.loads(Path('dependencies.json').read_text())
env=dict(os.environ,GGML_VK_MAX_NODES_PER_SUBMIT='8',GGML_VK_SERIALIZE_SUBMISSIONS='1')
if a.profile:env['BENCH_PROFILE']='1'
cmd=[str(root/'bin/bench'),str(root/'models'/lock['model']['filename']),a.prompt,str(a.tokens),str(out),str(a.generate),str(a.stride),a.reference]
start=time.monotonic()
with out.with_suffix('.log').open('w') as log:
    try:
        proc=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=log,timeout=a.timeout)
        result=dict(returncode=proc.returncode,stdout=proc.stdout)
        if proc.returncode==0:result.update(json.loads(proc.stdout))
    except subprocess.TimeoutExpired:result=dict(returncode=-1,error=f'{a.timeout} second timeout')
result.update(elapsed_s=time.monotonic()-start,command=cmd,environment={k:v for k,v in env.items() if k.startswith(('MOBA','GGML_VK_','BENCH_'))},prompt_sha256=hashlib.sha256(Path(a.prompt).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path('src/bench.cpp').read_bytes()).hexdigest(),vendor_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C','vendor/llama.cpp','diff'])).hexdigest(),moba_sha256=hashlib.sha256(Path("src/moba.h").read_bytes()).hexdigest(),dependencies=lock)
out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if result['returncode']:raise SystemExit(result['returncode'])
