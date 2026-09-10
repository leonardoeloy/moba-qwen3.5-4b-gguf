import argparse
import hashlib
import json
import os
import re
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
if env.get('BENCH_CAPTURE') and Path(env['BENCH_CAPTURE']).exists():raise RuntimeError('Use a fresh capture path')
cmd=[str(root/'bin/bench'),str(root/'models'/lock['model']['filename']),a.prompt,str(a.tokens),str(out),str(a.generate),str(a.stride),a.reference]
start=time.monotonic()
with out.with_suffix('.log').open('w') as log:
    try:
        proc=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=log,timeout=a.timeout)
        result=dict(returncode=proc.returncode,stdout=proc.stdout)
        if proc.returncode==0:result.update(json.loads(proc.stdout))
    except subprocess.TimeoutExpired:result=dict(returncode=-1,error=f'{a.timeout} second timeout')
result.update(elapsed_s=time.monotonic()-start,command=cmd,environment={k:v for k,v in env.items() if k.startswith(('MOBA','GGML_VK_','BENCH_'))},prompt_sha256=hashlib.sha256(Path(a.prompt).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path('src/bench.cpp').read_bytes()).hexdigest(),vendor_diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C','vendor/llama.cpp','diff'])).hexdigest(),moba_sha256=hashlib.sha256(Path("src/moba.h").read_bytes()).hexdigest(),dependencies=lock)
log_text=out.with_suffix('.log').read_text()
kv=re.search(r'llama_kv_cache: size =\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers',log_text)
if kv:result['kv_cache']=dict(allocated_mib=float(kv[1]),cells=int(kv[2]),layers=int(kv[3]))
result['vulkan_compute_mib']=sum(map(float,re.findall(r'sched_reserve:\s+Vulkan\d+ compute buffer size =\s*([\d.]+) MiB',log_text)))
result['kv_sources']={name:hashlib.sha256(Path('src',name).read_bytes()).hexdigest() for name in ['kv_share.h','kv_adapter.h']}
if env.get('BENCH_ADAPTER'):result['adapter_sha256']=hashlib.sha256(Path(env['BENCH_ADAPTER']).read_bytes()).hexdigest()
out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if result['returncode']:raise SystemExit(result['returncode'])
