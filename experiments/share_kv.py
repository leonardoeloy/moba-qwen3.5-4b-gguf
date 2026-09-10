import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np

root=Path(__file__).resolve().parents[1]
os.chdir(root)
p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--work',required=True)
a=p.parse_args()
out,work=Path(a.out),Path(a.work)
for path in [out,work]:
    if path.exists() and any(path.iterdir()):raise RuntimeError('Use fresh directories')
    path.mkdir(parents=True,exist_ok=True)
protocol=dict(training_prompt='data/shakespeare.txt',training_tokens=2048,fit_tokens=1536,selection_tokens=512,validation_prompt='data/technical.txt',validation_tokens=4096,source_layer=3,target_layer=7,cache_type='f16',moba=False,adapter='Independent 256x256 affine attention-output map per query head, before the existing gate; original model weights frozen',training='Vulkan computes covariance matrices; CPU solves regularized normal equations and scores four ridge candidates',scope='One shared pair, one calibration passage and one separate validation passage; no full-model backpropagation or CED conversion')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_VK_','BENCH_'))}
env.update(MOBA='0',BENCH_KV='f16')
def run(name,prompt,tokens,generate,stride,options,reference=None):
    cmd=['python3','experiments/benchmark.py','--out',str(out/name),'--prompt',prompt,'--tokens',str(tokens),'--generate',str(generate),'--stride',str(stride)]
    if reference:cmd+=['--reference',str(reference)]
    subprocess.run(cmd,env=dict(env,**options),stdout=subprocess.PIPE,text=True,check=True)
    return json.loads((out/(name+'.json')).read_text())
for name,mask in [('teacher',0),('student',1)]:
    run(name,protocol['training_prompt'],2048,0,0,dict(BENCH_SHARE_MASK=str(mask),BENCH_CAPTURE=str(work/(name+'.f32'))))
assert (out/'teacher.prompt.tokens').read_bytes()==(out/'student.prompt.tokens').read_bytes()
assert json.loads((out/'teacher.json').read_text())['kv_cache']['layers']==8
assert json.loads((out/'student.json').read_text())['kv_cache']['layers']==7
x=np.fromfile(work/'student.f32',dtype='<f4').reshape(2048,16,256).astype(np.float64)
y=np.fromfile(work/'teacher.f32',dtype='<f4').reshape(2048,16,256).astype(np.float64)
assert np.isfinite(x).all() and np.isfinite(y).all()
n=protocol['fit_tokens']
mx,my=x[:n].mean(axis=0),y[:n].mean(axis=0)
for name,z,m in [('x',x,mx),('y',y,my)]:np.ascontiguousarray((z[:n]-m).transpose(1,2,0),dtype='<f4').tofile(work/(name+'.f32'))
subprocess.run(['g++','-O2','-std=c++17','src/kv_fit.cpp','-Ivendor/llama.cpp/ggml/include','-Lbuild/bin','-Wl,-rpath,$ORIGIN/../build/bin','-lggml','-lggml-base','-o','bin/kv_fit'],check=True)
proc=subprocess.run(['bin/kv_fit',str(n),str(work/'x.f32'),str(work/'y.f32'),str(work/'covariance.f32')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
(out/'vulkan.log').write_text(proc.stderr)
gpu=json.loads(proc.stdout)
gram,cross=np.fromfile(work/'covariance.f32',dtype='<f4').reshape(2,16,256,256).astype(np.float64)
gram=(gram+gram.transpose(0,2,1))/2
assert np.isfinite(gram).all() and np.isfinite(cross).all()
scale=np.trace(gram,axis1=1,axis2=2)/256
identity=np.eye(256)[None,:,:]
trials=[]
best=None
start=time.monotonic()
for ridge in [0.0001,0.001,0.01,0.1]:
    w=np.linalg.solve(gram+ridge*scale[:,None,None]*identity,cross.transpose(0,2,1)).transpose(0,2,1)
    bias=my-np.einsum('hij,hj->hi',w,mx)
    pred=np.einsum('hij,nhj->nhi',w,x,optimize=True)+bias
    train=float(np.mean((pred[:n]-y[:n])**2))
    validation=float(np.mean((pred[n:]-y[n:])**2))
    trials.append(dict(ridge=ridge,fit_mse=train,selection_mse=validation))
    if best is None or validation<best[0]:best=(validation,w,bias,ridge)
adapter=out/'adapter.weights'
with adapter.open('wb') as f:
    np.asarray(best[1],dtype='<f4').tofile(f)
    np.asarray(best[2],dtype='<f4').tofile(f)
training=dict(gpu=gpu,cpu_fit_and_selection_s=time.monotonic()-start,trials=trials,selected_ridge=best[3],unadapted_selection_mse=float(np.mean((x[n:]-y[n:])**2)),adapted_selection_mse=best[0],parameters=int(best[1].size+best[2].size),adapter_bytes=adapter.stat().st_size,adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),capture_sha256={name:hashlib.sha256((work/(name+'.f32')).read_bytes()).hexdigest() for name in ['teacher','student']},fit_source_sha256=hashlib.sha256(Path('src/kv_fit.cpp').read_bytes()).hexdigest())
(out/'training.json').write_text(json.dumps(training,indent=2)+'\n')
run('adapter-check',protocol['training_prompt'],256,0,0,dict(BENCH_SHARE_MASK='1',BENCH_ADAPTER=str(adapter),BENCH_CAPTURE=str(work/'adapted-check.f32')))
actual=np.fromfile(work/'adapted-check.f32',dtype='<f4').reshape(256,16,256)
expected=np.einsum('hij,nhj->nhi',best[1],x[:256],optimize=True)+best[2]
training['runtime_relative_rmse']=float(np.sqrt(np.mean((actual-expected)**2)/max(1e-12,np.mean(expected**2))))
(out/'training.json').write_text(json.dumps(training,indent=2)+'\n')
assert training['runtime_relative_rmse']<0.001
print(json.dumps(training),flush=True)
rows=[]
for name,options in [('dense',{}),('shared',dict(BENCH_SHARE_MASK='1')),('adapted',dict(BENCH_SHARE_MASK='1',BENCH_ADAPTER=str(adapter)))]:
    row=run(name,protocol['validation_prompt'],4096,64,8,options,out/'dense.tokens' if rows else None)
    base=rows[0] if rows else row
    assert (out/'dense.prompt.tokens').read_bytes()==(out/(name+'.prompt.tokens')).read_bytes()
    row['name']=name
    row['comparison']=dict(ppl_ratio=math.exp(row['nll']-base['nll']),continuation_agreement=row['continuation_matches']/max(1,row['decode_steps']),prefill_speedup=base['prefill_s']/row['prefill_s'],kv_reduction=base['kv_cache']['allocated_mib']/row['kv_cache']['allocated_mib'])
    row['comparison']['quality_gate_pass']=row['comparison']['ppl_ratio']<=1.02 and row['comparison']['continuation_agreement']>=0.95
    rows.append(row)
    (out/'results.json').write_text(json.dumps(dict(protocol=protocol,training=training,trials=rows),indent=2)+'\n')
    print(json.dumps(dict(name=name,**row['comparison'])),flush=True)
