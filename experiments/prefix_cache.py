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
p.add_argument('--work',required=True)
p.add_argument('--tokens',type=int,default=32768)
a=p.parse_args()
if a.tokens<=0 or a.tokens%256:raise ValueError('Prefix length must be a positive multiple of 256')
out,work=Path(a.out),Path(a.work)
for path in [out,work]:
    if path.exists() and any(path.iterdir()):raise RuntimeError('Use fresh output and working directories')
    path.mkdir(parents=True,exist_ok=True)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
lock=json.loads(Path('dependencies.json').read_text())
model=root/'models'/lock['model']['filename']
if sha(model)!=lock['model']['sha256']:raise RuntimeError('Model checksum mismatch')
model_signature=(model.stat().st_size,model.stat().st_mtime_ns)
subprocess.run(['g++','-O2','-std=c++17','src/prefix.cpp','-Ivendor/llama.cpp/include','-Ivendor/llama.cpp/ggml/include','-Lbuild/bin','-Wl,-rpath,$ORIGIN/../build/bin','-lllama','-lggml','-lggml-base','-o','bin/prefix'],check=True)
prefix=work/'prefix.txt'
prefix.write_text('<|im_start|>system\nYou explain source code clearly and concisely.<|im_end|>\n<|im_start|>user\nHere is an excerpt of C source code:\n'+Path('data/code.txt').read_text())
questions=['Summarize the purpose of the code excerpt in two sentences.','Describe one numerical edge case that deserves care in this code.','Explain the difference between quantization and dequantization in this code.']
policy=dict(MOBA='1',MOBA_TOPK='32',MOBA_DENSE_LAYERS='2',MOBA_COMPACT='1',GGML_VK_MAX_NODES_PER_SUBMIT='8',GGML_VK_SERIALIZE_SUBMISSIONS='1')
env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_','BENCH_','LLAMA_'))}
env.update(policy)
key=dict(model_sha256=lock['model']['sha256'],prefix_source_sha256=sha(prefix),prefix_tokens=a.tokens,n_ctx=a.tokens+512,batch=256,threads=4,cache_k='q4_0',cache_v='q4_0',policy=policy,backend_commit=lock['llama_cpp']['commit'],libraries={f.name:sha(f) for f in sorted(Path('build/bin').glob('lib*.so'))},native_binary_sha256=sha('bin/prefix'),sources={f:sha(Path('src',f)) for f in ['prefix.cpp','moba.h','kv_share.h','kv_adapter.h']})
protocol=dict(prefix_tokens=a.tokens,questions=questions,generated_tokens_max=32,key=key,scope='Exact-prefix checkpoints containing both attention KV and recurrent state. All requests use the same aligned batching and approximate attention/quantization policy. Ordinary prefix prefill without sampled logits. Independent processes; component timings exclude model/context initialization, process timings include it.',restore_io='Standard restores have a warm filesystem page cache because integrity verification reads the checkpoint. One extra restore requests POSIX_FADV_DONTNEED after verification; eviction is best effort and does not establish a cold storage-device cache.',quality='Compare all vocabulary logits at every generated step and require identical freely generated token files. This tests restoration fidelity, not dense-model equivalence or answer correctness.')
write(out/'protocol.json',protocol)
(out/'prefix.cpp').write_bytes(Path('src/prefix.cpp').read_bytes())
(out/'driver.py').write_bytes(Path(__file__).read_bytes())
state=work/'prefix.state'
manifest=work/'prefix.manifest.json'
rows=[]
def run(name,mode,index,reference=None):
    start=time.monotonic()
    if mode.startswith('restore'):
        metadata=json.loads(manifest.read_text())
        if metadata['key']!=key or sha(prefix)!=key['prefix_source_sha256']:raise RuntimeError('Checkpoint identity mismatch')
        if (model.stat().st_size,model.stat().st_mtime_ns)!=model_signature:raise RuntimeError('Model changed during experiment')
        if sha('bin/prefix')!=key['native_binary_sha256'] or any(sha(Path('build/bin',name))!=digest for name,digest in key['libraries'].items()):raise RuntimeError('Runtime changed during experiment')
        if sha(state)!=metadata['state_sha256']:raise RuntimeError('Checkpoint integrity mismatch')
    validation_s=time.monotonic()-start
    suffix=out/f'question-{index+1}.txt'
    suffix.write_text('\n\n'+questions[index]+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    cmd=['bin/prefix',mode,str(model),str(prefix),str(a.tokens),str(suffix),str(state),str(out/name),str(reference) if reference else '-',str(work/(name+'.logits.f32'))]
    launch=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        try:
            proc=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=log,text=True,timeout=2400)
            row=dict(name=name,mode=mode,question=index+1,returncode=proc.returncode,stdout=proc.stdout,command=cmd,validation_s=validation_s,process_elapsed_s=time.monotonic()-launch)
        except subprocess.TimeoutExpired:
            row=dict(name=name,returncode=-1,error='2400 second timeout',validation_s=validation_s,process_elapsed_s=time.monotonic()-launch)
    if row['returncode']==0:row.update(json.loads(row['stdout']))
    write(out/(name+'.json'),row)
    if row['returncode']:raise RuntimeError(f'{name} failed; records preserved')
    if reference:
        cold=out/('cold-'+str(index+1))
        row['tokens_identical']=(out/(name+'.tokens')).read_bytes()==Path(str(cold)+'.tokens').read_bytes()
        row['prompt_identical']=(out/(name+'.prompt.tokens')).read_bytes()==Path(str(cold)+'.prompt.tokens').read_bytes()
        row['restoration_exact']=row['tokens_identical'] and row['prompt_identical'] and row['checked_logits']>0 and row['bitwise_unequal_logits']==0
        write(out/(name+'.json'),row)
    rows.append(row)
    write(out/'progress.json',rows)
    print(json.dumps({k:v for k,v in row.items() if k not in ['stdout','command']}),flush=True)
    if reference and not row['restoration_exact']:raise RuntimeError('Restoration fidelity failed; do not promote cache reuse')
    if mode=='seed':
        ids=(out/(name+'.prompt.tokens')).read_text().splitlines()[:a.tokens]
        metadata=dict(key=key,prefix_token_ids_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),state_sha256=sha(state),state_bytes=state.stat().st_size)
        write(manifest,metadata);write(out/'checkpoint.json',metadata)
for i in range(3):
    run('cold-'+str(i+1),'seed' if i==0 else 'cold',i)
    run('restore-'+str(i+1),'restore',i,work/('cold-'+str(i+1)+'.logits.f32'))
    if i==0:run('restore-evict-1','restore-evict',i,work/'cold-1.logits.f32')
pairs=[]
for i in range(1,4):
    cold=next(x for x in rows if x['name']==f'cold-{i}')
    hit=next(x for x in rows if x['name']==f'restore-{i}')
    c=cold['prefix_s']+cold['suffix_s'];h=hit['validation_s']+hit['restore_s']+hit['suffix_s']
    pairs.append(dict(question=i,uncached_model_ready_ttft_s=c,validated_cached_model_ready_ttft_s=h,ttft_speedup=c/h,uncached_model_work_s=c+cold['decode_s'],validated_cached_model_work_s=h+hit['decode_s'],model_work_speedup=(c+cold['decode_s'])/(h+hit['decode_s']),fresh_process_speedup=(cold['process_elapsed_s']-cold['save_fsync_s'])/(hit['validation_s']+hit['process_elapsed_s'])))
uncached=sum(x['uncached_model_work_s'] for x in pairs)
amortized=pairs[0]['uncached_model_work_s']+rows[0]['save_fsync_s']+sum(x['validated_cached_model_work_s'] for x in pairs[1:])
summary=dict(checkpoint_bytes=state.stat().st_size,save_fsync_s=rows[0]['save_fsync_s'],pairs=pairs,three_distinct_requests_uncached_model_work_s=uncached,three_distinct_requests_with_cache_model_work_s=amortized,three_request_amortized_speedup=uncached/amortized,all_restores_bitwise_exact=all(x['restoration_exact'] for x in rows if x['mode'].startswith('restore')),interpretation='Model-work sums include synchronized prefix/suffix/decode work and per-hit integrity validation. They exclude model startup and logit-comparison instrumentation. The first cached request pays full prefix computation and durable save; subsequent requests restore. First cold reference subtraction removes the separately timed checkpoint save.')
write(out/'results.json',dict(protocol=protocol,summary=summary,trials=rows))
print(json.dumps(summary),flush=True)
