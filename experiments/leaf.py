import argparse
import ast
import hashlib
import json
import os
import re
import resource
import subprocess
import time
from pathlib import Path

import jinja2

root=Path(__file__).resolve().parents[1]
os.chdir(root)
p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--work',required=True)
p.add_argument('--turns',type=int,default=4)
p.add_argument('--generate',type=int,default=256)
p.add_argument('--concise-tools',action='store_true')
a=p.parse_args()
if not 1<=a.turns<=12 or not 32<=a.generate<=512:raise ValueError('Invalid bounded rollout budget')
out,work=Path(a.out),Path(a.work).resolve()
for d in [out,work]:
    if d.exists() and any(d.iterdir()):raise RuntimeError('Use fresh directories')
    d.mkdir(parents=True,exist_ok=True)
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    resource.setrlimit(resource.RLIMIT_NPROC,(2048,2048))
def shell(repo,command):
    cmd=['timeout','--kill-after=1','8','bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--setenv','PATH','/usr/bin:/bin','--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--bind',str(repo),'/repo','--chdir','/repo','/bin/bash','--noprofile','--norc','-c',command]
    with (work/'command-output.txt').open('w+') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,preexec_fn=limits)
        f.seek(0);return dict(returncode=r.returncode,output=f.read(4096))
probe=work/'probe';probe.mkdir()
if shell(probe,'printf ready')['output']!='ready':raise RuntimeError('Sandbox unavailable')
def path_in(repo,name):
    path=(repo/name).resolve()
    if not path.is_relative_to(repo):raise ValueError('Path outside task repository')
    return path
specs=[('read','Read a UTF-8 file.',{'path':'string'}),('write','Write a complete UTF-8 file.',{'path':'string','content':'string'}),('edit','Replace one exact occurrence of old_string in a file.',{'path':'string','old_string':'string','new_string':'string'}),('glob','List relative files matching a glob pattern.',{'pattern':'string'}),('bash','Run a shell command in the isolated task repository; no network.',{'command':'string'})]
tools=[dict(type='function',function=dict(name=n,description=d,parameters=dict(type='object',properties={k:dict(type=v) for k,v in fields.items()},required=list(fields),additionalProperties=False))) for n,d,fields in specs]
def execute(repo,name,args):
    fields=next((f for n,d,f in specs if n==name),None)
    if fields is None or set(args)!=set(fields):raise ValueError('Invalid tool arguments')
    if name=='bash':return shell(repo,args['command'])
    if name=='glob':
        if args['pattern'].startswith('/') or '..' in Path(args['pattern']).parts:raise ValueError('Invalid glob')
        return [str(x.relative_to(repo)) for x in sorted(repo.glob(args['pattern'])) if x.resolve().is_relative_to(repo)][:100]
    path=path_in(repo,args['path'])
    if name=='read':
        with path.open() as f:return f.read(4096)
    if name=='write':path.write_text(args['content']);return 'written'
    text=path.read_text()
    if not args['old_string'] or text.count(args['old_string'])!=1:raise ValueError('Expected exactly one match')
    path.write_text(text.replace(args['old_string'],args['new_string'],1));return 'edited'
allowed=set('Module FunctionDef arguments arg Return If IfExp Compare BoolOp BinOp UnaryOp Name Load Store Constant And Or Not Eq NotEq Lt LtE Gt GtE Add Sub Mult Div FloorDiv Mod USub UAdd Assign Pass Expr Call'.split())
def grade(code,task):
    try:
        tree=ast.parse(code)
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!=task['function']:raise ValueError('Expected the task function')
        for node in ast.walk(tree):
            if type(node).__name__ not in allowed:raise ValueError('Unsupported grading syntax')
            if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in ('min','max','abs')):raise ValueError('Unsupported call')
            if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('Unsupported name')
        scope={'__builtins__':{},'min':min,'max':max,'abs':abs}
        exec(compile(tree,'candidate','exec'),scope)
        checks=[scope[task['function']](*args)==expected for args,expected in task['cases']]
        return dict(passed=all(checks),checks=checks)
    except Exception as e:return dict(passed=False,error=str(e))
def fail(message):raise ValueError(message)
jenv=jinja2.Environment()
jenv.globals['raise_exception']=fail
template=jenv.from_string(Path('configs/chat_template.jinja').read_text())
lock=json.loads(Path('dependencies.json').read_text())
model=Path('models',lock['model']['filename'])
if sha(model)!=lock['model']['sha256']:raise RuntimeError('Wrong checkpoint')
env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_','BENCH_','LLAMA_'))}
env.update(MOBA='1',MOBA_TOPK='32',MOBA_DENSE_LAYERS='2',MOBA_COMPACT='1',BENCH_KV='q4_0')
tasks=json.loads(Path('data/leaf/tasks.json').read_text())
instruction='You are a concise coding agent. Work in /repo using the provided tools. Inspect the repository, fix the issue, and then give a short final summary. Use relative file paths. Only files in /repo persist.'
if a.concise_tools:instruction+=' Act directly with tool calls; do not narrate reasoning or examples before tools. After editing, finish with one short sentence.'
protocol=dict(system_prompt=instruction,concise_tools=a.concise_tools,scope='Independent Leaf-inspired smoke: two handcrafted synthetic repair tasks, greedy non-thinking Qwen3.5, no RL, no TaskPilot synthesis, no SWE-bench. Recompute conversation each turn; no prefix cache.',turn_limit=a.turns,generation_limit_per_turn=a.generate,tools=tools,environment=env.copy(),dependencies=lock,task_sha256=sha('data/leaf/tasks.json'),template_sha256=sha('configs/chat_template.jinja'),binary_sha256=sha('bin/bench'),libraries={f.name:sha(f) for f in Path('build/bin').glob('lib*.so')},grading='Hidden during interaction, public in released fixtures; restricted arithmetic Python grammar; task reference passes and starter fails. No unit-test suite.',jinja2=jinja2.__version__)
protocol['environment']={k:v for k,v in env.items() if k.startswith(('MOBA','BENCH_'))}
save(out/'protocol.json',protocol)
(out/'driver.py').write_bytes(Path(__file__).read_bytes())
(out/'tasks.json').write_bytes(Path('data/leaf/tasks.json').read_bytes())
rows=[]
for task in tasks:
    repo=work/task['id'];repo.mkdir()
    for name,text in task['files'].items():(repo/name).write_text(text)
    target=repo/(task['module']+'.py')
    baseline=grade(target.read_text(),task);reference=grade(task['reference'],task)
    if baseline['passed'] or not reference['passed']:raise RuntimeError('Invalid task fixture')
    messages=[dict(role='system',content=instruction),dict(role='user',content=task['issue'])]
    trace=[];start=time.monotonic();termination='turn_limit'
    for turn in range(a.turns):
        prefix=out/(task['id']+'-'+str(turn+1))
        prompt=Path(str(prefix)+'.prompt.txt')
        prompt.write_text(template.render(messages=messages,tools=tools,add_generation_prompt=True,enable_thinking=False))
        r=subprocess.run(['python3','experiments/benchmark.py','--out',str(prefix),'--tokens','0','--prompt',str(prompt),'--generate',str(a.generate),'--stride','0'],env=env,stdout=subprocess.DEVNULL)
        metrics=json.loads(Path(str(prefix)+'.json').read_text())
        if r.returncode:termination='inference_failure';trace.append(dict(turn=turn+1,metrics=metrics));break
        text=Path(str(prefix)+'.txt').read_text()
        messages.append(dict(role='assistant',content=text))
        calls=re.findall(r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>',text,re.S)
        record=dict(turn=turn+1,response=text,metrics=metrics,tools=[])
        trace.append(record)
        if metrics['decode_steps']>=a.generate:termination='generation_limit';break
        if not calls:
            termination='malformed_tool_call' if '<tool_call>' in text else 'natural_finish';break
        for name,body in calls:
            try:
                args={k:v.strip('\n') for k,v in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',body,re.S)}
                result=execute(repo,name,args)
            except Exception as e:result=dict(error=str(e))
            record['tools'].append(dict(name=name,result=result))
            messages.append(dict(role='tool',content=json.dumps(result)))
        save(out/(task['id']+'.trace.json'),trace)
        print(json.dumps(dict(task=task['id'],turn=turn+1,tools=[x['name'] for x in record['tools']])),flush=True)
    final=target.read_text() if target.is_file() and target.resolve().is_relative_to(repo) else ''
    (out/(task['id']+'.candidate.py')).write_text(final)
    save(out/(task['id']+'.trace.json'),trace)
    row=dict(task=task['id'],termination=termination,turns=len(trace),elapsed_s=time.monotonic()-start,baseline=baseline,reference=reference,grade=grade(final,task),generated_tokens=sum(x['metrics'].get('decode_steps',0) for x in trace),tool_calls=sum(len(x.get('tools',[])) for x in trace))
    rows.append(row);save(out/'progress.json',rows);print(json.dumps(row),flush=True)
save(out/'results.json',dict(protocol=protocol,rows=rows,solved=sum(x['grade']['passed'] for x in rows),tasks=len(rows),elapsed_s=sum(x['elapsed_s'] for x in rows)))
