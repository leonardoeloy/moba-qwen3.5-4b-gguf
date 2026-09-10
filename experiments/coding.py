import argparse
import ast
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
p.add_argument('--topk',type=int,default=16)
p.add_argument('--dense-layers',type=int,default=2)
a=p.parse_args()
out=Path(a.out)
if out.exists() and any(out.iterdir()):raise RuntimeError('Use a fresh directory')
out.mkdir(parents=True,exist_ok=True)
allowed=set('Module FunctionDef arguments arg Return If IfExp Compare BoolOp BinOp UnaryOp Name Load Store Constant And Or Not Eq NotEq Lt LtE Gt GtE Add Sub Mult Div FloorDiv Mod USub UAdd Assign Pass Expr Call'.split())
def grade(text,task):
    match=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    code=match.group(1) if match else text.strip()
    tree=ast.parse(code)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('Expected one function')
    if tree.body[0].name!=task['function']:raise ValueError('Wrong function name')
    for node in ast.walk(tree):
        if type(node).__name__ not in allowed:raise ValueError('Unsupported syntax: '+type(node).__name__)
        if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in ('min','max','abs')):raise ValueError('Unsupported call')
        if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('Unsupported name')
    scope={'__builtins__':{},'min':min,'max':max,'abs':abs,'int':int,'float':float,'bool':bool}
    exec(compile(tree,'candidate','exec'),scope)
    checks=[scope[task['function']](*args)==expected for args,expected in task['cases']]
    return dict(passed=all(checks),checks=checks)
rows=[]
context=Path('data/code.txt').read_text()[:7000]
for task in json.loads(Path('data/coding.json').read_text()):
    prompt=out/(task['id']+'.prompt.txt')
    prompt.write_text('<|im_start|>system\nYou write concise correct Python functions.<|im_end|>\n<|im_start|>user\nReference source from an unrelated library; do not modify it:\n'+context+'\n\nYour independent task:\n'+task['request']+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    for mode in ('dense','moba'):
        env={k:v for k,v in os.environ.items() if not k.startswith(('MOBA','GGML_VK_'))}
        env.update(MOBA=str(int(mode=='moba')),MOBA_TOPK=str(a.topk),MOBA_DENSE_LAYERS=str(a.dense_layers))
        prefix=out/(task['id']+'-'+mode)
        start=time.monotonic()
        subprocess.run(['python3','experiments/benchmark.py','--out',str(prefix),'--tokens','0','--prompt',str(prompt),'--generate','160','--stride','0'],env=env,stdout=subprocess.PIPE,text=True)
        metrics=json.loads(prefix.with_suffix('.json').read_text())
        try:result=grade(prefix.with_suffix('.txt').read_text(),task) if not metrics['returncode'] else dict(passed=False,error='Inference failed')
        except Exception as e:result=dict(passed=False,error=str(e))
        row=dict(task=task['id'],mode=mode,elapsed_s=time.monotonic()-start,metrics=metrics,**result)
        rows.append(row)
        (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
summary={mode:dict(solved=sum(x['passed'] for x in rows if x['mode']==mode),tasks=sum(x['mode']==mode for x in rows),elapsed_s=sum(x['elapsed_s'] for x in rows if x['mode']==mode)) for mode in ('dense','moba')}
for v in summary.values():v['solved_per_hour']=3600*v['solved']/v['elapsed_s']
(out/'results.json').write_text(json.dumps(dict(scope='Three synthetic single-turn function tasks with unrelated context; no agent tools, training or SWE-bench; public deterministic cases, restricted Python grammar',task_sha256=hashlib.sha256(Path('data/coding.json').read_bytes()).hexdigest(),summary=summary,rows=rows),indent=2)+'\n')
