import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

root=Path(__file__).resolve().parents[1]
os.chdir(root)
p=argparse.ArgumentParser()
p.add_argument('--model',action='store_true')
p.add_argument('--stock',action='store_true')
p.add_argument('-j',type=int,default=4)
a=p.parse_args()
lock=json.loads(Path('dependencies.json').read_text())
def run(*cmd):subprocess.run(cmd,check=True)
vendor=Path('vendor/llama.cpp')
if not vendor.exists():
    vendor.mkdir(parents=True)
    run('git','init',str(vendor))
    run('git','-C',str(vendor),'remote','add','origin',lock['llama_cpp']['url'])
    run('git','-C',str(vendor),'fetch','--depth=1','origin',lock['llama_cpp']['commit'])
    run('git','-C',str(vendor),'checkout','--detach','FETCH_HEAD')
assert subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()==lock['llama_cpp']['commit']
patch=lock['llama_cpp'].get('patch')
if patch:
    patch=str(root/patch)
    applied=subprocess.run(['git','-C',str(vendor),'apply','--reverse','--check',patch],capture_output=True).returncode==0
    if a.stock and applied:run('git','-C',str(vendor),'apply','--reverse',patch)
    if not a.stock and not applied:
        run('git','-C',str(vendor),'apply','--check',patch)
        run('git','-C',str(vendor),'apply',patch)
if a.stock:run('git','-C',str(vendor),'diff','--exit-code')
run('cmake','-S',str(vendor),'-B','build','-DCMAKE_BUILD_TYPE=Release','-DGGML_VULKAN=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_CURL=OFF')
run('cmake','--build','build','--target','llama','ggml-vulkan','-j',str(a.j))
Path('bin').mkdir(exist_ok=True)
run('g++','-O2','-std=c++17','src/bench.cpp','-Ivendor/llama.cpp/include','-Ivendor/llama.cpp/ggml/include','-Lbuild/bin','-Wl,-rpath,$ORIGIN/../build/bin','-lllama','-lggml','-lggml-base','-o','bin/bench')
if a.model:
    m=lock['model'];path=Path('models')/m['filename'];path.parent.mkdir(exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix('.gguf.download')
        urllib.request.urlretrieve(f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{m['filename']}",tmp)
        with tmp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['sha256']
        tmp.rename(path)
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['sha256']
print('Build ready')
