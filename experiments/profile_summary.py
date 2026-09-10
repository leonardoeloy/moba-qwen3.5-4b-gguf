from pathlib import Path
import json
import sys
p=Path(sys.argv[1])
for name in ['dense','moba']:
    f=p/(name+'.profile.json')
    if not f.exists():continue
    d=json.loads(f.read_text());groups={}
    for op,item in d['operators'].items():
        group='Flash-attention kernels' if op.startswith('FLASH_ATTN_EXT') else 'DeltaNet core' if 'GATED_DELTA_NET' in op else 'Matrix multiplies' if 'MUL_MAT' in op else 'Pooling/top-k/scatter' if op.startswith(('POOL_2D','TOP_K','SET_ROWS')) else 'Other kernels'
        groups[group]=groups.get(group,0)+item['us']
    d['groups']={k:dict(seconds=v/1e6,fraction=v/d['total_operator_us']) for k,v in groups.items()}
    d['scope']='One instrumented 256-token append batch after the timed 32K prefill and decode, at approximately 33K cached tokens. Kernel synchronization differs from ordinary execution; not a whole-prefill time breakdown. Router cost is spread across multiple categories.'
    f.write_text(json.dumps(d,indent=2)+'\n')
    print(name,json.dumps(d['groups']))
