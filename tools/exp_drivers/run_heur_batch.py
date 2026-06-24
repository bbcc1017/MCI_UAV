import json, os, subprocess, time, yaml, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
REPO="/home/ryu/MCI_UAV"; PY="/home/ryu/anaconda3/envs/UAV/bin/python"
MANIFEST=sys.argv[1]; GATE=sys.argv[2]; P=int(sys.argv[3]) if len(sys.argv)>3 else 60
SUFFIX="" if GATE=="occ" else "_psent"
TAG=os.path.splitext(os.path.basename(MANIFEST))[0]
LOG=f"{REPO}/experiment_logs/_baseline_{TAG}_{GATE}_orch.log"
env=dict(os.environ); env.update(MCI_CAP_GATE=GATE, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONPATH="src/rl_src:src/sim_src", PYTHONIOENCODING="utf-8")
def find_out(d,acc=[None]):
    if isinstance(d,dict):
        for k,v in d.items():
            if k=='output_path': acc[0]=v
            find_out(v,acc)
    return acc[0]
def stat_path(cfgp):
    outp=find_out(yaml.safe_load(open(cfgp,encoding='utf-8')),[None])
    coord=os.path.basename(os.path.dirname(cfgp))
    return os.path.join(REPO, outp.lstrip('./'), coord, f"results_{coord}{SUFFIX}_stat.txt")
def run_one(name,cfgp):
    try:
        sp=stat_path(cfgp)
        if os.path.exists(sp) and os.path.getsize(sp)>0: return name,'skip',0.0
    except Exception: pass
    t0=time.time()
    r=subprocess.run([PY,"src/sim_src/main.py","--config_path",cfgp,"--no_log"],
                     env=env,cwd=REPO,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return name,('ok' if r.returncode==0 else 'FAIL'),time.time()-t0
def log(m):
    with open(LOG,'a',encoding='utf-8') as f: f.write(m+"\n")
def main():
    m=json.load(open(MANIFEST,encoding='utf-8')); items=list(m.items())
    open(LOG,'w').close(); log(f"[start {time.strftime('%m-%d %H:%M:%S')}] {TAG} gate={GATE} {len(items)}개 P={P} (OMP=1)")
    ok=fail=skip=0; t0=time.time()
    with ThreadPoolExecutor(max_workers=P) as ex:
        futs={ex.submit(run_one,n,c):n for n,c in items}
        for i,fut in enumerate(as_completed(futs),1):
            name,st,dt=fut.result()
            ok+=(st=='ok'); skip+=(st=='skip'); fail+=(st=='FAIL')
            log(f"[{i}/{len(items)}] {st:5} {name} ({dt:.0f}s) ok={ok} skip={skip} fail={fail}")
    log(f"[done {time.strftime('%m-%d %H:%M:%S')}] ok={ok} skip={skip} fail={fail} wall={(time.time()-t0)/60:.1f}min")
if __name__=="__main__": main()
