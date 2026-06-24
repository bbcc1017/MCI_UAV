import json, os, subprocess, time, yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
REPO="/home/ryu/MCI_UAV"; PY="/home/ryu/anaconda3/envs/UAV/bin/python"
MANIFEST=f"{REPO}/scenarios/manifests/sigungu_osrm_manifest.json"
LOG=f"{REPO}/experiment_logs/_baseline_psent_orch.log"
P=80
# ★ OMP/MKL/OPENBLAS/NUMEXPR=1 핀 필수 — 없으면 각 main.py 가 numpy BLAS 스레드(=코어수)를
#   띄워 loadavg 폭증(80×127스레드). 학습 런과 동일 핀.
env=dict(os.environ); env.update(
    MCI_CAP_GATE="psent",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    PYTHONPATH="src/rl_src:src/sim_src", PYTHONIOENCODING="utf-8")
def find_out(d,acc=[None]):
    if isinstance(d,dict):
        for k,v in d.items():
            if k=='output_path': acc[0]=v
            find_out(v,acc)
    return acc[0]
def psent_stat(cfgp):
    outp=find_out(yaml.safe_load(open(cfgp,encoding='utf-8')),[None])
    coord=os.path.basename(os.path.dirname(cfgp))
    return os.path.join(REPO, outp.lstrip('./'), coord, f"results_{coord}_psent_stat.txt")
def run_one(name,cfgp):
    try:
        sp=psent_stat(cfgp)
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
    open(LOG,'w').close(); log(f"[start {time.strftime('%m-%d %H:%M:%S')}] psent baseline {len(items)}시군구 P={P} (OMP=1)")
    ok=fail=skip=0; t0=time.time()
    with ThreadPoolExecutor(max_workers=P) as ex:
        futs={ex.submit(run_one,n,c):n for n,c in items}
        for i,fut in enumerate(as_completed(futs),1):
            name,st,dt=fut.result()
            if st=='ok':ok+=1
            elif st=='skip':skip+=1
            else:fail+=1
            log(f"[{i}/{len(items)}] {st:5} {name} ({dt:.0f}s) ok={ok} skip={skip} fail={fail}")
    log(f"[done {time.strftime('%m-%d %H:%M:%S')}] ok={ok} skip={skip} fail={fail} wall={(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":   # ★ 가드 — import 시 배치 우발 실행 방지
    main()
