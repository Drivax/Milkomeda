"""Reproduce representative runs, sweeps, uncertainty and numerical/performance checks."""
from dataclasses import replace,asdict
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from .config import Config,preset_config
from .simulation import simulate,save_run,run_provenance
from .sweeps import ensemble,write_rows
from .visualization import make_figures,make_animation,save

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    figures=output/"figures";figures.mkdir(exist_ok=True)
    provenance=run_provenance()
    (output/"experiment_provenance.json").write_text(json.dumps(provenance,indent=2))
    measured={}
    for name in ("current_estimate","high_transverse"):
        config=preset_config(name)
        data,metrics=simulate(config)
        path=output/"runs"/name/"simulation.h5"
        save_run(path,config,data,metrics)
        make_figures(data,metrics,path.parent/"figures")
        measured[name]=metrics
        print(name,metrics,flush=True)
        if name=="current_estimate":
            make_figures(data,metrics,figures)
            make_animation(data,figures/"05_simulation.gif",config.frame_every)
    small=replace(Config(),N=40)
    print("Transverse velocity: 5 values x 5 paired seeds",flush=True)
    ensemble(small,5,output/"sweep","andromeda_transverse_kms",np.linspace(0,100,5))
    distributions={"andromeda_transverse_kms":{"mean":17.0,"std":20.0},
                   "initial_distance_kpc":{"mean":785.0,"std":25.0}}
    print("Uncertainty: 20 runs",flush=True)
    ensemble(small,20,output/"uncertainty",distributions=distributions)
    (output/"uncertainty_distributions.json").write_text(json.dumps(distributions,indent=2))
    numerical=[]
    for parameter,values in (("dt",(1e6,2e6,4e6)),("softening",(2.5,5.,10.))):
        for value in values:
            config=replace(small,**{parameter:value})
            if parameter=="dt": config=replace(config,steps=round(6e9/value),sample_every=round(4e7/value))
            data,metrics=simulate(config,retain_particles=False)
            save_run(output/"numerical"/f"{parameter}_{value:g}.h5",config,data,metrics)
            numerical.append(dict(parameter=parameter,value=value,**metrics))
    write_rows(output/"numerical.csv",numerical)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,param,metric,xlabel,ylabel in (
        (axes[0],"dt","max_energy_drift_pct","Time step (Myr)","Max sampled energy drift (%)"),
        (axes[1],"softening","min_com_separation_kpc","Softening (kpc)","Minimum COM separation (kpc)")):
        rows=[r for r in numerical if r["parameter"]==param]
        ax.plot([r["value"]/(1e6 if param=="dt" else 1) for r in rows],[r[metric] for r in rows],"o-")
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
    fig.suptitle("Numerical sensitivity: same seed, N=40 per galaxy, 6 Gyr")
    fig.tight_layout();save(fig,figures/"06_numerical_sensitivity.png")
    perf=[]
    print("Performance: 50–1000 particles per galaxy, 3 repeats",flush=True)
    for n in (50,100,250,500,1000):
        for repeat in range(3):
            config=replace(small,N=n,steps=20,sample_every=20,validate=False,method="direct")
            _,metrics=simulate(config,retain_particles=False)
            perf.append(dict(N_per_galaxy=n,N_total=2*n,repeat=repeat,runtime_s=metrics["runtime_s"]))
    write_rows(output/"performance.csv",perf)
    (output/"performance_config.json").write_text(json.dumps(dict(base_config=asdict(small),
        overrides=dict(steps=20,sample_every=20,validate=False,method="direct"),
        repetitions=3,timing="Initialization + integration + sampled COM diagnostics; no IO/energy",
        note="Short runs measure scaling, not long-horizon physical accuracy"),indent=2))
    fig,ax=plt.subplots(figsize=(7,4))
    sizes=sorted({r["N_total"] for r in perf})
    values=[[r["runtime_s"] for r in perf if r["N_total"]==n] for n in sizes]
    means=np.mean(values,axis=1)
    ax.errorbar(sizes,means,yerr=np.std(values,axis=1,ddof=1),marker="o",label="Mean ± std, 3 repeats")
    ax.plot(sizes,means[0]*(np.array(sizes)/sizes[0])**2,ls=":",label="Quadratic guide")
    ax.set(xscale="log",yscale="log",xlabel="Total particles (2N)",ylabel="Runtime (s)",
           title="Direct solver: 20 integration steps, no energy diagnostics")
    ax.legend();ax.grid(alpha=.2);save(fig,figures/"07_runtime.png")
    shutil.copy2(output/"sweep"/"sensitivity.png",figures/"08_transverse_sensitivity.png")
    shutil.copy2(output/"uncertainty"/"uncertainty.png",figures/"09_uncertainty.png")
    (output/"measured_results.json").write_text(json.dumps(measured,indent=2))
    table="| Scenario | Minimum (kpc) | Time (Gyr) | Final (kpc) | Max energy drift (%) |\n|---|---:|---:|---:|---:|\n"
    for name,m in measured.items():
        table+=f'| {name} | {m["min_com_separation_kpc"]:.2f} | {m["min_com_separation_time_gyr"]:.2f} | {m["final_com_separation_kpc"]:.2f} | {m["max_energy_drift_pct"]:.4f} |\n'
    (output/"results.md").write_text(table)
    readme=Path(__file__).resolve().parents[1]/"README.md"
    content=readme.read_text(encoding="utf-8")
    begin,end="<!-- RESULTS:START -->","<!-- RESULTS:END -->"
    if output.resolve()==(readme.parent/"results").resolve() and begin in content:
        readme.write_text(content.split(begin)[0]+begin+"\n"+table+end+content.split(end)[1],encoding="utf-8")
    print(table,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=Path("results"))
    run(p.parse_args().output)
