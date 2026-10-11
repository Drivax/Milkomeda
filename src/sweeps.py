"""Paired-seed sensitivity and explicitly configured Gaussian uncertainty."""
from dataclasses import asdict,replace
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from .simulation import simulate,run_provenance
from .visualization import save

SWEEP_UNITS={"andromeda_transverse_kms":"km/s","andromeda_radial_kms":"km/s",
             "initial_distance_kpc":"kpc","m31_mass":"solar masses","softening":"kpc"}

def write_rows(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ensemble(config,runs,output,parameter=None,values=None,distributions=None):
    if runs<1:raise ValueError("runs must be positive")
    if parameter is not None and parameter not in SWEEP_UNITS:raise ValueError("Unsupported sweep parameter")
    distributions=distributions or {}
    for key,spec in distributions.items():
        if key not in SWEEP_UNITS:raise ValueError(f"Unsupported uncertainty parameter: {key}")
        if set(spec)!={"mean","std"} or not np.isfinite(list(spec.values())).all() or spec["std"]<0:
            raise ValueError("Each Gaussian requires finite mean and non-negative std")
    if parameter in distributions:raise ValueError("Cannot both sweep and sample the same parameter")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    provenance=run_provenance()
    values=[None] if parameter is None else list(values)
    if not values:raise ValueError("Empty sweep")
    rows=[]; traces=[]; summaries=[]; times=None
    for value in values:
        group=[]
        for i in range(runs):
            rng=np.random.default_rng(config.seed+100000+i)
            overrides={k:float(rng.normal(**dict(loc=s["mean"],scale=s["std"]))) for k,s in distributions.items()}
            if parameter: overrides[parameter]=float(value)
            # Positive parameters are not silently clipped: invalid draws require a different distribution.
            case=replace(config,seed=config.seed+i,**overrides)
            data,metrics=simulate(case,retain_particles=False);times=data["time"]/1e9
            row=dict(value=value,run=i) | asdict(case) | metrics;rows.append(row);group.append(metrics)
            traces.append(data)
        summary={"value":value}
        for key in ("min_com_separation_kpc","min_com_separation_time_gyr","max_energy_drift_pct"):
            numbers=[r[key] for r in group if r[key] is not None]
            for label,q in (("p05",.05),("median",.5),("p95",.95)):
                summary[key+"_"+label]=float(np.quantile(numbers,q)) if numbers else None
        summaries.append(summary)
    write_rows(output/"runs.csv",rows);write_rows(output/"summary.csv",summaries)
    (output/"experiment.json").write_text(json.dumps(dict(base_config=asdict(config),runs_per_value=runs,
        parameter=parameter,values=values,gaussians=distributions,provenance=provenance,
        interpretation="5-95 percentiles of configured toy ensemble, not observational confidence intervals",
        seed_pairing="same particle and uncertainty seeds across sweep values"),indent=2),encoding="utf-8")
    arrays={k:np.stack([t[k] for t in traces]) for k in ("time","com_mw","com_m31","separation")}
    np.savez_compressed(output/"trajectories.npz",**arrays)
    if parameter:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,key,ylabel in zip(axes,("min_com_separation_kpc","min_com_separation_time_gyr"),
                                ("Minimum COM separation (kpc)","Time of minimum (Gyr)")):
            median=[s[key+"_median"] for s in summaries]
            ax.plot(values,median,"o-",label="Median")
            ax.fill_between(values,[s[key+"_p05"] for s in summaries],[s[key+"_p95"] for s in summaries],
                            alpha=.2,label="5–95 percentile")
            ax.set(xlabel=f'{parameter} ({SWEEP_UNITS[parameter]})',ylabel=ylabel)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle("Paired-seed sensitivity (finite simulation window)")
        fig.tight_layout();save(fig,output/"sensitivity.png")
    else:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for trace in traces:
            for key,color in (("com_mw","#2563eb"),("com_m31","#ea580c")):
                axes[0,0].plot(trace[key][:,0],trace[key][:,1],alpha=.3,lw=.8,color=color)
        axes[0,0].set(xlabel="x (kpc)",ylabel="y (kpc)",title="COM ensemble: blue MW, orange M31")
        sep=arrays["separation"];low,median,high=np.quantile(sep,[.05,.5,.95],axis=0)
        axes[0,1].plot(times,median,label="Median")
        axes[0,1].fill_between(times,low,high,alpha=.2,label="5–95 percentile")
        axes[0,1].set(xlabel="Time (Gyr)",ylabel="COM separation (kpc)",title="Uncertain initial conditions")
        axes[0,1].legend()
        for ax,key,label in zip(axes[1],("min_com_separation_kpc","min_com_separation_time_gyr"),
                               ("Minimum separation (kpc)","Time of minimum (Gyr)")):
            ax.hist([r[key] for r in rows],bins=min(10,runs),edgecolor="white")
            ax.set(xlabel=label,ylabel="Runs",title="Ensemble distribution")
        fig.tight_layout();save(fig,output/"uncertainty.png")
    return rows,summaries
