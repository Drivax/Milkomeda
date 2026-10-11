"""Local, explicitly submitted simulations and lightweight A/B comparison."""
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import h5py
import matplotlib.pyplot as plt
import streamlit as st
from src.config import Config,PRESETS,preset_config
from src.simulation import simulate,save_run,load_summary

ROOT=Path(__file__).resolve().parent

def main():
    st.set_page_config(page_title="Milkomeda Explorer",layout="wide")
    st.title("Milkomeda Explorer")
    st.caption("A simplified N-body experiment, not a prediction of the Milky Way's future.")
    with st.sidebar:
        selected=st.selectbox("Preset",list(PRESETS),key="preset")
        if st.button("Load preset / reset parameters",key="load_preset"):
            for key,value in asdict(preset_config(selected)).items():st.session_state["cfg_"+key]=value
        defaults=asdict(Config())
        with st.form("simulation"):
            st.subheader("Initial conditions")
            values={}
            labels={"initial_distance_kpc":"Initial distance (kpc)",
                    "andromeda_radial_kms":"M31 radial velocity (km/s)",
                    "andromeda_transverse_kms":"M31 transverse velocity (km/s)",
                    "mw_mass":"MW mass (solar masses)","m31_mass":"M31 mass (solar masses)",
                    "mw_radius_kpc":"MW halo scale radius (kpc)","m31_radius_kpc":"M31 halo scale radius (kpc)",
                    "mw_inclination_deg":"MW inclination (deg)","m31_inclination_deg":"M31 inclination (deg)",
                    "dispersion":"Velocity dispersion / circular speed",
                    "N":"Particles per galaxy (N)","dt":"Time step (yr)","steps":"Integration steps",
                    "softening":"Softening (kpc)","seed":"Seed","sample_every":"Save every (steps)",
                    "frame_every":"Animation stride (snapshots)","theta":"Barnes-Hut opening angle",
                    "close_threshold_kpc":"Close-approach threshold (kpc)"}
            for key,label in labels.items():
                if key=="N": st.subheader("Numerical simulation")
                widget="cfg_"+key
                if widget not in st.session_state: st.session_state[widget]=defaults[key]
                values[key]=st.number_input(label,key=widget,min_value=0,step=1) if isinstance(defaults[key],int) else st.number_input(label,key=widget)
            if "cfg_method" not in st.session_state:st.session_state["cfg_method"]=defaults["method"]
            values["method"]=st.selectbox("Force method",["auto","direct","bh"],key="cfg_method")
            values["validate"]=True
            submitted=st.form_submit_button("Run simulation",key="run_simulation")
        st.caption("Load a preset, edit values, then submit. Changing controls alone never starts a simulation.")
    if submitted:
        try:
            config=Config(**values)
            path=ROOT/"results"/"runs"/("ui_"+uuid4().hex[:10])/"simulation.h5"
            with st.spinner("Integrating and saving diagnostics…"):
                data,metrics=simulate(config)
                save_run(path,config,data,metrics)
            st.session_state["last_run"]=str(path)
            st.success("Simulation saved")
        except (ValueError,FloatingPointError) as error:st.error(str(error))
    paths=sorted(str(p) for p in (ROOT/"results"/"runs").glob("*/simulation.h5"))
    if not paths:
        st.info("Choose parameters and run your first simulation.")
        return
    default=paths.index(st.session_state["last_run"]) if st.session_state.get("last_run") in paths else 0
    selected_path=st.selectbox("Inspect simulation",paths,index=default,format_func=lambda p:Path(p).parent.name)
    data,metrics,config=load_summary(selected_path)
    labels={"min_com_separation_kpc":"Minimum COM separation (kpc)",
            "min_com_separation_time_gyr":"Time of minimum (Gyr)",
            "first_close_approach_time_gyr":"First crossing below threshold (Gyr)",
            "final_com_separation_kpc":"Final COM separation (kpc)",
            "max_energy_drift_pct":"Maximum energy drift (%)",
            "max_momentum_drift_relative":"Relative momentum drift",
            "max_com_drift_kpc":"Global COM drift (kpc)",
            "duration_gyr":"Simulated duration (Gyr)","runtime_s":"Runtime (s)"}
    cols=st.columns(3)
    for i,(key,label) in enumerate(labels.items()):
        value=metrics[key]
        cols[i%3].metric(label,"Not reached / unavailable" if value is None else f"{value:.4g}")
    st.caption(f"Diagnostics sampled every {config['sample_every']} steps; threshold {config['close_threshold_kpc']} kpc. Minimum is within the finite simulated interval.")
    if metrics["max_energy_drift_pct"] is not None and metrics["max_energy_drift_pct"]>1:
        st.warning("Sampled energy drift exceeds 1%. Check timestep and softening before interpreting trajectories.")
    st.subheader("Compare scenarios")
    comparison=st.multiselect("Runs to compare",paths,default=[selected_path],
                             format_func=lambda p:Path(p).parent.name)
    fig,ax=plt.subplots(figsize=(9,3.5));table=[]
    for path in comparison:
        summary,m,_=load_summary(path)
        ax.plot(summary["time"]/1e9,summary["separation"],label=Path(path).parent.name)
        table.append({"Scenario":Path(path).parent.name,**{label:m[key] for key,label in labels.items()}})
    ax.set(xlabel="Time (Gyr)",ylabel="COM separation (kpc)",title="Scenario comparison")
    if comparison:ax.legend(fontsize=8)
    ax.grid(alpha=.2);st.pyplot(fig);plt.close(fig)
    st.dataframe(table)
    if "energy_drift" in data:
        fig,ax=plt.subplots(figsize=(9,3))
        ax.plot(data["time"]/1e9,100*data["energy_drift"])
        ax.set(xlabel="Time (Gyr)",ylabel="Relative energy drift (%)",title="Numerical conservation")
        st.pyplot(fig);plt.close(fig)
    st.subheader("Particle snapshot")
    snapshot=st.slider("Saved snapshot",0,len(data["time"])-1,0)
    view=st.selectbox("Projection",["xy","xz","yz"])
    i,j={"xy":(0,1),"xz":(0,2),"yz":(1,2)}[view]
    # Read only the displayed snapshot; comparisons above never load particles.
    with h5py.File(selected_path) as hf:points=hf["pos"][snapshot]
    n=config["N"];fig,ax=plt.subplots(figsize=(8,5))
    for k,(name,color) in enumerate(zip(("Milky Way","Andromeda"),("#2563eb","#ea580c"))):
        cloud=points[k*n:(k+1)*n]
        stride=max(1,len(cloud)//3000)
        ax.scatter(cloud[::stride,i],cloud[::stride,j],s=5,label=name,color=color,alpha=.6)
    ax.set(xlabel=f"{view[0]} (kpc)",ylabel=f"{view[1]} (kpc)",
           title=f't = {data["time"][snapshot]/1e9:.3f} Gyr')
    ax.axis("equal");ax.legend();st.pyplot(fig);plt.close(fig)
    with st.expander("Saved parameters"):st.json(config)

if __name__=="__main__":main()
