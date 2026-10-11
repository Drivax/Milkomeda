from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np

COLORS=("#2563eb","#ea580c")
def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)

def make_figures(data,metrics,output):
    output=Path(output); times=data["time"]/1e9
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    fig,ax=plt.subplots(figsize=(8,5))
    for key,name,color in zip(("com_mw","com_m31"),("Milky Way","Andromeda"),COLORS):
        points=data[key]
        ax.plot(points[:,0],points[:,1],label=name,color=color)
        ax.scatter(*points[0,:2],color=color,marker="o")
        ax.scatter(*points[-1,:2],color=color,marker="x")
    ax.set(xlabel="x (kpc)",ylabel="y (kpc)",title="Galaxy centers: barycentric frame\nCircles: initial; crosses: final")
    ax.axis("equal");ax.legend();ax.grid(alpha=.2)
    save(fig,output/"01_center_trajectories.png")
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(times,data["separation"],color=COLORS[0],label="COM separation")
    ax.scatter(metrics["min_com_separation_time_gyr"],metrics["min_com_separation_kpc"],
               color=COLORS[1],label=f'Minimum: {metrics["min_com_separation_kpc"]:.1f} kpc',zorder=5)
    ax.axhline(metrics["close_threshold_kpc"],ls=":",color="gray",label="Close-approach threshold")
    if metrics["first_close_approach_time_gyr"] is not None:
        ax.axvline(metrics["first_close_approach_time_gyr"],ls="--",color=COLORS[1],label="First threshold crossing (sampled)")
    ax.set(xlabel="Time (Gyr)",ylabel="COM separation (kpc)",title="Galaxy separation through time")
    ax.legend();ax.grid(alpha=.2);save(fig,output/"02_separation.png")
    if "energy_drift" in data:
        fig,ax=plt.subplots(figsize=(8,4.5))
        ax.plot(times,100*data["energy_drift"],color=COLORS[0])
        ax.axhline(0,color="gray",lw=.7)
        ax.set(xlabel="Time (Gyr)",ylabel="(E(t) − E(0)) / |E(0)| (%)",
               title=f'Energy conservation: max sampled drift {metrics["max_energy_drift_pct"]:.3g}%')
        ax.grid(alpha=.2);save(fig,output/"03_energy_conservation.png")
    if "pos" in data:
        fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
        n=len(data["mass"])//2
        for ax,index,title in zip(axes,(0,-1),("Initial",f"Final: {times[-1]:.2f} Gyr")):
            for j,(name,color) in enumerate(zip(("Milky Way","Andromeda"),COLORS)):
                points=data["pos"][index,j*n:(j+1)*n]
                ax.scatter(points[:,0],points[:,1],s=6,alpha=.6,color=color,label=name)
            ax.set(xlabel="x (kpc)",ylabel="y (kpc)",title=title);ax.set_aspect("equal");ax.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        fig.suptitle("Particle distributions (all particles; fixed galaxy membership)")
        fig.tight_layout();save(fig,output/"04_initial_final.png")

def make_animation(data,path,frame_every=3):
    positions=data["pos"];n=positions.shape[1]//2
    frames=list(range(0,len(positions),frame_every))
    if frames[-1]!=len(positions)-1:frames.append(len(positions)-1)
    fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(projection="3d")
    lo=positions.min(axis=(0,1)); hi=positions.max(axis=(0,1))
    mid=(lo+hi)/2; radius=max((hi-lo).max()/2,1)
    ax.set(xlim=(mid[0]-radius,mid[0]+radius),ylim=(mid[1]-radius,mid[1]+radius),
           zlim=(mid[2]-radius,mid[2]+radius),xlabel="x (kpc)",ylabel="y (kpc)",zlabel="z (kpc)")
    ax.set_box_aspect((1,1,1))
    scatters=[ax.scatter([],[],[],s=4,alpha=.65,color=c,label=name)
              for c,name in zip(COLORS,("Milky Way","Andromeda"))]
    lines=[ax.plot([],[],[],color=c,lw=1)[0] for c in COLORS]
    centers=[ax.scatter([],[],[],marker="x",s=60,color=c) for c in COLORS]
    ax.legend()
    def update(i):
        for j,key in enumerate(("com_mw","com_m31")):
            points=positions[i,j*n:(j+1)*n];scatters[j]._offsets3d=tuple(points.T)
            com=data[key][:i+1];lines[j].set_data_3d(com.T)
            centers[j]._offsets3d=tuple(com[-1:,axis] for axis in range(3))
        ax.set_title(f'Simplified galaxy interaction — t = {data["time"][i]/1e9:.2f} Gyr')
        return scatters+lines+centers
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    FuncAnimation(fig,update,frames=frames,interval=100).save(path,writer=PillowWriter(fps=10),dpi=90)
    plt.close(fig)
