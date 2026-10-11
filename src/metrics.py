import numpy as np

def center(pos, mass):
    return np.sum(pos*mass[:,None],axis=0)/mass.sum()

def encounter_metrics(times, separation, threshold):
    i=int(np.argmin(separation))
    hits=np.flatnonzero(separation<=threshold)
    return dict(min_com_separation_kpc=float(separation[i]),
                min_com_separation_time_gyr=float(times[i]/1e9),
                first_close_approach_time_gyr=float(times[hits[0]]/1e9) if hits.size else None,
                final_com_separation_kpc=float(separation[-1]),
                closest_at_boundary=bool(i in (0,len(times)-1)),
                close_threshold_kpc=threshold)

def diagnostics(pos,vel,mass,initial_center,initial_velocity,t,momentum_scale):
    momentum=np.sum(mass[:,None]*vel,axis=0)
    expected=initial_center+initial_velocity*t
    return (momentum, float(np.linalg.norm(momentum-mass.sum()*initial_velocity)/momentum_scale),
            float(np.linalg.norm(center(pos,mass)-expected)))
