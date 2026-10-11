"""Regenerate scientific figures and an optional 3D GIF from a saved run."""
import argparse
import json
from pathlib import Path
import h5py
from src.visualization import make_figures,make_animation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,help="Optional GIF output path")
    p.add_argument("--frame-every",type=int,help="Override config snapshot stride")
    args=p.parse_args()
    if args.frame_every is not None and args.frame_every<1:p.error("frame-every must be positive")
    with h5py.File(args.input) as hf:
        data={k:hf[k][:] for k in hf}
        metrics=json.loads(hf.attrs["metrics_json"]);config=json.loads(hf.attrs["config_json"])
    make_figures(data,metrics,args.input.parent/"figures")
    if args.output:make_animation(data,args.output,args.frame_every or config["frame_every"])
if __name__=="__main__":main()
