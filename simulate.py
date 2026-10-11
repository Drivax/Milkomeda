"""Run a configurable softened N-body interaction; no merger prediction is implied."""
import json
from pathlib import Path
from src.config import parser,resolve
from src.simulation import simulate,save_run
from src.visualization import make_figures,make_animation

def parse_args(argv=None):
    p=parser(__doc__)
    p.add_argument("--output",type=Path,default=Path("results/runs/default/simulation.h5"))
    p.add_argument("--animate",action="store_true",help="Also render a 3D GIF")
    return p.parse_args(argv)

def main():
    args=parse_args(); config=resolve(args)
    data,metrics=simulate(config)
    save_run(args.output,config,data,metrics)
    figures=args.output.parent/"figures"
    make_figures(data,metrics,figures)
    if args.animate: make_animation(data,figures/"05_simulation.gif",config.frame_every)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
