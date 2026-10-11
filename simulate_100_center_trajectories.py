"""Monte Carlo COM trajectories with configurable Gaussian initial conditions."""
import json
from src.config import parser,resolve
from src.sweeps import ensemble

def main():
    p=parser(__doc__)
    p.add_argument("--runs",type=int,default=100)
    p.add_argument("--distributions",help='JSON file mapping parameters to {"mean":..., "std":...}; absent = particle sampling only')
    p.add_argument("--output",default="results/uncertainty")
    args=p.parse_args()
    distributions=json.load(open(args.distributions,encoding="utf-8")) if args.distributions else {}
    ensemble(resolve(args),args.runs,args.output,distributions=distributions)
if __name__=="__main__":main()
