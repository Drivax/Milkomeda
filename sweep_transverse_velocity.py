"""Generic paired-seed sweep; defaults to transverse velocity."""
import numpy as np
from src.config import parser,resolve
from src.sweeps import ensemble,SWEEP_UNITS

def main():
    p=parser(__doc__)
    p.add_argument("--parameter",choices=SWEEP_UNITS,default="andromeda_transverse_kms")
    p.add_argument("--vtrans-min","--minimum",dest="minimum",type=float,default=0)
    p.add_argument("--vtrans-max","--maximum",dest="maximum",type=float,default=100)
    p.add_argument("--vtrans-count","--count",dest="count",type=int,default=5)
    p.add_argument("--runs-per-v","--runs",dest="runs",type=int,default=5)
    p.add_argument("--output",default="results/sweep")
    args=p.parse_args()
    if args.count<2 or args.maximum<=args.minimum:p.error("Need count >=2 and maximum > minimum")
    ensemble(resolve(args),args.runs,args.output,args.parameter,np.linspace(args.minimum,args.maximum,args.count))
if __name__=="__main__":main()
