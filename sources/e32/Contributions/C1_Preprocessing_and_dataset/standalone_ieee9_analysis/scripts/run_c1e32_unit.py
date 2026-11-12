"""Entry point for one externally authorized, hash-pinned unit."""
import argparse
import c1e32_worker
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('unit-id','role','assignment-sha256','configuration-sha256','configuration','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--expected-fits',type=int,required=True)
    c1e32_worker.execute(parser.parse_args())
