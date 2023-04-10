import pickle
import sys
import pdb
import os

if __name__ == '__main__':
    # fname = os.path.join(sys.argv[1], "metrics-per-snapshot.pt")
    fname = "/home/thao/Desktop/graduation thesis/Code/Time-Aware-Incremental-Embedding/experiments/DE-icews14-complex-patience-10-addition-deletion-up-weight-factor-1-self-kd-factor-1-KD-CE-historical-sampling-train-seq-len-10-num-samples-each-time-step-1000-sample-neg-entity-neg-rate-reservoir-50-seed-123/version_202204081444/metrics-per-snapshot.pt"
    with open(fname, 'rb') as fp:
        metrics = pickle.load(fp)

    for k, vs in metrics.items():
        print(k)
        if "overall" in k:
        # pdb.set_trace()
            if type(vs) == dict:
                for v in vs.values():
                    print(v)
            else:
                print(vs)
            print()
