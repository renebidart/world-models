"""
This weird stuff uses command line to call python. 
Turning it into a fuction too is not good but it works for now.
"""
from os import makedirs
from os.path import join
import argparse
from multiprocessing import Pool
from subprocess import call

# I have no idea why this is necessary to use to call multiprocessing from inside a function.
# Otherwise an error: AttributeError: Can't pickle local object 'generate_data.<locals>._threaded_generation'
from pathos.multiprocessing import ProcessingPool as Pool


def generate_data(rollouts, rootdir, threads, policy='brown'):
    rollouts=int(rollouts)
    threads=int(threads)

    rpt = rollouts // threads + 1

    def _threaded_generation(i):
        tdir = join(rootdir, 'thread_{}'.format(i))
        makedirs(tdir, exist_ok=True)
        cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
        cmd += ['--server-num={}'.format(i + 1)]
        cmd += ["python", "data/carracing.py", "--dir",
                tdir, "--rollouts", str(rpt), "--policy", policy]
        cmd = " ".join(cmd)
        print(cmd)
        call(cmd, shell=True)
        return True

    with Pool(threads) as p:
        p.map(_threaded_generation, range(threads))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Total number of rollouts.")
    parser.add_argument('--threads', type=int, help="Number of threads")
    parser.add_argument('--rootdir', type=str, help="Directory to store rollout "
                        "directories of each thread")
    parser.add_argument('--policy', type=str, choices=['brown', 'white'],
                        help="Directory to store rollout directories of each thread",
                        default='brown')
    args = parser.parse_args()

    # rpt = args.rollouts // args.threads + 1

    # def _threaded_generation(i):
    #     tdir = join(args.rootdir, 'thread_{}'.format(i))
    #     makedirs(tdir, exist_ok=True)
    #     cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
    #     cmd += ['--server-num={}'.format(i + 1)]
    #     cmd += ["python", "data/carracing.py", "--dir",
    #             tdir, "--rollouts", str(rpt), "--policy", args.policy]
    #     cmd = " ".join(cmd)
    #     print(cmd)
    #     call(cmd, shell=True)
    #     return True

    # with Pool(args.threads) as p:
    #     p.map(_threaded_generation, range(args.threads))

    generate_data(args.rollouts, args.rootdir, args.threads, policy=args.policy)



