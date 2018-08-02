"""
This weird stuff uses command line to call python. 
Turning it into a fuction too is not good but it works for now.
"""
from os import makedirs
from os.path import join, exists
import argparse
from multiprocessing import Pool
from subprocess import call

# I have no idea why this is necessary to use to call multiprocessing from inside a function.
# Otherwise an error: AttributeError: Can't pickle local object 'generate_data.<locals>._threaded_generation'
from pathos.multiprocessing import ProcessingPool as Pool


def run_generate_data(rollouts, logdir, threads, noise_type, exp_prob, randomness_factor, use_ctrl_exp, device):
    # rootdir=data_dir
    data_dir = join(logdir, 'train')
    if not exists(data_dir):
        mkdir(data_dir)

    rollouts=int(rollouts)
    threads=int(threads)

    rpt = rollouts // threads + 1

    def _threaded_generation(i):
        tdir = join(data_dir, 'thread_{}'.format(i))
        makedirs(tdir, exist_ok=True)

        cmd = ['xvfb-run', '-s', '"-screen 0 1400x900x24"']
        cmd += ['--server-num={}'.format(i + 1)]
        cmd += ["python", "data/carracing_ctrl.py", "--dir",
                tdir, "--rollouts", str(rpt)]
        cmd += ['--logdir', logdir]
        cmd += ['--noise_type', str(noise_type)]
        cmd += ['--exp_prob', str(exp_prob)]
        cmd += ['--randomness_factor', str(randomness_factor)]
        cmd += ['--device', device]
        if use_ctrl_exp:
            cmd += ['--use_ctrl_exp']
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
    parser.add_argument('--logdir', type=str, help="Directory to store rollout "
                        "directories of each thread")
    parser.add_argument('--noise_type', type=str, choices=['white', 'brown'], default='brown')
    parser.add_argument('--exp_prob', type=float, default=.5)
    parser.add_argument('--randomness_factor', type=float, default=.1)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--use_ctrl_exp', action='store_true')
    args = parser.parse_args()

    run_generate_data(args.rollouts, args.logdir, args.threads, args.noise_type, args.exp_prob, args.randomness_factor, args.use_ctrl_exp, args.device)



