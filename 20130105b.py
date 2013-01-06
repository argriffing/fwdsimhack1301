"""
Draw a plot.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use('WX')

import matplotlib.pyplot as plt



def get_marginal_AB_Ab_distn(N, M, v):
    """
    @param N: population size
    @param M: population state definitions
    @param v: distribution over all population states
    @return: distribution over AB+Ab sums
    """
    distn = np.zeros(N+1)
    for (AB, Ab, aB, ab), p in zip(M, v):
        distn[AB + Ab] += p
    return distn


def main(args):
    """
    Assume that the data files are in known but ad-hoc formats.
    """

    # Get the population size from the command line.
    N = args.N

    # Load the continuous-time Moran exact distribution.
    M_cm = np.loadtxt(args.cm, dtype=int, usecols=(1, 2, 3, 4), skiprows=1)
    v_cm = np.loadtxt(args.cm, dtype=float, usecols=(5,), skiprows=1)

    # Load the Wright-Fisher-like exact distribution.
    M_wf = np.loadtxt(args.wf, dtype=int, usecols=(0, 1, 2, 3), skiprows=7)
    v_wf = np.loadtxt(args.wf, dtype=float, usecols=(4,), skiprows=7)

    # Load the simulation distribution.
    M_sm = np.loadtxt(args.sm, dtype=int, usecols=(0, 1, 2, 3))
    v_sm = np.loadtxt(args.sm, dtype=float, usecols=(4,))
    v_sm = v_sm / np.sum(v_sm)

    # Get the marginal distributions.
    y_cm = get_marginal_AB_Ab_distn(N, M_cm, v_cm)
    y_wf = get_marginal_AB_Ab_distn(N, M_wf, v_wf)
    y_sm = get_marginal_AB_Ab_distn(N, M_sm, v_sm)

    # Make the figure.
    x = np.arange(N+1)
    #plt.ylim(0, 2.0 / N)
    plt.ylim(0.02, 0.06)
    plt.xlim(-1, N+1)
    plt.plot(
            x, y_sm, 'ro',
            x, y_wf, 'go',
            x, y_cm, 'bo',
            )
    #plt.show()
    plt.savefig('comparison_AB_plus_Ab.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--N', type=int, default=20, help='population size')
    parser.add_argument(
            '--sm', help='path to Wright-Fisher-like simulation counts')
    parser.add_argument(
            '--wf', help='path to Wright-Fisher-like exact distribution')
    parser.add_argument(
            '--cm', help='path to continuous time Moran exact distribution')
    main(parser.parse_args())

