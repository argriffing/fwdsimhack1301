"""
Draw another plot.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use('WX')

import matplotlib.pyplot as plt


RAW_RATE = 'raw-rate'
ADJUSTED_RATE = 'adjusted-rate'
DETERMINISTIC_MUT = 'deterministic-mut'
STOCHASTIC_MUT = 'stochastic-mut'


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

    # load the distributions
    distns = []
    filenames = []
    for rate_adjustment in (RAW_RATE, ADJUSTED_RATE):
        for mut_style in (DETERMINISTIC_MUT, STOCHASTIC_MUT):
            file_description = '_'.join((rate_adjustment, mut_style))
            filename = file_description + '.txt'
            filenames.append(filename)
            # get the distribution
            M = np.loadtxt(filename, dtype=int, usecols=range(4), skiprows=1)
            v = np.loadtxt(filename, dtype=float, usecols=(4,), skiprows=1)
            # append the marginal distribution
            y = get_marginal_AB_Ab_distn(N, M, v)
            distns.append(y)

    # Make the figure.
    fig = plt.figure()
    ax = plt.subplot(111)
    x = np.arange(N+1)
    colors = ('ro', 'go', 'bo', 'ko')
    for y, c, filename in zip(distns, colors, filenames):
        ax.plot(x, y, c, label=filename)
    plt.ylim(0.02, 0.08)
    plt.xlim(-1, N+1)
    ax.legend(loc='upper center')
    plt.savefig('four_distns.png')


    #x = np.arange(N+1)
    #plt.ylim(0.02, 0.08)
    #plt.xlim(-1, N+1)
    #plt.plot(
            #x, distns[0], 'ro',
            #x, distns[1], 'go',
            #x, distns[2], 'bo',
            #x, distns[3], 'ko',
            #)
    #plt.savefig('four_distns.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--N', type=int, default=20, help='population size')
    main(parser.parse_args())

