"""
This should use the same process and notation as Jeff's C code.

The process is some variant of Wright-Fisher population genetics processes.
"""

import argparse
import numpy as np
import scipy.linalg
import omnomnomial

def get_stationary_distribution(P):
    nstates = len(P)
    b = np.zeros(nstates)
    A = P.T - np.eye(nstates)
    A[0] = np.ones(nstates)
    b[0] = 1
    v = np.abs(scipy.linalg.solve(A, b, overwrite_a=True, overwrite_b=True))
    return v / np.sum(v)

def gen_states(N, k):
    if k == 1:
        yield [N]
    else:
        for i in range(N+1):
            for suffix in gen_states(N-i, k-1):
                yield [i] + suffix

def main(args):

    # initialize some variables
    N = args.N
    mu = args.mu
    k = 4

    # construct the microstate mutational transition matrix
    mmu = np.zeros((k, k))
    ascii_states = ['AB', 'Ab', 'aB', 'ab']
    for i, si in enumerate(ascii_states):
        for j, sj in enumerate(ascii_states):
            hamdist = sum(1 for a, b in zip(si, sj) if a != b)
            mmu[i, j] = (mu ** hamdist) * ((1 - mu) ** (2 - hamdist))

    # construct the population genetic transition matrix
    M = np.array(list(gen_states(N, k)))
    lmcs = omnomnomial.get_lmcs(M)
    lps = np.log(np.dot(M, mmu) / N)
    P = np.exp(omnomnomial.get_log_transition_matrix(M, lmcs, lps))

    # compute the equilibrium distribution
    v = get_stationary_distribution(P)

    # print the equilibrium distribution
    print '\t'.join(ascii_states + ['probability'])
    for i, (AB, Ab, aB, ab) in enumerate(M):
        print '\t'.join(str(x) for x in (AB, Ab, aB, ab, v[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-N', type=int, default=2,
            help='haploid population size')
    parser.add_argument('-mu', type=float, default=0.005,
            help='mutation probability per site per generation')
    main(parser.parse_args())

