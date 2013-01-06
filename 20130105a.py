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

def hamdist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def get_jeff_mut_trans(ascii_states, mu):
    k = len(ascii_states)
    mmu = np.zeros((k, k))
    for i, si in enumerate(ascii_states):
        for j, sj in enumerate(ascii_states):
            h = hamdist(si, sj)
            mmu[i, j] = (mu ** h) * ((1 - mu) ** (2 - h))
    return mmu

def get_mut_rate_matrix(ascii_states):
    k = len(ascii_states)
    pre_Q = np.zeros((k, k))
    for i, si in enumerate(ascii_states):
        for j, sj in enumerate(ascii_states):
            if hamdist(si, sj) == 1:
                pre_Q[i, j] = 1
    Q = pre_Q - np.diag(np.sum(pre_Q, axis=1))
    return Q

def main(args):

    # initialize some variables
    N = args.N
    mu = args.mu
    ascii_states = ['AB', 'Ab', 'aB', 'ab']
    k = len(ascii_states)

    # construct the microstate mutational transition matrix
    #mmu = get_jeff_mut(ascii_states, mu)
    Q = get_mut_rate_matrix(ascii_states)
    mmu = scipy.linalg.expm(mu*Q)

    # show the transition matrix
    print 'transition matrix:'
    print mmu
    print

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

