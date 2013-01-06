"""
Write equilibrium distributions for four related processes.

Each process is a Wright-Fisher variant.
"""

import argparse
import math

import numpy as np
import scipy.linalg

import omnomnomial


RAW_RATE = 'raw-rate'
ADJUSTED_RATE = 'adjusted-rate'
DETERMINISTIC_MUT = 'deterministic-mut'
STOCHASTIC_MUT = 'stochastic-mut'

def get_inverse_dict(M):
    """
    The input M[i,j] is count of allele j for pop state index i.
    The output T[(i,j,...)] maps allele count tuple to pop state index
    @param M: multinomial state map
    @return: T
    """
    return dict((tuple(state), i) for i, state in enumerate(M))

def get_macro_mut_rate_matrix(M, T):
    nstates = M.shape[0]
    pre_Q = np.zeros((nstates, nstates))
    for i, (AB, Ab, aB, ab) in enumerate(M):
        if AB > 0:
            pre_Q[i, T[AB-1, Ab+1, aB,   ab  ]] = AB
            pre_Q[i, T[AB-1, Ab,   aB+1, ab  ]] = AB
        if Ab > 0:
            pre_Q[i, T[AB+1, Ab-1, aB,   ab  ]] = Ab
            pre_Q[i, T[AB,   Ab-1, aB,   ab+1]] = Ab
        if aB > 0:
            pre_Q[i, T[AB+1, Ab,   aB-1, ab  ]] = aB
            pre_Q[i, T[AB,   Ab,   aB-1, ab+1]] = aB
        if ab > 0:
            pre_Q[i, T[AB,   Ab+1, aB,   ab-1]] = ab
            pre_Q[i, T[AB,   Ab,   aB+1, ab-1]] = ab
    Q = pre_Q - np.diag(np.sum(pre_Q, axis=1))
    return Q


def get_adjusted_rate(mu):
    """
    Get a rate that gives an expm matrix with the same entries as Jeff has.
    """
    return -0.25 * math.log(1 - 4*mu*(1-mu))

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
    #FIXME: unused here, maybe add a test or something
    k = len(ascii_states)
    mmu = np.zeros((k, k))
    for i, si in enumerate(ascii_states):
        for j, sj in enumerate(ascii_states):
            h = hamdist(si, sj)
            mmu[i, j] = (mu ** h) * ((1 - mu) ** (2 - h))
    return mmu

def get_micro_mut_rate_matrix(ascii_states):
    k = len(ascii_states)
    pre_Q = np.zeros((k, k))
    for i, si in enumerate(ascii_states):
        for j, sj in enumerate(ascii_states):
            if hamdist(si, sj) == 1:
                pre_Q[i, j] = 1
    Q = pre_Q - np.diag(np.sum(pre_Q, axis=1))
    return Q


def main(args):

    # do some initialization
    ascii_states = ['AB', 'Ab', 'aB', 'ab']
    k = len(ascii_states)
    N = args.N
    M = np.array(list(gen_states(N, k)))
    T = get_inverse_dict(M)
    lmcs = omnomnomial.get_lmcs(M)
    lps_neutral = np.log(M / float(N))

    # write a file for each of the 2x2 option combos
    for rate_adjustment in (RAW_RATE, ADJUSTED_RATE):
        for mut_style in (DETERMINISTIC_MUT, STOCHASTIC_MUT):

            # define the mutation rate between adjacent states
            if rate_adjustment == RAW_RATE:
                mu = args.mu
            elif rate_adjustment == ADJUSTED_RATE:
                mu = get_adjusted_rate(args.mu)
            else:
                raise Exception
            
            # construct population genetic transition matrix P
            if mut_style == DETERMINISTIC_MUT:
                Q = get_micro_mut_rate_matrix(ascii_states)
                mmu = scipy.linalg.expm(mu*Q)
                lps = np.log(np.dot(M, mmu) / N)
                P = np.exp(omnomnomial.get_log_transition_matrix(M, lmcs, lps))
            elif mut_style == STOCHASTIC_MUT:
                Q_mut = get_macro_mut_rate_matrix(M, T)
                P_mut = scipy.linalg.expm(mu*Q_mut)
                P_dft = np.exp(omnomnomial.get_log_transition_matrix(
                    M, lmcs, lps_neutral))
                P = np.dot(P_mut, P_dft)
            else:
                raise Exception

            # compute the equilibrium distribution
            v = get_stationary_distribution(P)

            # write the equilibrium distribution to the file
            file_description = '_'.join((rate_adjustment, mut_style))
            filename = file_description + '.txt'
            with open(filename, 'w') as fout:
                print >> fout, '\t'.join(ascii_states + ['probability'])
                for i, (AB, Ab, aB, ab) in enumerate(M):
                    row = (AB, Ab, aB, ab, v[i])
                    print >> fout, '\t'.join(str(x) for x in row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-N', type=int, default=20,
            help='haploid population size')
    parser.add_argument('-mu', type=float, default=0.025,
            help='mutation probability per site per generation')
    main(parser.parse_args())

