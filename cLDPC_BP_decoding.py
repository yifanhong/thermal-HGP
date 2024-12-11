import numpy as np
from scipy.sparse import csc_matrix
from ldpc import bp_decoder
from numba import njit

@njit(fastmath=True)
def equilibrate(H, check_inds, T, init_state, init_syndrome, eq_time):
    N = len(H[0])
    state = init_state.copy()
    syndrome = init_syndrome.copy()
    sites = np.arange(N)
    probs = np.exp(-np.arange(50)/T)         # Precompute transition probabilities
    for t in range(eq_time):
        np.random.shuffle(sites)
        for site in sites:
            red_syndrome = syndrome[check_inds[site]]
            dE = 2*(len(red_syndrome) - 2*np.sum(red_syndrome))
            if dE <= 0 or np.random.rand() < probs[dE]:
                state[site] = 1 - state[site]
                syndrome[check_inds[site]] = 1 - syndrome[check_inds[site]]
    return state, syndrome

def get_BP_failures(H, T, bp_iters, init_state, init_syndrome, eq_time, iters):
    check_inds = [np.nonzero(check)[0] for check in H.T]
    bpd = bp_decoder(csc_matrix(H), error_rate=1/(1+np.exp(2/T)), max_iter = bp_iters, bp_method = 'minimum_sum')
    failures = 0
    for i in range(iters):
        # print('Iters: {} / {} ; Res: {} / {}        '.format(i,iters,res[0],res[1]), end='\r')
        state, syndrome = equilibrate(H, check_inds, T, init_state, init_syndrome, eq_time)
        correction = bpd.decode(syndrome)
        final_state = state ^ correction
        if np.sum(final_state) != 0:
            failures += 1
    return failures