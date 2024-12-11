import numpy as np
from scipy.sparse import csc_matrix
from ldpc import bposd_decoder
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

def get_BPOSD_failures(code, T, par, init_state, init_syndrome, eq_time, iters):
    H = code.hx
    logicals = code.lx
    check_inds = [np.nonzero(site)[0] for site in H.T]
    bpd = bposd_decoder(csc_matrix(H), error_rate=1/(1+np.exp(2/T)), max_iter = par[0], bp_method = 'msl', osd_method = 'osd_0', osd_order = par[1])
    failures = 0
    for i in range(iters):
        state, syndrome = equilibrate(H, check_inds, T, init_state, init_syndrome, eq_time)
        bpd.decode(syndrome)
        correction = bpd.osdw_decoding
        final_state = state ^ correction
        if (logicals@final_state%2).any():
            failures += 1
    return failures