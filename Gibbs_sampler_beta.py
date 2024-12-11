import numpy as np
from bposd.hgp import hgp
from ldpc import bposd_decoder
from numba import njit


@njit(fastmath=True)
def greedy_decode_beta(input_state, input_syndrome, check_inds, num_sweeps,beta):
    state = input_state.copy()
    syndrome = input_syndrome.copy()
    sites = np.arange(len(state))
    for t in range(num_sweeps):
        np.random.shuffle(sites)
        for site in sites:
            red_syndrome = syndrome[check_inds[site]]
            dE = 2*len(red_syndrome) - 4*np.sum(red_syndrome)
            if dE <= 0:      # Always accept if energy is lowered
                state[site] = 1 - state[site]
                syndrome[check_inds[site]] = 1 - syndrome[check_inds[site]]
            elif np.random.rand() <  np.exp(-beta*dE):
                state[site] = 1 - state[site]
                syndrome[check_inds[site]] = 1 - syndrome[check_inds[site]]
    return state, syndrome



def Gibbs_sampler_beta(H,N,lz,t,iters,beta,depth): # only flip the parant qubit
    
    bpd=bposd_decoder(
    H,#the parity check matrix
    error_rate=0.1,
    channel_probs=[None], #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=N, #the maximum number of iterations for BP)
    bp_method="ms",
    ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=depth #the osd search depth
    )
    
    LERs = 0
            
            
    check_inds = [np.nonzero(site)[0] for site in H.T]
            
    
    for shots in range(0,iters):
        state=np.zeros(N).astype(int)
        syndrome = H @ state % 2


        state, syndrome = greedy_decode_beta(state, syndrome, check_inds, t,beta)
             
            
        syndrome=H@state %2
        bpd.decode(syndrome)
        residual_error=(bpd.osdw_decoding+state) %2
        if sum(lz@residual_error%2)>0:
            LERs += 1
        
    LERs = LERs /iters
    
    return LERs

