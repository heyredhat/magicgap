from magic_gap import *
import pickle
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('extremal_qubit_magic.log')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

qubit_results = {}
n_range = np.arange(2, 6)
R, S, M = 20, 10, 1200
logger.info("\nqubit/qudit | R: %d, S: %d, M: %d | n_range: %s" % (R, S, M, n_range))
for n in n_range:
    d_big = 2**n
    D_big = wh_operators(*[2]*n)
    big_avg_magic = avg_magic(D_big, exact=d_big <= 5, M=M, R=R)
    for d_small in np.arange(2, d_big):
        D_small = qudit_wh_operators(d_small)
        small_avg_magic = avg_magic(D_small, exact=d_big <= 5, M=M, R=R)
        
        B_min, result_min = extremize_subspace_magic(D_big, d_small, "min", R=R, S=S, M=M, exact=d_big <= 5)
        B_max, result_max = extremize_subspace_magic(D_big, d_small, "max", R=R, S=S, M=M, exact=d_big <= 5)

        min_avg_magic = avg_magic_subspace(D_big, B_min, exact=d_big <= 5, M=M, R=R)
        max_avg_magic = avg_magic_subspace(D_big, B_max, exact=d_big <= 5, M=M, R=R)

        qubit_results[(d_big,d_small)] = {"big_avg_magic": big_avg_magic,
                                          "small_avg_magic": small_avg_magic,\
                                          "min_avg_magic":  min_avg_magic,\
                                          "max_avg_magic":  max_avg_magic,\
                                          "B_min": B_min, "result_min": result_min,\
                                          "B_max": B_min, "result_max": result_min,\
                                          "D_big": D_big, "D_small": D_small, "R": R, "S": S, "M": M}
        logger.info("\nd_big: %d, d_small: %d\n\tmax_avg_magic: %.5f +/- %.5f\n\tbig_avg_magic: %.5f +/- %.5f\n\tmin_avg_magic: %.5f +/- %.5f\n\tsmall_avg_magic: %.5f +/- %.5f"\
                    % (d_big, d_small, max_avg_magic[0], max_avg_magic[1], big_avg_magic[0], big_avg_magic[1], min_avg_magic[0], min_avg_magic[1], small_avg_magic[0], small_avg_magic[1]))
        file_handler.flush()
        with open("min_qubit_qudit_magic.pkl", "wb") as f:  
            pickle.dump(qubit_results, f)
logger.info("Done!")