from magic_gap import *
import pickle
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('extremal_qudit_magic.log')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

qudit_results = {}
d_range = np.arange(3, 33)
R, S, m = 25, 5, 750
logger.info("\nqudit/qudit | R: %d, S: %d | d_range: %s" % (R, S, d_range))
for d_big in d_range:
    M = int(m*d_big/3)
    logger.info("\n\tM: %d" % M)
    D_big = qudit_wh_operators(d_big)
    big_avg_magic = avg_magic(D_big, exact=d_big <= 5, M=M, R=R)
    for d_small in np.arange(2, d_big):
        D_small = qudit_wh_operators(d_small)
        small_avg_magic = avg_magic(D_small, exact=d_big <= 5, M=M, R=R)
        
        B_min, result_min = extremize_subspace_magic(D_big, d_small, "min", R=R, S=S, M=M, exact=d_big <= 5)
        B_max, result_max = extremize_subspace_magic(D_big, d_small, "max", R=R, S=S, M=M, exact=d_big <= 5)

        min_avg_magic = avg_magic_subspace(D_big, B_min, exact=d_big <= 5, M=M, R=R)
        max_avg_magic = avg_magic_subspace(D_big, B_max, exact=d_big <= 5, M=M, R=R)

        qudit_results[(d_big,d_small)] = {"big_avg_magic": big_avg_magic,\
                                          "small_avg_magic": small_avg_magic,\
                                          "min_avg_magic":  min_avg_magic,\
                                          "max_avg_magic":  max_avg_magic,\
                                          "B_min": B_min,\
                                          "B_max": B_min,\
                                          "R": R, "S": S, "M": M}
        logger.info("\nd_big: %d, d_small: %d\n\tmax_avg_magic: %.5f +/- %.5f\n\tbig_avg_magic: %.5f +/- %.5f\n\tmin_avg_magic: %.5f +/- %.5f\n\tsmall_avg_magic: %.5f +/- %.5f"\
                    % (d_big, d_small, max_avg_magic[0], max_avg_magic[1], big_avg_magic[0], big_avg_magic[1], min_avg_magic[0], min_avg_magic[1], small_avg_magic[0], small_avg_magic[1]))
        file_handler.flush()
        with open("min_qudit_qudit_magic.pkl", "wb") as f:  
            pickle.dump(qudit_results, f)
logger.info("Done!")