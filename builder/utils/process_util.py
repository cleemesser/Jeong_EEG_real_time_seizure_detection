# coding: utf-8

from multiprocessing import Pool
from tqdm import tqdm


def run_multi_process(f, l: list, n_processes=40):
    n_processes = min(n_processes, len(l))
    print(n_processes)

    pool = Pool(processes=n_processes)
    results = list(tqdm(pool.imap_unordered(f, l), total=len(l), ncols=75))
    pool.close()
    pool.join()

    return results
