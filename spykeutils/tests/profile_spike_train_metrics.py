#!/usr/bin/env python

import argparse
import cProfile
import quantities as pq
import spykeutils.signal_processing as sigproc
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm


def calc_analytic_metrics(trains):
    tau = 5.0 * pq.ms
    stm.event_synchronization(trains, tau, sort=False)
    stm.hunter_milton_similarity(trains, tau)
    stm.schreiber_similarity(trains, sigproc.GaussianKernel(tau), sort=False)
    stm.van_rossum_dist(trains, tau, sort=False)
    stm.victor_purpura_dist(trains, 2.0 / tau)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Profile the analytic spike train distances.")
    parser.add_argument(
        'output', type=str, nargs=1,
        help="Output file for the profiling information.")
    args = parser.parse_args()

    trains = [stg.gen_homogeneous_poisson(50.0 * pq.Hz, t_stop=10.0 * pq.s)
              for i in xrange(50)]
    cProfile.run('calc_analytic_metrics(trains)', args.output[0])
