#!/usr/bin/env python

import argparse
import cProfile
import os
import pickle
import pstats
import quantities as pq
import spykeutils.signal_processing as sigproc
import spykeutils.spike_train_generation as stg
import spykeutils.spike_train_metrics as stm
import sys
import tempfile


tau = 5.0 * pq.ms


def trains_as_multiunits(trains):
    half = len(trains) // 2
    return {0: trains[:half], 1: trains[half:2 * half]}


metrics = {
    'es': ('event synchronization',
           lambda trains: stm.event_synchronization(trains, tau, sort=False)),
    'hm': ('Hunter-Milton similarity measure',
           lambda trains: stm.hunter_milton_similarity(trains, tau)),
    'ss': ('Schreiber et al. similarity measure',
           lambda trains: stm.schreiber_similarity(
               trains, sigproc.GaussianKernel(tau), sort=False)),
    'vr': ('van Rossum distance',
           lambda trains: stm.van_rossum_dist(trains, tau, sort=False)),
    'vp': ('Victor-Purpura\'s distance',
           lambda trains: stm.victor_purpura_dist(trains, 2.0 / tau)),
    'vr_mu': ('van Rossum multi-unit distance',
              lambda trains: stm.van_rossum_multiunit_dist(
                  trains_as_multiunits(trains), 0.5, tau)),
    'vp_mu': ('Victor-Purpura\'s multi-unit distance',
              lambda trains: stm.victor_purpura_multiunit_dist(
                  trains_as_multiunits(trains), 0.3, 2.0 / tau))}


def print_available_metrics():
    for key in metrics:
        print "%s  (%s)" % (key, metrics[key][0])


def print_summary(profile_file):
    stats = pstats.Stats(profile_file)
    stats.strip_dirs().sort_stats('cumulative').print_stats(
        r'^spike_train_metrics.py:\d+\([^_<].*(?<!compute)\)')


def profile_metrics(trains, to_profile):
    for key in to_profile:
        metrics[key][1](trains)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Profile the spike train distances.")
    parser.add_argument(
        '--trains', '-t', type=str, nargs=1,
        help="Use given file to load spike trains. If not existent, spike " +
        "trains will be generated as usual and saved in this file.")
    parser.add_argument(
        '--distances', '-d', type=str, nargs='*', default=metrics.iterkeys(),
        choices=metrics.keys(),
        help="Spike train distances to profile. Defaults to all available " +
        "distances.")
    parser.add_argument(
        '--list-distances', '-l', action='store_const', default=False,
        const=True,
        help="Print a list of spike train distance which can be used with " +
        "the -d option.")
    parser.add_argument(
        '--output', '-o', type=str, nargs=1,
        help="Output file for the profiling information.")
    args = parser.parse_args()

    if args.list_distances:
        print_available_metrics()
        sys.exit(0)

    try:
        with open(args.trains[0], 'r') as f:
            trains = pickle.load(f)
        print "Loaded stored trains."
    except:
        trains = [stg.gen_homogeneous_poisson(50.0 * pq.Hz, t_stop=4.0 * pq.s)
                  for i in xrange(6)]
        if args.trains is not None:
            with open(args.trains[0], 'w') as f:
                pickle.dump(trains, f)
            print "Stored trains."

    if args.output is None:
        dummy, out_file = tempfile.mkstemp()
    else:
        out_file = args.output[0]

    cProfile.run('profile_metrics(trains, args.distances)', out_file)
    print_summary(out_file)

    if args.output is None:
        os.unlink(out_file)
