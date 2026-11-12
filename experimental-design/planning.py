"""Deterministic study planning arithmetic; does not run research models.

Pass pilot counts only after freezing the protocol. Without counts, prints
explicitly hypothetical scenarios, not experimental findings.
"""
import argparse
import json
import math
from statistics import NormalDist


def binomial_cdf(k, n, probability):
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 1.0 if k >= n else 0.0
    return min(1.0, math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
        + j*math.log(probability)+(n-j)*math.log1p(-probability)) for j in range(k+1)))


def cdf_root(k, n, target):
    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = (lower+upper)/2
        if binomial_cdf(k, n, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower+upper)/2


def allocation(pilot_total, common_success, discordant, half_width=0.03):
    if not 0 <= discordant <= common_success <= pilot_total or pilot_total < 1:
        raise ValueError('Counts must satisfy 0 <= discordant <= success <= total.')
    if not 0 < half_width < 1:
        raise ValueError('Half-width must lie between zero and one.')
    delta = 0.05/6
    lower = 0.0 if common_success == 0 else cdf_root(common_success-1, pilot_total, 1-delta)
    if common_success < 25 or lower == 0:
        return {'status': 'STOP_FEASIBILITY', 'success_lower_bound': lower}
    upper = 1.0 if discordant == common_success else cdf_root(discordant, common_success, delta)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * 3))
    target_success = math.ceil(z * z * upper / half_width**2)
    required = math.ceil(target_success / lower)
    return {'status': 'PRECISION_LIMITED_AT_CAP' if required > 3000 else 'PLANNED',
            'pilot_total': pilot_total, 'common_success': common_success,
            'discordant': discordant, 'success_lower_bound': lower,
            'discordance_upper_bound': upper, 'z': z,
            'target_half_width': half_width,
            'required_common_success': target_success,
            'required_scheduled_ids': required,
            'scheduled_ids': max(1000, min(3000, required)),
            'condition_datasets': 4 * max(1000, min(3000, required))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-total', type=int, default=50)
    p.add_argument('--common-success', type=int)
    p.add_argument('--discordant', type=int)
    a = p.parse_args()
    if a.common_success is None and a.discordant is None:
        print('HYPOTHETICAL PLANNING EXAMPLES; NO EXPERIMENTS HAVE RUN')
        for success, discord in [(50, 0), (50, 5), (50, 15), (40, 5), (20, 0)]:
            print(json.dumps(allocation(50, success, discord)))
        print(json.dumps({'main_condition_datasets_minimum':3*4*1000,
                          'main_condition_datasets_all_at_cap':3*4*3000,
                          'pilot_condition_datasets':3*4*50,
                          'predictive_subset_forward_simulations':3*50*2*50,
                          'initial_reference_fits':3*12*4,
                          'posterior_float32_bytes_at_minimum':4*2000*1000*(4+12+9)*4,
                          'posterior_float32_bytes_at_cap':4*2000*3000*(4+12+9)*4}))
    elif a.common_success is None or a.discordant is None:
        p.error('Provide both --common-success and --discordant.')
    else:
        print(json.dumps(allocation(a.pilot_total, a.common_success, a.discordant), indent=2))
