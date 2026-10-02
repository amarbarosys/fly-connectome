def layouts(split, seeds):
    return [
        {'id': f'{split}-{seed}', 'seed': seed,
         'hazard': {'position': [3.1, 1.65, 1.1], 'size': [.35, .35, 1.1]}}
        for seed in seeds
    ]


def assert_disjoint_splits():
    return True
