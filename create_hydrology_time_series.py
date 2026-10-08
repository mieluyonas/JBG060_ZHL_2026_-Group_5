from pathlib import Path

import numpy as np
import loading


def run(region=('Ayod', 'Bor South', 'Duk', 'Fangak', 'Twic East'), start=None, end=None, admin_level=2,
        lake=('victoria', 'Kyoga', 'Albert'), root=None,
        lake_max_gap=35, missing_flood_days='nan', output=None):
    if output is None:
        path = Path(__file__).with_name('hydrology_time_series.npz')
    else:
        path = Path(output)

    dates, series = loading.load_region_series(
        region=region, start=start, end=end, admin_level=admin_level,
        lake=lake, root=root, lake_max_gap=lake_max_gap,
        missing_flood_days=missing_flood_days)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dates=dates, **series, region=region,
                        admin_level=admin_level, lake=lake,
                        lake_max_gap=lake_max_gap, missing_flood_days=missing_flood_days)
    return path


if __name__ == '__main__':
    print(run())
