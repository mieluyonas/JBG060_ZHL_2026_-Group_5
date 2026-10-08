from pathlib import Path
import numpy as np

VARIABLES = ('lake_level_victoria', 'lake_level_kyoga', 'lake_level_albert',
             'evapotranspiration', 'rainfall', 'runoff')


def detrend(values):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return result
    time = np.arange(values.size, dtype=float)
    centre = time[valid].mean()
    design = np.column_stack((time[valid] - centre, np.ones(valid.sum())))
    coefficients = np.linalg.lstsq(design, values[valid], rcond=None)[0]
    result[valid] = values[valid] - design @ coefficients
    return result


def cross_correlations(dates, flood, predictor, min_pairs=30,
                       min_lag=7, max_lag=250):
    lags = np.arange(min_lag, max_lag + 1)
    correlations = np.full(lags.size, np.nan)
    counts = np.zeros(lags.size, dtype=int)
    for index, lag in enumerate(lags):
        if lag >= len(dates):
            continue
        x = predictor[:-lag]
        y = flood[lag:]
        valid = np.isfinite(x) & np.isfinite(y)
        counts[index] = valid.sum()
        if counts[index] < min_pairs:
            continue
        x = x[valid]
        y = y[valid]
        a = x - x.mean()
        b = y - y.mean()
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            continue
        denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
        correlations[index] = np.clip(np.dot(a, b) / denominator, -1, 1)
    return lags, correlations, counts


def variance_inflation_factors(series, min_pairs=30):
    variables = tuple(name for name in series if name != 'flood_ratio')
    data = np.column_stack([series[name] for name in variables])
    data = data[np.isfinite(data).all(axis=1)]

    factors = np.full(len(variables), np.nan)
    if len(data) < max(min_pairs, len(variables) + 1):
        return factors, len(data)
    
    centred = data - data.mean(axis=0)
    scales = data.std(axis=0)
    varying = scales > 1e-12
    centred[:, varying] /= scales[varying]


    for index in np.flatnonzero(varying):
        target = centred[:, index]
        predictors = centred[:, varying & (np.arange(len(variables)) != index)]
        coefficients = np.linalg.lstsq(predictors, target, rcond=None)[0]
        residual = target - predictors @ coefficients
        unexplained = np.dot(residual, residual) / np.dot(target, target)
        if unexplained <= 1e-12:
            factors[index] = np.inf
        else:
            factors[index] = max(1.0, 1 / unexplained)

    
    return factors, len(data)


def run(input_file=None, min_pairs=30, top=1, min_lag=7, max_lag=250):
    if input_file is None:
        path = Path(__file__).with_name('hydrology_time_series.npz')
    else:
        path = Path(input_file)

    series = {}
    with np.load(path, allow_pickle=False) as saved:
        dates = saved['dates'].astype('datetime64[D]')
        variables = tuple(name for name in VARIABLES if name in saved)
        if 'lake_level' in saved:
            variables = ('lake_level',) + variables
        for name in ['flood_ratio'] + list(variables):
            series[name] = saved[name].astype(float)

    residuals = {}
    for name in series:
        residuals[name] = detrend(series[name])

    reports = []
    for label, values in (('Not detrended', series), ('Detrended', residuals)):
        lines = [f'{label}:']

        for name in variables:
                lags, correlations, _ = cross_correlations(
                    dates, values['flood_ratio'], values[name], min_pairs,
                    min_lag=min_lag, max_lag=max_lag)
                valid = np.flatnonzero(np.isfinite(correlations))
                order = np.argsort(-np.abs(correlations[valid]), kind='stable')
                ranked = valid[order][:top]
                if not len(ranked):
                    lines.append(f'{name} - nan - nan')
                for index in ranked:
                    lines.append(f'{name} - {lags[index]} - {correlations[index]:.6f}')
            
        factors, count = variance_inflation_factors(values, min_pairs)
        lines.append(f'VIF ({count} complete days):')

        for name, factor in zip(variables, factors):
            lines.append(f'{name} - {factor:.6f}')

        reports.append('\n'.join(lines))

    return '\n\n'.join(reports)


if __name__ == '__main__':
    print(run(min_lag=7, max_lag=250))
