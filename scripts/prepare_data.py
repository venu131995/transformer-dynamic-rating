"""Turn the raw Swissgrid and MeteoSwiss files into one tidy 15-minute UTC table.

Inputs (downloaded by scripts/fetch_data.sh into data/):
  EnergieUebersichtCH-<year>.xlsx      Swissgrid, 15-min energy per interval, local Swiss time,
                                       timestamp marks the END of the interval
  ogd-smn_bus_h_historical_2020-2029.csv  MeteoSwiss station Buchs/Aarau (BUS, ~15 km from Baden),
                                       hourly mean air temperature tre200h0, UTC

Output: data/load_ambient_15min.csv.gz with columns
  time_utc (interval start), load_mw (Swiss end-user consumption, mean power), ambient_c
"""
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
YEARS = [2023, 2024, 2025]


def read_swissgrid(year):
    wb = openpyxl.load_workbook(DATA / f'EnergieUebersichtCH-{year}.xlsx', read_only=True)
    ws = wb['Zeitreihen0h15']
    ts, kwh = [], []
    for row in ws.iter_rows(min_row=3, max_col=2, values_only=True):
        if row[0] is None:
            break
        ts.append(row[0])
        kwh.append(row[1])
    end_local = pd.to_datetime(pd.Series(ts), format='%d.%m.%Y %H:%M')
    # local -> UTC; the repeated hour in October is resolved from the order of the rows
    end_utc = end_local.dt.tz_localize('Europe/Zurich', ambiguous='infer').dt.tz_convert('UTC')
    start_utc = end_utc - pd.Timedelta(minutes=15)
    mw = np.asarray(kwh, float) * 4 / 1000.0  # kWh per 15 min -> mean MW
    return pd.DataFrame({'time_utc': start_utc, 'load_mw': mw})


def read_meteoswiss():
    df = pd.read_csv(DATA / 'ogd-smn_bus_h_historical_2020-2029.csv', sep=';',
                     usecols=['reference_timestamp', 'tre200h0'])
    t = pd.to_datetime(df['reference_timestamp'], format='%d.%m.%Y %H:%M').dt.tz_localize('UTC')
    # hourly mean labelled by the end of the hour -> place it at the hour's centre
    s = pd.Series(df['tre200h0'].astype(float).values, index=t - pd.Timedelta(minutes=30))
    return s


def main():
    load = pd.concat([read_swissgrid(y) for y in YEARS], ignore_index=True)
    load = load.drop_duplicates('time_utc').set_index('time_utc').sort_index()
    grid = pd.date_range(f'{YEARS[0]}-01-01', f'{YEARS[-1] + 1}-01-01', freq='15min', tz='UTC',
                         inclusive='left')
    load = load.reindex(grid)
    n_gap = int(load['load_mw'].isna().sum())
    load['load_mw'] = load['load_mw'].interpolate(limit_direction='both')

    amb = read_meteoswiss()
    amb = amb[(amb.index >= grid[0] - pd.Timedelta(hours=2)) & (amb.index <= grid[-1] + pd.Timedelta(hours=2))]
    n_amb_gap = int(amb.isna().sum())
    amb = amb.interpolate(limit_direction='both')
    # 15-min interval centres, linear interpolation in time
    centres = grid + pd.Timedelta(minutes=7.5)
    amb_15 = np.interp(centres.asi8, amb.index.asi8, amb.values)

    out = pd.DataFrame({'time_utc': grid, 'load_mw': load['load_mw'].values, 'ambient_c': amb_15})
    out.to_csv(DATA / 'load_ambient_15min.csv.gz', index=False, float_format='%.4f')
    print(f'{len(out)} rows, {out.time_utc.iloc[0]} .. {out.time_utc.iloc[-1]}')
    print(f'load gaps filled: {n_gap}; ambient gaps filled: {n_amb_gap}')
    print(out.describe().round(2).to_string())


if __name__ == '__main__':
    main()
