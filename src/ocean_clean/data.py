"""NDBC standard meteorological archive ingestion; retain provenance."""
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd

# NDBC sentinels are column specific: 99 degrees is a valid wind direction.
SENTINELS = {"WDIR": 999, "WSPD": 99, "GST": 99, "WVHT": 99,
             "DPD": 99, "APD": 99, "MWD": 999, "PRES": 9999,
             "ATMP": 999, "WTMP": 999, "DEWP": 999, "VIS": 99,
             "PTDY": 99, "TIDE": 99}


def parse_ndbc(text):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines)
                   if line.lstrip('#').split()[:1] in (["YY"], ["YYYY"])), None)
    if header is None:
        raise ValueError("Missing NDBC YY/YYYY header")
    columns = lines[header].lstrip('#').split()
    body = '\n'.join(line for line in lines[header + 1:]
                     if line.strip() and not line.lstrip().startswith('#'))
    frame = pd.read_csv(io.StringIO(body), sep=r'\s+', names=columns, na_values=['MM'])
    frame = frame.apply(pd.to_numeric, errors='coerce')
    year = frame[columns[0]]
    year = year.where(year >= 100, np.where(year >= 70, year + 1900, year + 2000))
    timestamps = pd.to_datetime(dict(year=year, month=frame['MM'], day=frame['DD'],
                                    hour=frame['hh'], minute=frame.get('mm', 0)),
                                errors='coerce', utc=True)
    values = frame.drop(columns=[c for c in columns if c in {'YY','YYYY','MM','DD','hh','mm'}])
    values = values.rename(columns={'BAR': 'PRES', 'WD': 'WDIR'})
    for col, sentinel in SENTINELS.items():
        if col in values:
            values[col] = values[col].mask(values[col] == sentinel)
    values.index = pd.DatetimeIndex(timestamps, name='time')
    values = values.loc[~values.index.isna()].sort_index()
    if values.index.has_duplicates:
        raise ValueError("Duplicate timestamps: inspect source before aggregation")
    return values


def download(station, year, directory):
    if not re.fullmatch(r'[a-zA-Z0-9]{5}', station):
        raise ValueError('Station must be a five-character NDBC identifier')
    if not 1970 <= year <= 2100:
        raise ValueError('Invalid archive year')
    station = station.lower()
    url = f'https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / f'{station}h{year}.txt.gz'
    if raw.exists():
        payload = raw.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        # Validate before caching an error page or malformed response.
        parse_ndbc(gzip.decompress(payload).decode())
        raw.write_bytes(payload)
    frame = parse_ndbc(gzip.decompress(payload).decode())
    frame.to_csv(directory / f'{station}h{year}.csv')
    metadata = {'source': url, 'sha256': hashlib.sha256(payload).hexdigest(),
                'station': station, 'year': year, 'rows': len(frame),
                'download_or_cache_check_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    raw.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return frame


def load_csv(path):
    frame = pd.read_csv(path, index_col='time', parse_dates=['time'])
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates:
        raise ValueError('Duplicate timestamps')
    return frame.sort_index().apply(pd.to_numeric, errors='raise')


def hourly(frame):
    # Scalar variables only: ordinary averaging is incorrect for directions.
    frame = frame.drop(columns=['WDIR', 'MWD'], errors='ignore')
    return frame.resample('1h').mean()


def quality_report(frame):
    result = {'rows': len(frame), 'start': str(frame.index.min()),
              'end': str(frame.index.max()), 'variables': {}}
    for col in frame:
        missing = frame[col].isna()
        runs = missing.groupby((missing != missing.shift()).cumsum()).sum()
        result['variables'][col] = {'missing_fraction': float(missing.mean()),
                                   'longest_missing_hours': int(runs.max()) if len(runs) else 0,
                                   'observed_count': int((~missing).sum())}
    return result


def parse_ndbc_netcdf(path):
    """Read NOAA THREDDS classic NetCDF; honor per-variable fill metadata."""
    from scipy.io import netcdf_file
    with netcdf_file(path, mmap=False) as dataset:
        clock = dataset.variables['time']
        if clock.units != b'seconds since 1970-01-01 00:00:00 UTC':
            raise ValueError(f'Unsupported time units: {clock.units!r}')
        index = pd.to_datetime(clock[:].copy(), unit='s', utc=True)
        columns = {}
        for name, variable in dataset.variables.items():
            if name in {'time', 'latitude', 'longitude'}:
                continue
            if variable.shape != (len(index), 1, 1):
                raise ValueError(f'Unexpected station dimensions for {name}')
            values = variable[:].copy().reshape(-1).astype(float)
            fill = getattr(variable, '_FillValue', None)
            if fill is not None:
                values[values == fill] = np.nan
            key = variable.short_name.decode().upper()
            columns[key] = values
        frame = pd.DataFrame(columns, index=pd.DatetimeIndex(index, name='time'))
    if frame.index.has_duplicates:
        raise ValueError('Duplicate NetCDF timestamps')
    return frame.sort_index()


def download_thredds(station, year, directory):
    """Official alternative archive; does not substitute synthetic data."""
    if not re.fullmatch(r'[a-zA-Z0-9]{5}', station) or not 1970 <= year <= 2100:
        raise ValueError('Invalid station/year')
    station = station.lower()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / f'{station}h{year}.nc'
    url = f'https://dods.ndbc.noaa.gov/thredds/fileServer/data/stdmet/{station}/{station}h{year}.nc'
    if not raw.exists():
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        temporary = raw.with_suffix('.nc.part')
        temporary.write_bytes(payload)
        try:
            parse_ndbc_netcdf(temporary)
            temporary.replace(raw)
        finally:
            temporary.unlink(missing_ok=True)
    frame = parse_ndbc_netcdf(raw)
    frame.to_csv(directory / f'{station}h{year}.csv')
    metadata = {'source': url, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                'station': station, 'year': year, 'rows': len(frame), 'format': 'netcdf',
                'download_or_cache_check_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    raw.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return frame
