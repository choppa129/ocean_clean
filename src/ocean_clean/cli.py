import argparse
import json
import hashlib
from importlib.metadata import version
from pathlib import Path
from .data import download, load_csv, hourly, quality_report
from .benchmark import evaluate


def main():
    parser = argparse.ArgumentParser(description='OceanClean offline imputation toolkit')
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download')
    fetch.add_argument('--station', required=True)
    fetch.add_argument('--year', type=int, required=True)
    fetch.add_argument('--out', default='data/raw')
    for name in ['quality', 'benchmark']:
        p = commands.add_parser(name)
        p.add_argument('--csv', required=True)
        p.add_argument('--out', default=f'outputs/{name}')
        if name == 'benchmark':
            p.add_argument('--target', default='WTMP')
            p.add_argument('--blocks', type=int, default=20)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--scenario', choices=['sensor','station'], default='sensor')
    args = parser.parse_args()
    if args.command == 'download':
        frame = download(args.station, args.year, args.out)
        print(f'Downloaded/loaded {len(frame)} observations')
        return
    frame = hourly(load_csv(args.csv))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == 'quality':
        report = quality_report(frame)
        (out / 'quality.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    else:
        metrics, predictions = evaluate(frame, args.target, blocks=args.blocks,
                                        seed=args.seed, scenario=args.scenario)
        metrics.to_csv(out / 'metrics.csv', index=False)
        predictions.to_csv(out / 'predictions.csv', index=False)
        metadata = dict(vars(args))
        metadata['input_sha256'] = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
        metadata['versions'] = {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn']}
        metadata['split'] = 'chronological 60/20/20; hourly scalar means'
        (out / 'run.json').write_text(json.dumps(metadata, indent=2))
        print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()
