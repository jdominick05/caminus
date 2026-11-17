"""report.py: a stock report from a CSV file.

usage: python3 tools/report.py [--format text|json] [--out PATH] CSV
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import inventory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='a stock report from a CSV file')
    ap.add_argument('csv')
    ap.add_argument('--format', choices=('text', 'json'), default='text')
    ap.add_argument('--out', help='write the report here instead of printing it')
    a = ap.parse_args()
    items = inventory.load(a.csv)
    rep = dict(items=len(items), total=inventory.total_value(items), low=inventory.low_stock(items, 10))
    text = json.dumps(rep, indent=2) if a.format == 'json' else '\n'.join(f'{k}: {v}' for k, v in rep.items())
    if a.out:
        with open(a.out, 'w') as f:
            f.write(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
