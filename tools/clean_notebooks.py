"""Clear saved notebook results, or fail if any would enter a commit.

Usage: python tools/clean_notebooks.py [--check]
"""
import argparse
import json
from pathlib import Path


def clean_notebook(data):
    data.get('metadata', {}).pop('widgets', None)
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        for key in ('execution', 'ExecuteTime', 'trusted', 'scrolled', 'collapsed'):
            cell.get('metadata', {}).pop(key, None)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check only; write nothing.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dirty = []
    for path in sorted(root.glob('*.ipynb')):
        data = json.loads(path.read_text(encoding='utf-8'))
        before = json.dumps(data, sort_keys=True)
        clean_notebook(data)
        if before == json.dumps(data, sort_keys=True):
            continue
        dirty.append(path.name)
        if not args.check:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if args.check and dirty:
        parser.exit(1, 'Clear notebook outputs before committing: ' + ', '.join(dirty) + '\n')
    print(('Would clean: ' if args.check else 'Cleaned: ') + ', '.join(dirty) if dirty else 'All notebooks have empty outputs and execution counts.')


if __name__ == '__main__':
    main()
