"""Check that every place naming a version of ACT or of a package it pins names the same one"""

import re
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ['docker/image.sh']


def read(path):
    return (ROOT / path).read_text()


if __name__ == '__main__':
    project = tomllib.loads(read('pyproject.toml'))['project']
    readme = read('README.md')
    failures = []

    # The compatibility table: its columns, then each release, newest first
    section = readme.split('## Version Compatibility\n')[1].split('\n## ')[0]
    lines = [line for line in section.splitlines() if line.startswith('|')]
    table = [[cell.strip() for cell in line.strip('|').split('|')] for line in lines]
    columns, releases = [cell.lower() for cell in table[0]], table[2:]
    newest = dict(zip(columns, releases[0]))

    # Each newest-row cell links the release it names
    for column, cell in newest.items():
        link = re.fullmatch(r'\[(\S+)\]\((\S+)\)', cell)
        if not link or not link[2].endswith(link[1].rsplit(':', 1)[-1]):
            failures.append(
                f'the {column} cell of the newest row does not link what it names: {cell}'
            )
    labels = {column: cell.split('](')[0].rsplit('v', 1)[-1] for column, cell in newest.items()}

    # No two rows name one ACT release
    if len({release[0] for release in releases}) != len(releases):
        failures.append('two rows of the compatibility table name one ACT release')

    # Every place that names a version, by what it names
    places = {}
    for dependency in project['dependencies']:
        name, pin = dependency.split(' @ ')
        places[name] = {
            'pyproject.toml': pin.rsplit('@', 1)[-1].removeprefix('v'),
            "README.md's table": labels[name],
            'installed': version(name),
        }
    places['act'] = {
        'pyproject.toml': project['version'],
        'installed': version('act-ecosystem'),
        'act/__init__.py': re.search(r"__version__ = '(\S+)'", read('act/__init__.py'))[1],
        "README.md's install line": re.search(r'act\.git@v(\S+)', readme)[1],
        "README.md's table": labels['act'],
    }
    for script in SCRIPTS:
        # The tag each image name carries, ahead of its variant
        tags = set(re.findall(r':v([\d.]+)\$\{', read(script)))
        places['act'][script] = ' and '.join(sorted(tags))
    places["the table's newest release"] = {
        'its ACT cell': labels['act'],
        'its image cell': labels['docker image'],
    }

    # Every place of a name is held to its first
    for name, found in places.items():
        print(f'{name}: ' + ', '.join(f'{place} {value}' for place, value in found.items()))
        first_place, first = next(iter(found.items()))
        failures += [
            f'{name} is {first} in {first_place} and {value} in {place}'
            for place, value in found.items()
            if value != first
        ]

    # The installed packages' requirements, as pip reads them
    sys.stdout.flush()
    if subprocess.run([sys.executable, '-m', 'pip', 'check']).returncode != 0:
        failures.append('pip check fails')

    if failures:
        sys.exit('\n'.join(failures))
