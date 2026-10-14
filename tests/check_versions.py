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


def values(pattern, text):
    """Every distinct match, as one value"""
    return ' and '.join(sorted(set(re.findall(pattern, text))))


def page(home, label):
    """Where a release is published: a tag of its repository, or of the image the label names"""
    if home:
        return f'{home}/releases/tag/{label}'
    image, _, tag = label.partition(':')
    return f'https://hub.docker.com/r/{image}/tags?name={tag}'


if __name__ == '__main__':
    project = tomllib.loads(read('pyproject.toml'))['project']
    pins = [dependency.split(' @ ') for dependency in project['dependencies']]
    readme = read('README.md')
    failures = []

    # The compatibility table: its columns, then each release, newest first
    section = readme.split('## Version Compatibility\n')[1].split('\n## ')[0]
    lines = [line for line in section.splitlines() if line.startswith('|')]
    table = [[cell.strip() for cell in line.strip('|').split('|')] for line in lines]
    columns, releases = [cell.lower() for cell in table[0]], table[2:]
    newest = dict(zip(columns, releases[0]))

    # Each column's repository, from the pins and the install line
    homes = {name: pin.removeprefix('git+').rsplit('.git@', 1)[0] for name, pin in pins}
    homes['act'] = re.search(r'git\+(\S+)\.git@', readme)[1]

    # Each newest-row cell links where its release is published
    for column, cell in newest.items():
        link = re.fullmatch(r'\[(\S+)\]\((\S+)\)', cell)
        if not link or link[2] != page(homes.get(column), link[1]):
            failures.append(
                f'the {column} cell of the newest row does not link what it names: {cell}'
            )
    labels = {column: cell.split('](')[0].rsplit('v', 1)[-1] for column, cell in newest.items()}

    # No two rows name one ACT release
    if len({release[0] for release in releases}) != len(releases):
        failures.append('two rows of the compatibility table name one ACT release')

    # The newest release's sentence below the table, and column titles
    opening = f'ACT v{labels["act"]} '
    sentence = next((line for line in section.splitlines() if line.startswith(opening)), '')
    if not sentence:
        failures.append(f'no line below the compatibility table opens with {opening.strip()}')
    titles = dict(zip(columns, table[0]))

    # Every place that names a version, by what it names
    places = {}
    for name, pin in pins:
        places[name] = {
            'pyproject.toml': pin.rsplit('@', 1)[-1].removeprefix('v'),
            "README.md's table": labels[name],
            'installed': version(name),
        }
        # The sentence, where it names this package
        named = re.search(rf'\b{re.escape(titles[name])} v([\d.]*\d)', sentence)
        if named:
            places[name]["README.md's sentence"] = named[1]
    places['act'] = {
        'pyproject.toml': project['version'],
        'installed': version('act-ecosystem'),
        'act/__init__.py': re.search(r"__version__ = '(\S+)'", read('act/__init__.py'))[1],
        "README.md's install line": values(r'act\.git@v(\S+)', readme),
        "README.md's table": labels['act'],
    }
    for script in SCRIPTS:
        # The tag of every image name, wherever it stands
        places['act'][script] = values(r':v(\d+(?:\.\d+)*)', read(script))
    places["the table's newest release"] = {
        'its ACT cell': labels['act'],
        'its image cell': labels['docker image'],
    }
    # The Python and JAX of requirements.yml
    requirements = read('requirements.yml')
    python = re.search(r'python=(\S+)', requirements)[1]
    jax = re.search(r'jax==(\S+)', requirements)[1]
    places['environment'] = {
        'docker/act-env.env': re.search(r'ACT_ENV_BUILD_TAG=(\S+)', read('docker/act-env.env'))[1],
        'requirements.yml': f'py{python}-jax{jax}',
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
    if subprocess.run([sys.executable, '-m', 'pip', 'check'], check=False).returncode != 0:
        failures.append('pip check fails')

    if failures:
        sys.exit('\n'.join(failures))
