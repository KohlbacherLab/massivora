import argparse
import importlib.util
import os
import runpy
import sys
from massivora.executors import align, concatenate, couple
from massivora.config import load_project_and_system_config
from massivora.logging_utils import setup_logging


def _massivora_pkg_dir():
    spec = importlib.util.find_spec('massivora')
    if spec and spec.submodule_search_locations:
        return os.fspath(list(spec.submodule_search_locations)[0])


def _executor_path(module):
    pkg_dir = _massivora_pkg_dir()
    return os.path.join(pkg_dir, 'executors', module)


def _run_executor(module_path, argv):
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(module_path, run_name='__main__')
    finally:
        sys.argv = old_argv


def main(argv=None):
    parser = argparse.ArgumentParser(prog='massiworker')
    sub = parser.add_subparsers(dest='command', required=True)

    p_align = sub.add_parser('align', help='Run align executor')
    p_align.add_argument('--config', required=True, type=str, help='Project config YAML')
    p_align.add_argument('payload', nargs='?', help='Optional payload file path (e.g. proteins list)')
    p_align.add_argument('args', nargs=argparse.REMAINDER, help='Extra args forwarded to executor')

    p_concat = sub.add_parser('concat', help='Run concatenate executor')
    p_concat.add_argument('--config', required=True, type=str, help='Project config YAML')
    p_concat.add_argument('payload', nargs='?', help='Optional payload file path')
    p_concat.add_argument('args', nargs=argparse.REMAINDER, help='Extra args forwarded to executor')

    p_coupling = sub.add_parser('couple', help='Run coupling executor')
    p_coupling.add_argument('--config', required=True, type=str, help='Project config YAML')
    p_coupling.add_argument('payload', nargs='?', help='Optional payload file path')
    p_coupling.add_argument('args', nargs=argparse.REMAINDER, help='Extra args forwarded to executor')

    ns = parser.parse_args(argv)

    cfg = load_project_and_system_config(ns.config)
    setup_logging(cfg)

    forwarded = ['--config', ns.config]
    if ns.payload:
        forwarded.append(ns.payload)
    if ns.args:
        forwarded.extend(ns.args)

    if ns.command == 'align':
        _run_executor(_executor_path('align.py'), ['align.py'] + forwarded)
        return 0

    if ns.command == 'concat':
        _run_executor(_executor_path('concatenate.py'), ['concatenate.py'] + forwarded)
        return 0

    if ns.command == 'couple':
        _run_executor(_executor_path('couple.py'), ['couple.py'] + forwarded)
        return 0

    raise SystemExit(f"Unknown command: {ns.command}")


if __name__ == '__main__':
    raise SystemExit(main())
