import argparse
import os
import shutil
from importlib import resources as ir

from massivora.config import load_project_and_system_config, write_text_if_missing
from massivora.db import create_job_db
from massivora.job_loader import LocalJobLoader, SlurmJobLoader


def usage():
    return (
        "massivora - pipeline manager\n\n"
        "examples:\n"
        "  massivora new <project_config.yml> [--overwrite]\n"
        "  massivora run <project_config.yml> [--system-config <system.yml>]\n"
        "  massivora sysconf <system_config.yml>\n\n"
    )


def _build_parser():
    p = argparse.ArgumentParser(
        prog='massivora',
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=usage(),
    )
    sub = p.add_subparsers(dest='command', required=False)

    new_p = sub.add_parser('new', help='Generate new project config (only generate without run)')
    new_p.add_argument('project_config', type=str, help='Path to project config YAML to create')
    new_p.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

    run_p = sub.add_parser('run', help='Run with a project config')
    run_p.add_argument('project_config', type=str, help='Path to project config YAML')
    run_p.add_argument(
        '--system-config',
        type=str,
        default=None,
        help='Optional path to system config YAML (merged before project config)',
    )

    sys_p = sub.add_parser('sysconf', help='Replace default system config in the installed massivora package')
    sys_p.add_argument('system_config', type=str, help='Path to system config YAML to copy into the installed package')

    return p


def cmd_new(config_path, overwrite):
    print("Creating new project config...")
    write_text_if_missing(config_path, overwrite=overwrite)
    return 0


def cmd_run(project_config, system_config):
    cfg = load_project_and_system_config(project_config, system_config)

    project_path = cfg.get('project').get('project_path')
    proteins_list = cfg.get('project').get('proteins_list')
    os.makedirs(project_path, mode=0o755, exist_ok=True)

    paths = cfg.get('paths')
    alignment_dir = os.path.join(project_path, paths.get('monomers'))
    output_dir = os.path.join(project_path, paths.get('couplings'))
    job_db = os.path.join(project_path, paths.get('job_db'))
    os.makedirs(alignment_dir, mode=0o755, exist_ok=True)
    os.makedirs(output_dir, mode=0o755, exist_ok=True)

    if not all([project_path, proteins_list, alignment_dir, output_dir, job_db]):
        raise SystemExit('Missing required config keys under project:/paths:.')

    create_job_db(job_db)

    with open(proteins_list, 'r', encoding='utf-8') as f:
        protein_list = [line.strip() for line in f.read().splitlines() if line.strip()]

    manager = cfg.get('manager')
    run_mode = manager.get('type', 'local')
    if run_mode == 'local':
        loader = LocalJobLoader(project_cfg=project_config)
    elif run_mode == 'slurm':
        loader = SlurmJobLoader(project_cfg=project_config)
    else:
        raise SystemExit('Invalid config: manager.type')

    pipeline = cfg.get('pipeline')
    do_download = bool(pipeline.get('download', False))
    do_align = bool(pipeline.get('align', False))
    do_concatenate = bool(pipeline.get('concatenate', False))
    do_coupling = bool(pipeline.get('coupling', False))

    if do_download:
        system = cfg.get('system')
        download_processes = int(system.get('download_processes', 4))
        loader.run_download(protein_list, processes=download_processes)

    if do_align:
        loader.run_align(protein_list)

    if do_concatenate:
        loader.run_concatenate()

    if do_coupling:
        loader.run_couple()

    return 0


def cmd_sysconf(system_config):
    try:
        pkg_root = ir.files('massivora')
    except Exception as e:
        raise SystemExit('Unable to locate installed massivora package directory') from e

    src = system_config
    dst = pkg_root.joinpath('config_template', 'system.yml')

    dst_path = os.fspath(dst)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    try:
        shutil.copyfile(src, dst_path)
    except PermissionError as e:
        raise SystemExit(f'Permission denied writing to installed package path: {dst_path}') from e

    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, 'command', None):
        parser.print_help()
        return 2

    if args.command == 'new':
        return cmd_new(args.project_config, args.overwrite)
    if args.command == 'run':
        return cmd_run(args.project_config, args.system_config)
    if args.command == 'sysconf':
        return cmd_sysconf(args.system_config)


if __name__ == '__main__':
    raise SystemExit(main())
