import argparse
import os
import shlex
import shutil
import subprocess
import sys
from itertools import combinations

from massivora.config import (ensure_system_config,
                              load_project_and_system_config,
                              write_project_config)
from massivora.db import STATUS, create_job_db, connect_db, get_table_names, quote_identifier
from massivora.job_loader import LocalJobLoader, SlurmJobLoader
from massivora.massiveilance import ensure_cron_entry


def usage():
    return (
        "massivora - task manager\n\n"
        "examples:\n"
        "  massivora new [project_name] [--overwrite]\n"
        "  massivora run align <project_config.yml> [--system-config <system.yml>]\n"
        "  massivora run concat <project_config.yml> [--system-config <system.yml>]\n"
        "  massivora run couple <project_config.yml> [--system-config <system.yml>]\n"
        "  massivora batch align <project_config.yml> [--system-config <system.yml>]\n"
        "  massivora batch concat <project_config.yml> [--system-config <system.yml>]\n"
        "  massivora batch couple <project_config.yml> [--system-config <system.yml>]\n"
        "  massivora sysconf\n\n"
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
    new_p.add_argument('project_name', nargs='?', default='my_project', type=str, help='Default project name')
    new_p.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

    run_p = sub.add_parser('run', help='Run a pipeline stage locally')
    _add_stage_subparsers(run_p)

    batch_p = sub.add_parser('batch', help='Submit a pipeline stage to a SLURM cluster')
    _add_stage_subparsers(batch_p)

    sub.add_parser('sysconf', help='Open the user system config (~/.massivora.yml) in an editor')

    return p


def _add_stage_subparsers(parent_parser):
    """Attach the shared ``align``/``couple`` stage subparsers to a command."""
    stage_sub = parent_parser.add_subparsers(dest='stage', required=True)
    for stage, helptext in (
        ('align', 'Run download + alignment stage'),
        ('concat', 'Run concatenation stage'),
        ('couple', 'Run coupling stage'),
    ):
        sp = stage_sub.add_parser(stage, help=helptext)
        sp.add_argument('project_config', type=str, help='Path to project config YAML')
        sp.add_argument(
            '--system-config',
            type=str,
            default=None,
            help='Optional path to system config YAML (merged before project config)',
        )


def cmd_new(default_project_name, overwrite):
    print("Creating new project config...")

    default_name = default_project_name

    try:
        project_name = input(f"Project name [{default_name}]: ").strip()
    except EOFError:
        project_name = ''
    if not project_name:
        project_name = default_name

    default_path = os.path.join(os.getcwd(), project_name)

    try:
        project_path = input(f"Project path [{default_path}]: ").strip()
    except EOFError:
        project_path = ''
    if not project_path:
        project_path = default_path

    written_path = write_project_config(
        project_name, os.path.dirname(project_path), overwrite
    )

    os.makedirs(project_path, mode=0o755, exist_ok=True)
    target_path = os.path.join(project_path, os.path.basename(written_path))
    if os.path.abspath(written_path) != os.path.abspath(target_path):
        if os.path.exists(target_path) and not overwrite:
            raise SystemExit(
                f"Config already exists at: {target_path} (use --overwrite to replace)"
            )
        shutil.move(written_path, target_path)
        written_path = target_path

    print(f"Wrote project config to: {written_path}")
    return 0


def cmd_run(project_config, system_config, stage, mode):
    cfg = load_project_and_system_config(project_config, system_config)

    project_path = cfg.get('project').get('project_path')
    proteins_list = cfg.get('project').get('proteins_list')
    os.makedirs(project_path, mode=0o755, exist_ok=True)

    paths = cfg.get('paths')
    alignment_dir = os.path.join(project_path, paths.get('monomers'))
    output_dir = os.path.join(project_path, paths.get('couplings'))
    os.makedirs(alignment_dir, mode=0o755, exist_ok=True)
    os.makedirs(output_dir, mode=0o755, exist_ok=True)

    if not all([project_path, proteins_list, alignment_dir, output_dir]):
        raise SystemExit('Missing required config keys under project:/paths:.')

    # Create DB with dynamic table names
    create_job_db(cfg)

    # Extract table names
    table_names = get_table_names(cfg)
    align_table = quote_identifier(table_names['alignments'])
    couple_table = quote_identifier(table_names['couplings'])

    if mode == 'local':
        loader = LocalJobLoader(project_cfg=project_config)
    elif mode == 'slurm':
        loader = SlurmJobLoader(project_cfg=project_config)
    else:
        raise SystemExit(f'Invalid run mode: {mode}')

    # CLI subcommands for explicit stage runs.
    if stage == 'align':
        # Load proteins into the job DB
        system = cfg.get('system')
        download_processes = int(system.get('download_processes', 4))
        with open(proteins_list, 'r', encoding='utf-8') as f:
            protein_list = [line.strip() for line in f.read().splitlines() if line.strip()]
        conn = connect_db(cfg)
        cursor = conn.cursor()
        try:
            # proteins not downloaded yet are with status NULL
            cursor.executemany(
                f"INSERT OR IGNORE INTO {align_table} (pid) VALUES (?)",
                [(protein,) for protein in protein_list],
            )
            conn.commit()

            # put coupling jobs too (with NOOPT status so they're ready to be claimed)
            # TODO: status need to be null to be unclaimable for `run all` command.
            cursor.executemany(
                f"INSERT OR IGNORE INTO {couple_table} (name, pid1, pid2, status) VALUES (?, ?, ?, ?)",
                [(pid1+'-'+pid2, pid1, pid2, STATUS['NOOPT']) for pid1, pid2 in combinations(protein_list, 2)]
            )
            conn.commit()

            # check if all the proteins are downloaded
            cursor.execute(f"SELECT pid FROM {align_table} WHERE status IS NULL")
            protein_to_download = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        if protein_to_download:
            loader.run_download(protein_to_download, processes=download_processes)

        # Then run alignment
        loader.run_align()
    elif stage == 'concat':
        loader.run_concatenate()
    elif stage == 'couple':
        loader.run_couple()

    # Put massiveilance into crontab. The massivora command name (run/batch) is
    # passed through so the monitor knows whether to resubmit locally or to SLURM.
    massiveilance = os.path.join(os.path.dirname(sys.executable), "massiveilance")
    command = 'batch' if mode == 'slurm' else 'run'
    if ensure_cron_entry(f'0 * * * * {massiveilance} {command} {stage} {os.path.abspath(project_config)}'):
        print(f"massivora: added massiveilance {command} {stage} {project_config} to crontab")

    return 0


def cmd_sysconf():
    # Make sure the file exists (seed it from the bundled template if needed),
    # then open it in the user's preferred editor.
    config_path = ensure_system_config()

    editor = os.environ.get('VISUAL') or os.environ.get('EDITOR')
    if not editor:
        editor = 'notepad' if os.name == 'nt' else 'vi'

    try:
        rv = subprocess.run([*shlex.split(editor), config_path])
    except FileNotFoundError as e:
        raise SystemExit(f"Editor not found: {editor!r}. Set $EDITOR to a valid editor.") from e

    return rv.returncode


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, 'command', None):
        parser.print_help()
        return 2

    if args.command == 'new':
        return cmd_new(args.project_name, args.overwrite)
    if args.command == 'run':
        return cmd_run(args.project_config, args.system_config, args.stage, mode='local')
    if args.command == 'batch':
        return cmd_run(args.project_config, args.system_config, args.stage, mode='slurm')
    if args.command == 'sysconf':
        return cmd_sysconf()


if __name__ == '__main__':
    raise SystemExit(main())
