import argparse
import subprocess

from .db import STATUS, connect_db


def _slurm_job_states(job_ids):
    """Return dict(job_id -> state-string) for given job ids.

    Uses `sacct` first (works for finished jobs), fallback to `squeue` for running jobs.
    """
    job_ids = [str(x) for x in job_ids if x]
    if not job_ids:
        return {}

    states = {}

    # sacct is best because it includes completed jobs
    try:
        rv = subprocess.run(
            ['sacct', '-P', '-n', '-j', ','.join(job_ids), '-o', 'JobIDRaw,State'],
            capture_output=True,
            text=True,
        )
        if rv.returncode == 0:
            for line in (rv.stdout or '').splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) < 2:
                    continue
                jid, state = parts[0].strip(), parts[1].strip()
                # sacct may return multiple lines per job (steps); keep the first terminal state if possible
                if jid and jid not in states:
                    states[jid] = state
    except FileNotFoundError:
        pass

    # squeue for jobs not returned by sacct (often currently running/pending)
    missing = [jid for jid in job_ids if jid not in states]
    if missing:
        try:
            rv = subprocess.run(
                ['squeue', '-h', '-j', ','.join(missing), '-o', '%i|%T'],
                capture_output=True,
                text=True,
            )
            if rv.returncode == 0:
                for line in (rv.stdout or '').splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    jid, state = line.split('|', 1)
                    states[jid.strip()] = state.strip()
        except FileNotFoundError:
            pass

    return states


def _map_state_to_status(state):
    s = (state or '').upper()

    # Common SLURM states
    if s in {'PENDING', 'CONFIGURING'}:
        return STATUS['PENDING']
    if s in {'RUNNING', 'COMPLETING', 'STAGE_OUT'}:
        return STATUS['RUNNING']

    # Terminal success
    if s in {'COMPLETED'}:
        return STATUS['DONE']

    # Terminal failures
    if s in {'CANCELLED', 'CANCELLED+', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED'}:
        return STATUS['FAILED']

    # Unknown -> keep as running-ish to avoid mistakenly finishing
    return STATUS['RUNNING']


def update_job_db(db_path):
    conn = connect_db(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # collect active jobs (skip done rows)
    align_rows = cursor.execute(
        "SELECT id, pid, status, job_id FROM alignments WHERE job_id IS NOT NULL AND status != ?",
        (STATUS['DONE'],),
    ).fetchall()
    coup_rows = cursor.execute(
        "SELECT id, pid1, pid2, status, job_id FROM couplings WHERE job_id IS NOT NULL AND status != ?",
        (STATUS['DONE'],),
    ).fetchall()

    job_ids = set()
    for r in align_rows:
        job_ids.add(str(r['job_id']))
    for r in coup_rows:
        job_ids.add(str(r['job_id']))

    state_map = _slurm_job_states(sorted(job_ids))

    # update alignments
    for r in align_rows:
        jid = str(r['job_id'])
        new_status = _map_state_to_status(state_map.get(jid))
        # do not regress DONE
        if r['status'] == STATUS['DONE']:
            continue
        cursor.execute(
            "UPDATE alignments SET status = ? WHERE id = ?",
            (int(new_status), int(r['id'])),
        )

    # update couplings
    for r in coup_rows:
        jid = str(r['job_id'])
        new_status = _map_state_to_status(state_map.get(jid))
        if r['status'] == STATUS['DONE']:
            continue
        cursor.execute(
            "UPDATE couplings SET status = ? WHERE id = ?",
            (int(new_status), int(r['id'])),
        )

    conn.commit()
    conn.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog='massiveilance')
    p.add_argument('--db', required=True, help='Path to job_db.sqlite3')
    args = p.parse_args(argv)
    update_job_db(args.db)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
