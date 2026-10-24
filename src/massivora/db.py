import hashlib
import logging
import math
import os
import re
import sqlite3
import sys

STATUS = {
    'NOOPT': -1,
    'DONE': 0,
    'RUNNING': 1,
    'PENDING': 2,
    'RETRYING': 3,
    'FAILED': 4,
    # left out by couple.cascade; never claimed, terminal like DONE/FAILED
    'SKIPPED': 5
}


def get_db_config(cfg):
    """
    Extract database configuration from project config.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict

    Returns
    -------
    `dict`
        Database configuration with keys: backend, filename (for sqlite),
        host, port, user, password, database (for other backends)
    """
    if 'database' not in cfg:
        raise ValueError('No database configuration found in config. Add a "database" section to your project.yml')

    db_cfg = cfg['database'].copy()
    backend = db_cfg.get('backend', 'sqlite').lower()
    db_cfg['backend'] = backend

    # For SQLite, convert relative path to absolute path
    if backend == 'sqlite' and 'filename' in db_cfg:
        db_filename = db_cfg['filename']
        if not os.path.isabs(db_filename):
            project_path = cfg.get('project').get('project_path')
            db_cfg['filename'] = os.path.join(project_path, db_filename)

    return db_cfg


def is_network_filesystem(path):
    """
    Detect if a path is on a network filesystem.

    Parameters
    ----------
    `path` — str
        File or directory path

    Returns
    -------
    `bool`
        True if on network filesystem (NFS, Lustre, GPFS, etc.)
    """
    if not os.path.exists(os.path.dirname(path)):
        return False

    try:
        if sys.platform == 'linux':
            # Check filesystem type on Linux
            stat = os.statvfs(path if os.path.exists(path) else os.path.dirname(path))

            # Read /proc/mounts to get filesystem type
            with open('/proc/mounts', 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        mount_point = parts[1]
                        fs_type = parts[2]

                        # Check if path is under this mount point
                        try:
                            if os.path.commonpath([path, mount_point]) == mount_point:
                                # Network filesystem types
                                if fs_type in ('nfs', 'nfs4', 'cifs', 'smb', 'lustre', 'gpfs', 'glusterfs', 'beegfs', 'orangefs'):
                                    return True
                        except ValueError:
                            continue
        elif sys.platform == 'darwin':
            # macOS - check mount output
            import subprocess
            result = subprocess.run(['df', '-T', path if os.path.exists(path) else os.path.dirname(path)],
                                   capture_output=True, text=True)
            if 'nfs' in result.stdout.lower() or 'smbfs' in result.stdout.lower():
                return True
    except Exception:
        pass

    return False


def get_db_path(cfg):
    """
    Get database path from config (SQLite only).

    Parameters
    ----------
    `cfg` — dict
        Configuration dict

    Returns
    -------
    `str`
        Database path for SQLite, or raises error for other backends
    """
    db_cfg = get_db_config(cfg)
    if db_cfg['backend'] != 'sqlite':
        raise ValueError(f"get_db_path() only works with SQLite backend, not {db_cfg['backend']}")
    return db_cfg['filename']


# Backend-specific connection implementations

def _connect_sqlite(db_path, mode='rw', shared_filesystem=None):
    """
    Connect to SQLite database.

    Parameters
    ----------
    `db_path` — str
        Path to SQLite database file
    `mode` — str
        Connection mode: 'ro', 'rw', 'create' (default: 'rw')
    `shared_filesystem` — bool or None
        If True, uses DELETE journal mode for network filesystem safety.
        If False, uses WAL mode for better performance.
        If None (default), auto-detects filesystem type.

    Returns
    -------
    `connection`
        SQLite database connection
    """
    import logging

    # Auto-detect network filesystem if not specified
    if shared_filesystem is None:
        shared_filesystem = is_network_filesystem(db_path)

    # Choose journal mode based on filesystem type
    journal_mode = 'DELETE' if shared_filesystem else 'WAL'

    if shared_filesystem:
        logging.info(f"SQLite on network filesystem detected, using {journal_mode} journal mode for multi-node safety")
    else:
        logging.debug(f"SQLite on local filesystem, using {journal_mode} journal mode for performance")

    if mode == 'ro':
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
        conn.execute('PRAGMA busy_timeout=60000;')
        conn.execute('PRAGMA foreign_keys=ON;')
    elif mode == 'rw':
        conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True, timeout=60)
        conn.execute('PRAGMA busy_timeout=60000;')
        conn.execute(f'PRAGMA journal_mode={journal_mode};')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute('PRAGMA foreign_keys=ON;')
    else:  # create mode
        conn = sqlite3.connect(db_path, timeout=60)
        conn.execute('PRAGMA busy_timeout=60000;')
        conn.execute(f'PRAGMA journal_mode={journal_mode};')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute('PRAGMA foreign_keys=ON;')

    return conn


def _connect_mysql(db_cfg, mode='rw'):
    """Connect to MySQL/MariaDB database."""
    try:
        import pymysql
    except ImportError:
        raise ImportError("pymysql is required for MySQL/MariaDB connections. Install with: pip install pymysql")

    conn_params = {
        'host': db_cfg.get('host', 'localhost'),
        'port': int(db_cfg.get('port', 3306)),
        'user': db_cfg.get('user'),
        'password': db_cfg.get('password'),
        'database': db_cfg.get('database'),
        'connect_timeout': int(db_cfg.get('connect_timeout', 60)),
        'charset': db_cfg.get('charset', 'utf8mb4'),
        'autocommit': False,
    }

    return pymysql.connect(**conn_params)


def _connect_postgresql(db_cfg, mode='rw'):
    """Connect to PostgreSQL database."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 is required for PostgreSQL connections. Install with: pip install psycopg2-binary")

    conn_params = {
        'host': db_cfg.get('host', 'localhost'),
        'port': int(db_cfg.get('port', 5432)),
        'user': db_cfg.get('user'),
        'password': db_cfg.get('password'),
        'database': db_cfg.get('database'),
        'connect_timeout': int(db_cfg.get('connect_timeout', 60)),
    }

    return psycopg2.connect(**conn_params)


# Public connection API

def connect_db(cfg: dict):
    """
    Connect to database (create/write mode).

    Parameters
    ----------
    `cfg` — dict
        Configuration dict containing database section

    Returns
    -------
    `connection`
        Database connection object
    """
    db_cfg = get_db_config(cfg)
    backend = db_cfg['backend']

    if backend == 'sqlite':
        shared_fs = db_cfg.get('shared_filesystem', None)
        return _connect_sqlite(db_cfg['filename'], mode='create', shared_filesystem=shared_fs)
    elif backend in ('mysql', 'mariadb'):
        return _connect_mysql(db_cfg, mode='create')
    elif backend == 'postgresql':
        return _connect_postgresql(db_cfg, mode='create')
    else:
        raise ValueError(f"Unsupported database backend: {backend}")


def connect_db_ro(cfg: dict):
    """
    Connect to database (read-only mode).

    Parameters
    ----------
    `cfg` — dict
        Configuration dict containing database section

    Returns
    -------
    `connection`
        Database connection object (read-only)
    """
    db_cfg = get_db_config(cfg)
    backend = db_cfg['backend']

    if backend == 'sqlite':
        shared_fs = db_cfg.get('shared_filesystem', None)
        return _connect_sqlite(db_cfg['filename'], mode='ro', shared_filesystem=shared_fs)
    elif backend in ('mysql', 'mariadb'):
        return _connect_mysql(db_cfg, mode='ro')
    elif backend == 'postgresql':
        return _connect_postgresql(db_cfg, mode='ro')
    else:
        raise ValueError(f"Unsupported database backend: {backend}")


def connect_db_rw(cfg: dict):
    """
    Connect to database (read-write mode).

    Parameters
    ----------
    `cfg` — dict
        Configuration dict containing database section

    Returns
    -------
    `connection`
        Database connection object (read-write)
    """
    db_cfg = get_db_config(cfg)
    backend = db_cfg['backend']

    if backend == 'sqlite':
        shared_fs = db_cfg.get('shared_filesystem', None)
        return _connect_sqlite(db_cfg['filename'], mode='rw', shared_filesystem=shared_fs)
    elif backend in ('mysql', 'mariadb'):
        return _connect_mysql(db_cfg, mode='rw')
    elif backend == 'postgresql':
        return _connect_postgresql(db_cfg, mode='rw')
    else:
        raise ValueError(f"Unsupported database backend: {backend}")


def get_table_names(cfg):
    """
    Extract table names from config with backwards-compatible defaults.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict

    Returns
    -------
    `dict`
        Dictionary mapping table types to their names
    """
    align_table = cfg.get('align', {}).get('db_table', 'alignments')
    couple_table = cfg.get('couple', {}).get('db_table', 'couplings')
    return {
        'alignments': align_table,
        'couplings': couple_table
    }


def quote_identifier(identifier):
    """
    Quote SQL identifier for safe dynamic table name usage. SQLite uses double 
    quotes for identifiers. This prevents SQL injection while allowing dynamic 
    table names.

    Parameters
    ----------
    `identifier` — str
        Table or column name

    Returns
    -------
    `str`
        Quoted identifier safe for SQL

    """
    # Replace any existing double quotes with two double quotes (SQL escape)
    # and wrap in double quotes
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


def compute_file_hash(name):
    """
    Compute file hash from file name.

    Parameters
    ----------
    `name` — str
        Filename (basename without directories)

    Returns
    -------
    `bytes`
        First 4 bytes of SHA-256 hash (stored as BLOB in database)
    """
    return hashlib.sha256(name.encode('utf-8')).digest()[:4]


def hash_to_hex(file_hash):
    """
    Convert hash bytes to hex string for directory paths.

    Parameters
    ----------
    `file_hash` — bytes
        4-byte hash

    Returns
    -------
    `str`
        8-character hex string
    """
    if isinstance(file_hash, bytes):
        return file_hash.hex()
    return file_hash  # Already a string


def get_hashed_file_path(name, file_hash=None, base_dir=''):
    """
    Get hash-based file path for a filename.

    Parameters
    ----------
    `name` — str
        Filename (basename without directories)
    `file_hash` — bytes or str (optional)
        Pre-computed hash as bytes (BLOB) or hex string. If None, will compute from name. (default: `None`)
    `base_dir` — str (optional)
        Base output directory (default: `''`)

    Returns
    -------
    `tuple`
        (full_path, file_hash_bytes) where full_path is in the format: `base_dir/ab/cd/name`
        where `ab` and `cd` are the first 4 characters of the hash hex for fanout.
        file_hash_bytes is the 4-byte hash suitable for database storage.
    """
    if file_hash is None:
        file_hash = compute_file_hash(name)

    # Convert to bytes if needed
    if isinstance(file_hash, str):
        file_hash_bytes = bytes.fromhex(file_hash)
    else:
        file_hash_bytes = file_hash

    # Get hex representation for path
    hash_hex = hash_to_hex(file_hash_bytes)

    # 2-level fanout: ab/cd/name
    dir_path = os.path.join(base_dir, hash_hex[0:2], hash_hex[2:4])
    return os.path.join(dir_path, name), file_hash_bytes


def ensure_columns(cfg: dict):
    """
    Add columns to existing tables if they don't exist.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict

    Returns
    -------
    `None`
    """
    tables = get_table_names(cfg)
    conn = connect_db(cfg)
    cur = conn.cursor()

    def add_col(table, col_def):
        try:
            quoted_table = quote_identifier(table)
            cur.execute(f'ALTER TABLE {quoted_table} ADD COLUMN {col_def}')
            conn.commit()
        except (sqlite3.OperationalError, Exception):
            pass

    add_col(tables['couplings'], 'claimed_at REAL')
    add_col(tables['couplings'], 'job_id TEXT')
    add_col(tables['couplings'], 'file_hash BLOB')
    add_col(tables['couplings'], 'score_file_id INTEGER')
    add_col(tables['couplings'], 'score_offset INTEGER')
    add_col(tables['couplings'], 'score_len INTEGER')
    add_col(tables['couplings'], 'pcontact REAL')
    add_col(tables['couplings'], 'neffoverL REAL')
    add_col(tables['couplings'], 'log_lr_raw REAL')
    add_col(tables['couplings'], 'confidence_flag BOOLEAN')
    add_col(tables['couplings'], 'high_precision_hit BOOLEAN')
    add_col(tables['alignments'], 'job_id TEXT')
    add_col(tables['alignments'], 'claimed_at REAL')

    # Create index on file_hash if column exists
    try:
        quoted_table = quote_identifier(tables['couplings'])
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS idx_couplings_file_hash ON {quoted_table}(file_hash)'
        )
        conn.commit()
    except (sqlite3.OperationalError, Exception):
        pass

    conn.close()


def _index_name(table, suffix):
    """
    Build an index name scoped to its table.

    Parameters
    ----------
    `table` — str
        Unquoted table name
    `suffix` — str
        Short description of the indexed columns

    Returns
    -------
    `str`
        Index name that stays unique when a project overrides `db_table`
    """
    safe = re.sub(r'\W+', '_', table).strip('_')
    return f"idx_{safe}_{suffix}"


def create_job_db(cfg: dict):
    """
    Create job database with configurable table names.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict

    Returns
    -------
    `None`
    """
    tables = get_table_names(cfg)
    align_table = quote_identifier(tables['alignments'])
    couple_table = quote_identifier(tables['couplings'])

    conn = connect_db(cfg)
    cursor = conn.cursor()

    cursor.execute(
        f'''
    CREATE TABLE IF NOT EXISTS {align_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pid TEXT NOT NULL UNIQUE,
        length INTEGER,
        align_nseqs INTEGER,
        filtered_nseqs INTEGER,
        status INTEGER,
        job_id TEXT,
        claimed_at REAL
    );
    '''
    )

    cursor.execute(
        f'''
    CREATE TABLE IF NOT EXISTS {couple_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        pid1 TEXT NOT NULL,
        pid2 TEXT NOT NULL,
        length INTEGER,
        number INTEGER,
        effnumber INTEGER,
        pcontact REAL,
        neffoverL REAL,
        log_lr_raw REAL,
        confidence_flag BOOLEAN,
        high_precision_hit BOOLEAN,
        status INTEGER,
        job_id TEXT,
        claimed_at REAL,
        file_hash BLOB,
        score_file_id INTEGER,
        score_offset INTEGER,
        score_len INTEGER,
        UNIQUE (name),
        FOREIGN KEY (pid1) REFERENCES {align_table}(pid),
        FOREIGN KEY (pid2) REFERENCES {align_table}(pid)
    );
    '''
    )

    # Create index for fast hash lookups
    cursor.execute(
        f'CREATE INDEX IF NOT EXISTS idx_couplings_file_hash ON {couple_table}(file_hash)'
    )

    # Ensure all columns exist
    ensure_columns(cfg)

    # Covering indexes for protein interaction-network lookups
    couple_name = tables['couplings']
    for suffix, columns in (
        ('net1', 'pid1, log_lr_raw DESC, pid2, neffoverL'),
        ('net2', 'pid2, log_lr_raw DESC, pid1, neffoverL'),
    ):
        index_name = quote_identifier(_index_name(couple_name, suffix))
        cursor.execute(
            f'CREATE INDEX IF NOT EXISTS {index_name} ON {couple_table}({columns})'
        )

    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        status TEXT
    );
    '''
    )

    conn.commit()
    conn.close()


# ORDER BY clauses `couple.cascade.rank_by` may choose from.
# keep_fraction 0.30 keeps ~97 % of plmDCA's top-5 % calls on the he2021 benchmark
# and H. pylori, at ~4.5x less screen compute; 0.16 keeps only ~90 % (8.3x).
CASCADE_RANKS = {
    'log_lr_raw': 'log_lr_raw DESC',
    'high_precision_hit': 'high_precision_hit DESC, log_lr_raw DESC',
}

# `couple.cascade.keep_denominator`: what `keep_fraction` is a fraction of.
CASCADE_DENOMINATORS = ('ranked', 'ranked_plus_skipped')


def _cascade_keep_rule(cascade: dict):
    """
    Validate the keep rule of a `couple.cascade` block.

    Returns (fraction, count, denominator): exactly one of fraction / count is
    set, or neither (keep every ranked pair: the block only gates or skips).
    """
    has_fraction = cascade.get('keep_fraction') is not None
    has_count = cascade.get('keep_count') is not None
    if has_fraction and has_count:
        raise ValueError("couple.cascade: set keep_fraction or keep_count, not both")
    fraction = count = None
    if has_fraction:
        fraction = float(cascade['keep_fraction'])
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"couple.cascade.keep_fraction must be in (0, 1], got {fraction}")
    if has_count:
        raw = cascade['keep_count']
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw != int(raw) or int(raw) < 1:
            raise ValueError(f"couple.cascade.keep_count must be a positive integer, got {raw!r}")
        count = int(raw)
    denominator = cascade.get('keep_denominator')
    if denominator is None:
        denominator = 'ranked'
    elif has_count:
        raise ValueError("couple.cascade.keep_denominator only applies to keep_fraction, not keep_count")
    elif denominator not in CASCADE_DENOMINATORS:
        raise ValueError(f"couple.cascade.keep_denominator must be one of {CASCADE_DENOMINATORS}, "
                         f"got {denominator!r}")
    return fraction, count, denominator


def apply_cascade(cfg: dict):
    """
    Restrict a coupling run to the best pairs of a finished GaussDCA run.

    With `couple.cascade` set, only the best-ranked DONE pairs of the source
    run stay claimable; every other pair the source scored is marked SKIPPED.
    The keep rule is `keep_fraction` (a share) or `keep_count` (a number of
    pairs), exactly one of the two; with neither, every ranked pair is kept
    and the block only gates (`min_neffoverL`) or skips (`skip_unranked`).

    `keep_fraction` is by default a share of the ranked (DONE) source rows.
    With `keep_denominator: ranked_plus_skipped` it is a share of the ranked
    rows plus the rows the source left SKIPPED, so the number of pairs kept
    does not shrink when the source stage itself gates by depth (the
    GaussDCA pre-gate marks pairs below `min_neffoverL` SKIPPED with no
    score). NOTE: this reproduces the pair *budget* of an ungated source, not
    the gated *set*: the gated set "top 20 % of all scored pairs, then
    Neff / L >= 0.1" is the ungated pairs above one score threshold, and
    under a pre-gate that is `keep_count` (the gated set's size) -- a fraction
    of ranked + skipped keeps 0.2 of all pairs, all of them above the gate,
    i.e. 3x the gated set on the H. pylori screen.

    With `min_neffoverL` set, kept pairs whose source `neffoverL` (Beff / L
    of the concatenated alignment) is below it are marked SKIPPED as well:
    below Beff / L ~ 0.1 the detector fires on 6-7 % of labelled negatives and
    3 % of labelled positives (its design FPR is 0.1 %), so a call there is
    an artefact of the shallow alignment. A block that carries only
    `min_neffoverL` (no keep rule) keeps everything above the gate. Without a
    source run (no `source_db` / `source_table`: the block belongs to the
    stage that applies the gate while scoring, i.e. the GaussDCA pre-gate)
    there is nothing to rank and the call is a no-op.

    A source row with NULL `neffoverL`, or a source table without the column,
    is never gated; both are logged at WARNING with counts, because a gate
    that gates nothing costs 2-3x the GPU time it was set to save.

    Pairs the source did not finish are not ranked, so by default they stay
    claimable at full cost. `skip_unranked` (default false) marks them SKIPPED
    instead. That is a coverage decision, not a depth gate -- on the H. pylori
    screen those 12,457 FAILED rows have median B 7,129, i.e. they are the deep
    pairs the gate exists to protect -- so it is a separate key and off by
    default. RUNNING rows are never skipped: the source may still be working on
    them, and every worker calls apply_cascade at start. Pairs the source
    itself SKIPPED (its own cascade or depth gate declined them) are unranked
    too, but they are a decision, not an unfinished result: they are always
    SKIPPED here, whatever `skip_unranked` and the gate say, and they never
    count towards `keep_fraction` of `ranked`.

    Only NOOPT and SKIPPED rows change, so running, finished and retried pairs
    are left alone. Idempotent, and a larger keep rule, a lower gate or a
    source row that later finishes hands skipped pairs back, so every worker
    calls it at start.

    Parameters
    ----------
    `cfg` — dict
        Configuration dict. `couple.cascade` keys: `source_db` (SQLite file of
        the GaussDCA run, relative to project_path; default this project's DB),
        `source_table` (default `couple.db_table`), `keep_fraction` (float in
        (0, 1]) or `keep_count` (positive int), at most one of them (default:
        keep every ranked pair), `keep_denominator` (`ranked`, the default, or
        `ranked_plus_skipped`; keep_fraction only), `rank_by` (a key of
        `CASCADE_RANKS`, default `log_lr_raw`), `min_neffoverL` (float,
        default None: no depth gate) and `skip_unranked` (bool, default False:
        pairs the source never finished stay claimable)

    Returns
    -------
    `tuple` or `None`
        (pairs ranked, pairs kept, rows changed), or None without a cascade or
        without a source run to rank; "kept" counts the ranked pairs that stay
        claimable after the gate
    """
    cascade = (cfg.get('couple') or {}).get('cascade')
    if not cascade:
        return None
    if cascade.get('adaptive') is not None:
        return _apply_adaptive_cascade(cfg)
    fraction, count, denominator = _cascade_keep_rule(cascade)
    rank_by = cascade.get('rank_by', 'log_lr_raw')
    if rank_by not in CASCADE_RANKS:
        raise ValueError(f"couple.cascade.rank_by must be one of {sorted(CASCADE_RANKS)}, got {rank_by!r}")
    gate = cascade.get('min_neffoverL')
    if gate is not None:
        gate = float(gate)
        if not gate >= 0.0:
            raise ValueError(f"couple.cascade.min_neffoverL must be >= 0 or null, got {gate}")
    skip_unfinished = bool(cascade.get('skip_unranked', False))
    if fraction is None and count is None and gate is None and not skip_unfinished:
        raise ValueError("couple.cascade does nothing: set keep_fraction or keep_count, "
                         "min_neffoverL, or skip_unranked")

    couple_table = get_table_names(cfg)['couplings']
    source_table = cascade.get('source_table', couple_table)
    source_cfg = dict(cfg)
    if cascade.get('source_db'):
        source_cfg['database'] = {**cfg['database'], 'filename': cascade['source_db']}
    if (get_db_config(source_cfg).get('filename') == get_db_config(cfg).get('filename')
            and source_table == couple_table):
        if fraction is None and count is None:
            # The block belongs to the stage that gates while scoring (the
            # GaussDCA pre-gate reads min_neffoverL itself); nothing to rank here.
            logging.info("couple.cascade names no source run (no source_db / source_table) "
                         "and no keep rule: nothing to rank, the gate applies in the scoring stage")
            return None
        raise ValueError("couple.cascade needs the GaussDCA run in another DB (source_db) "
                         "or another table (source_table)")

    src = connect_db_ro(source_cfg)
    try:
        cur = src.cursor()
        quoted_source = quote_identifier(source_table)
        # An older source DB may predate the neffoverL column: then nothing is gated
        cur.execute(f"SELECT * FROM {quoted_source} LIMIT 0")
        has_depth = 'neffoverL' in {d[0] for d in cur.description}
        depth_col = 'neffoverL' if has_depth else 'NULL'
        cur.execute(
            f"SELECT name, {depth_col} FROM {quoted_source} WHERE status = ? "
            f"ORDER BY {CASCADE_RANKS[rank_by]}, id",
            (STATUS['DONE'],))
        ranked = cur.fetchall()
        # Rows the source declined (its own cascade or depth gate): never claimable
        cur.execute(f"SELECT name FROM {quoted_source} WHERE status = ?", (STATUS['SKIPPED'],))
        skip_source = {row[0] for row in cur.fetchall()}
        unranked = []
        if skip_unfinished:
            # RUNNING is deliberately excluded: a source worker may still be on that
            # pair, and every worker calls apply_cascade at start, so skipping RUNNING
            # would make the kept set depend on who ran when (fused mode races).
            cur.execute(
                f"SELECT name FROM {quoted_source} "
                f"WHERE status IS NOT NULL AND status NOT IN (?, ?, ?)",
                (STATUS['DONE'], STATUS['RUNNING'], STATUS['SKIPPED']))
            unranked = [row[0] for row in cur.fetchall()]
    finally:
        src.close()
    if count is not None:
        n_keep = min(count, len(ranked))
        rule = f"top {count} pairs"
    elif fraction is not None:
        base = len(ranked) + (len(skip_source) if denominator == 'ranked_plus_skipped' else 0)
        n_keep = min(len(ranked), math.ceil(fraction * base))
        rule = f"top {fraction:g} of {denominator} ({base})"
    else:
        n_keep = len(ranked)
        rule = "every ranked pair"
    skip_rank = {name for name, _ in ranked[n_keep:]}
    skip_depth = set()
    ungated = 0
    if gate is not None:
        skip_depth = {name for name, depth in ranked[:n_keep]
                      if depth is not None and float(depth) < gate}
        ungated = sum(1 for _, depth in ranked[:n_keep] if depth is None)
    skip_unranked = set(unranked)
    skip = skip_rank | skip_depth | skip_unranked | skip_source

    # A gate that silently gates nothing is a 2-3x cost blowout, not a no-op: on the
    # H. pylori screen the top 20 % is 238,721 pairs (50.7 model h) against 76,212
    # (22.6 h) gated. Say so loudly rather than in an info line nobody reads.
    if gate is not None and not has_depth:
        logging.warning(
            f"couple.cascade.min_neffoverL = {gate:g} is set, but source table "
            f"{source_table} has no neffoverL column: NO pair is gated by depth and all "
            f"{n_keep} ranked pairs stay claimable. Re-run the source so it stores "
            f"neffoverL, or remove the gate.")
    elif gate is not None and ungated:
        logging.warning(
            f"couple.cascade.min_neffoverL = {gate:g}: {ungated} of {n_keep} kept pairs "
            f"have NULL neffoverL in {source_table} and are NOT gated by depth.")

    table = quote_identifier(couple_table)
    conn = connect_db_rw(cfg)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT id, name, status FROM {table} WHERE status IN (?, ?)",
                    (STATUS['NOOPT'], STATUS['SKIPPED']))
        changes = []
        for pair_id, name, status in cur.fetchall():
            new = STATUS['SKIPPED'] if name in skip else STATUS['NOOPT']
            if new != status:
                changes.append((new, pair_id, status))
        # The status guard keeps a pair another worker claimed meanwhile
        cur.executemany(f"UPDATE {table} SET status = ? WHERE id = ? AND status = ?", changes)
        conn.commit()
    finally:
        conn.close()
    kept = n_keep - len(skip_depth)
    logging.info(f"cascade: keeping {kept} of {len(ranked)} GaussDCA pairs from "
                 f"{source_table} ({rule} by {rank_by}"
                 + (f", neffoverL >= {gate:g}" if gate is not None else "") + "); "
                 f"skipped by rank {len(skip_rank)}, by depth {len(skip_depth)}, "
                 f"unfinished-in-source {len(skip_unranked)}, skipped-in-source {len(skip_source)}, "
                 f"ungated (NULL depth) {ungated}; {len(changes)} rows changed")
    return len(ranked), kept, len(changes)


# -- couple.cascade.adaptive: the self-consistent hand-off threshold -------------
#
# The plmDCA run releases the gated GaussDCA pairs in batches (frontier in rank
# order plus a stratified exploration draw below it), re-estimates after each
# batch how many hits a tighter threshold would lose, and stops once the
# threshold t* is reached and stable (massivora.adaptive_cascade). State lives in
# two tables of the plmDCA DB, so every worker sees the same released set:
#   cascade_adaptive      (seq, batch, name, role 'main' | 'explore')
#   cascade_adaptive_log  (batch, estimate of the threshold after that batch, stop)
# Every step runs under BEGIN IMMEDIATE (SQLite only).

_ADAPTIVE_DDL = (
    "CREATE TABLE IF NOT EXISTS cascade_adaptive (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
    "batch INTEGER NOT NULL, name TEXT NOT NULL UNIQUE, role TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS cascade_adaptive_log (batch INTEGER PRIMARY KEY, at REAL, "
    "n_done INTEGER, n_results INTEGER, frontier INTEGER, t_star INTEGER, t_star_rank REAL, "
    "found REAL, total_hat REAL, budget REAL, missed_ucb_now REAL, recall_hat REAL, theta REAL, "
    "released INTEGER, stop INTEGER)",
)


def _adaptive_population(cfg, cascade):
    """Ranked, gated population from the GaussDCA source: names, rank (position /
    all ranked rows, 0 = best), the source row's Neff/L, N (length), B (number),
    expected plmDCA seconds (adaptive_cascade.cost_s), names to skip and the
    number of ranked rows."""
    import numpy as np
    from massivora.adaptive_cascade import cost_s
    couple_table = get_table_names(cfg)['couplings']
    source_table = cascade.get('source_table', couple_table)
    source_cfg = dict(cfg)
    if cascade.get('source_db'):
        source_cfg['database'] = {**cfg['database'], 'filename': cascade['source_db']}
    if (get_db_config(source_cfg).get('filename') == get_db_config(cfg).get('filename')
            and source_table == couple_table):
        raise ValueError("couple.cascade.adaptive needs the GaussDCA run in another DB "
                         "(source_db) or another table (source_table)")
    rank_by = cascade.get('rank_by', 'log_lr_raw')
    if rank_by not in CASCADE_RANKS:
        raise ValueError(f"couple.cascade.rank_by must be one of {sorted(CASCADE_RANKS)}, got {rank_by!r}")
    gate = cascade.get('min_neffoverL')
    src = connect_db_ro(source_cfg)
    try:
        cur = src.cursor()
        q = quote_identifier(source_table)
        cur.execute(f"SELECT * FROM {q} LIMIT 0")
        cols = {d[0] for d in cur.description}
        depth = 'neffoverL' if 'neffoverL' in cols else 'NULL'
        cur.execute(f"SELECT name, {depth}, length, number FROM {q} WHERE status = ? "
                    f"ORDER BY {CASCADE_RANKS[rank_by]}, id", (STATUS['DONE'],))
        ranked = cur.fetchall()
        cur.execute(f"SELECT name FROM {q} WHERE status = ?", (STATUS['SKIPPED'],))
        skip = {r[0] for r in cur.fetchall()}
        if cascade.get('skip_unranked', False):
            cur.execute(f"SELECT name FROM {q} WHERE status IS NOT NULL AND status NOT IN (?, ?, ?)",
                        (STATUS['DONE'], STATUS['RUNNING'], STATUS['SKIPPED']))
            skip |= {r[0] for r in cur.fetchall()}
    finally:
        src.close()
    if gate is not None and depth == 'NULL':
        logging.warning("couple.cascade.adaptive: min_neffoverL is set but the source has no "
                        "neffoverL column: nothing is gated")
    n_all = max(len(ranked), 1)
    names, rank, nl, N, B = [], [], [], [], []
    for i, (name, d, n_, b_) in enumerate(ranked):
        if gate is not None and d is not None and float(d) < float(gate):
            skip.add(name)
            continue
        names.append(name)
        rank.append((i + 0.5) / n_all)
        nl.append(float('nan') if d is None else float(d))
        N.append(float(n_ or 0))
        B.append(float(b_ or 0))
    N, B = np.asarray(N), np.asarray(B)
    return dict(names=names, rank=np.asarray(rank), cost=cost_s(N, B), neffoverL=np.asarray(nl), N=N, B=B,
                skip=skip, n_ranked=len(ranked), has_depth=depth != 'NULL')


def _adaptive_source(cfg, cascade):
    """(names, rank, cost, skip, n_ranked) of _adaptive_population."""
    pop = _adaptive_population(cfg, cascade)
    return pop['names'], pop['rank'], pop['cost'], pop['skip'], pop['n_ranked']


_POPULATION_CACHE = {}


def _adaptive_population_cached(cfg, cascade):
    """_adaptive_population, cached per process while the GaussDCA source DB is
    unchanged (the cost-aware mode reads it at every step)."""
    source_cfg = dict(cfg)
    if cascade.get('source_db'):
        source_cfg['database'] = {**cfg['database'], 'filename': cascade['source_db']}
    path = get_db_config(source_cfg).get('filename')
    stamp = []
    for f in (path, f"{path}-wal"):
        try:
            st = os.stat(f)
            stamp.append((st.st_mtime_ns, st.st_size))
        except (OSError, TypeError):
            stamp.append(None)
    key = (path, cascade.get('source_table'), cascade.get('rank_by', 'log_lr_raw'), cascade.get('min_neffoverL'),
           bool(cascade.get('skip_unranked', False)), get_db_config(cfg).get('filename'), tuple(stamp))
    hit = _POPULATION_CACHE.get(key)
    if hit is None:
        _POPULATION_CACHE.clear()
        hit = _POPULATION_CACHE[key] = _adaptive_population(cfg, cascade)
    return hit


def _adaptive_state(cur, names, rank, cost, acfg, couple_table):
    """Rebuild the estimator from the released set and the plmDCA results."""
    import numpy as np
    from massivora.adaptive_cascade import AdaptiveCascade
    st = AdaptiveCascade(rank, cost, acfg)
    index = {n: i for i, n in enumerate(names)}
    cur.execute("SELECT name, role FROM cascade_adaptive ORDER BY seq")
    main, explore = [], []
    for name, role in cur.fetchall():
        i = index.get(name)
        if i is not None:
            (explore if role == 'explore' else main).append(i)
    st.release(main, explore)
    cur.execute(f"SELECT name, status, log_lr_raw, high_precision_hit FROM {quote_identifier(couple_table)} "
                f"WHERE status IN (?, ?)", (STATUS['DONE'], STATUS['FAILED']))
    idx, llr, hit = [], [], []
    for name, status, l, h in cur.fetchall():
        i = index.get(name)
        if i is None or not st.done[i]:
            continue
        idx.append(i)
        ok = status == STATUS['DONE'] and l is not None
        llr.append(float(l) if ok else float('nan'))
        hit.append(bool(h) if ok else False)
    if idx:
        st.observe(idx, llr, hit)
    # observe() only sets results; replayed pairs without a result stay "released"
    st.llr[~st.done] = np.nan
    cur.execute("SELECT t_star, frontier, n_done FROM cascade_adaptive_log ORDER BY batch")
    for t_star, frontier, n_done in cur.fetchall():
        st.history.append(dict(t_star=t_star, frontier=frontier, n_done=n_done))
    return st, len(idx)


def _adaptive_cfg(cascade):
    from massivora.adaptive_cascade import config
    acfg = dict(cascade.get('adaptive') or {})
    if cascade.get('keep_fraction') is not None or cascade.get('keep_count') is not None:
        raise ValueError("couple.cascade.adaptive replaces keep_fraction / keep_count; set one or the other")
    return config(acfg)


def _base_fallback_cfg(c):
    """Batch-mode configuration of the cost-aware mode's small-population fallback
    (the base_k2 loop of the validation: top-fraction hits, batch defaults)."""
    from massivora.adaptive_cascade import config
    return config(dict(hit='top_fraction', top_fraction=c['top_fraction'], target_recall=c['target_recall'],
                       seed=c['seed']))


def adaptive_mode(cfg: dict):
    """The mode `couple.cascade.adaptive` runs in: None (off), 'batch',
    'costaware', or the cost-aware mode's small-population fallback 'base'
    (batch mode) or 'full' (every gated pair)."""
    cascade = (cfg.get('couple') or {}).get('cascade') or {}
    if cascade.get('adaptive') is None:
        return None
    acfg = _adaptive_cfg(cascade)
    if acfg.get('model') != 'costaware':
        return 'batch'
    pop = _adaptive_population_cached(cfg, cascade)
    if len(pop['names']) < acfg['min_population']:
        return acfg['small_population_mode']
    return 'costaware'


def adaptive_cascade_step(cfg: dict, _init=False):
    """
    One step of `couple.cascade.adaptive`.

    Batch mode (default, `model: batch`): called by a worker whose claim came
    back empty. While released pairs are still claimable it does nothing
    ('wait'). Otherwise it re-estimates the threshold from the plmDCA results so
    far, logs it, and either stops ('stopped': converged or nothing left) or
    releases the next batch ('released'). Returns dict(status, released, estimate).

    `model: costaware`: see _adaptive_costaware_step; below `min_population`
    gated pairs it runs the batch mode ('base') or every gated pair ('full').
    """
    cascade = (cfg.get('couple') or {}).get('cascade') or {}
    if cascade.get('adaptive') is None:
        return dict(status='off', released=0)
    acfg = _adaptive_cfg(cascade)
    if acfg.get('model') == 'costaware':
        pop = _adaptive_population_cached(cfg, cascade)
        if len(pop['names']) < acfg['min_population'] and acfg['small_population_mode'] == 'base':
            src = (pop['names'], pop['rank'], pop['cost'], pop['skip'], pop['n_ranked'])
            return dict(_adaptive_batch_step(cfg, _base_fallback_cfg(acfg), src, _init), mode='base')
        full = len(pop['names']) < acfg['min_population']
        return _adaptive_costaware_step(cfg, acfg, pop, full=full)
    return _adaptive_batch_step(cfg, acfg, _adaptive_source(cfg, cascade), _init)


def _adaptive_batch_step(cfg, acfg, src, _init=False):
    import time
    import numpy as np
    names, rank, cost, skip_src, n_ranked = src
    couple_table = get_table_names(cfg)['couplings']
    table = quote_identifier(couple_table)
    conn = connect_db_rw(cfg)
    conn.isolation_level = None
    try:
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        for ddl in _ADAPTIVE_DDL:
            cur.execute(ddl)
        cur.execute("SELECT * FROM cascade_adaptive_log LIMIT 0")
        if 'stop' not in {d[0] for d in cur.description}:
            raise ValueError("couple.cascade.adaptive: this DB holds a run of the cost-aware mode; "
                             "the batch mode needs a fresh plmDCA DB")
        cur.execute("SELECT stop FROM cascade_adaptive_log ORDER BY batch DESC LIMIT 1")
        last = cur.fetchone()
        if last and last[0]:
            cur.execute("COMMIT")
            return dict(status='stopped', released=0)
        cur.execute(f"SELECT COUNT(*) FROM {table} t JOIN cascade_adaptive a ON a.name = t.name "
                    f"WHERE t.status IN (?, ?)", (STATUS['NOOPT'], STATUS['PENDING']))
        if cur.fetchone()[0] and not _init:
            cur.execute("COMMIT")
            return dict(status='wait', released=0)
        cur.execute("SELECT COUNT(*) FROM cascade_adaptive")
        if _init and cur.fetchone()[0]:
            cur.execute("COMMIT")
            return dict(status='wait', released=0)
        st, n_results = _adaptive_state(cur, names, rank, cost, acfg, couple_table)
        cur.execute("SELECT COALESCE(MAX(batch), 0) FROM cascade_adaptive_log")
        batch = int(cur.fetchone()[0]) + 1
        est, stop = {}, 0
        if st.done.any():
            st.rng = np.random.default_rng([int(acfg['seed']), batch])
            est = st.estimate()
            stop = int(st.converged())
        main, explore = (np.array([], int), np.array([], int)) if stop else st.next_batch()
        if not stop and not len(main) and not len(explore):
            stop = 1
        rel = [(batch, names[i], 'main') for i in main] + [(batch, names[i], 'explore') for i in explore]
        cur.executemany("INSERT OR IGNORE INTO cascade_adaptive (batch, name, role) VALUES (?, ?, ?)", rel)
        cur.executemany(f"UPDATE {table} SET status = ? WHERE name = ? AND status = ?",
                        [(STATUS['NOOPT'], n, STATUS['SKIPPED']) for _, n, _ in rel])
        cur.execute("INSERT INTO cascade_adaptive_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (batch, time.time(), int(st.done.sum()), n_results, est.get('frontier'), est.get('t_star'),
                     est.get('t_star_rank'), est.get('found'), est.get('total_hat'), est.get('budget'),
                     est.get('missed_ucb_now'), est.get('recall_hat'), est.get('theta'), len(rel), stop))
        cur.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()
    logging.info(f"cascade.adaptive batch {batch}: {len(names)} gated pairs, {int(st.done.sum())} released, "
                 f"{n_results} with results; t* = {est.get('t_star')} (rank {est.get('t_star_rank')}), "
                 f"frontier {est.get('frontier')}, est. recall {est.get('recall_hat')}; "
                 + ("STOP" if stop else f"released {len(main)} + {len(explore)} exploration"))
    return dict(status='stopped' if stop else 'released', released=len(rel), estimate=est)


def _apply_adaptive_cascade(cfg):
    """apply_cascade for `couple.cascade.adaptive`: the first batch is released if
    none is; every ranked pair outside the released set is SKIPPED."""
    cascade = cfg['couple']['cascade']
    acfg = _adaptive_cfg(cascade)
    adaptive_cascade_step(cfg, _init=True)
    if acfg.get('model') == 'costaware':
        pop = _adaptive_population_cached(cfg, cascade)
        names, skip_src, n_ranked = pop['names'], pop['skip'], pop['n_ranked']
    else:
        names, rank, cost, skip_src, n_ranked = _adaptive_source(cfg, cascade)
    table = quote_identifier(get_table_names(cfg)['couplings'])
    conn = connect_db_rw(cfg)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM cascade_adaptive")
        released = {r[0] for r in cur.fetchall()}
        population = set(names)
        cur.execute(f"SELECT id, name, status FROM {table} WHERE status IN (?, ?)",
                    (STATUS['NOOPT'], STATUS['SKIPPED']))
        changes = []
        for pid, name, status in cur.fetchall():
            if name in skip_src or (name in population and name not in released):
                new = STATUS['SKIPPED']
            elif name in released:
                new = STATUS['NOOPT']
            else:
                new = status        # neither ranked nor gated: unfinished in source, left as is
            if new != status:
                changes.append((new, pid, status))
        cur.executemany(f"UPDATE {table} SET status = ? WHERE id = ? AND status = ?", changes)
        conn.commit()
    finally:
        conn.close()
    kept = len(released & population)
    logging.info(f"cascade.adaptive: {len(population)} gated of {n_ranked} ranked pairs; "
                 f"{kept} released so far; {len(changes)} rows changed")
    return n_ranked, kept, len(changes)


def reclaim_stale_claims(cur, table, timeout, released_only=False):
    """
    Hand back claims older than `timeout` seconds that never finished (a worker
    killed without handing its pairs back leaves them RUNNING / RETRYING with
    claimed_at set): RUNNING -> NOOPT, RETRYING -> PENDING, no attempt spent --
    massiveilance.fix_zombie_tasks' rule, by age instead of by SLURM job. A
    worker that is still running such a pair can still commit its result.
    `released_only`: only pairs in cascade_adaptive. Returns the rows reclaimed.
    """
    import time
    extra = " AND name IN (SELECT name FROM cascade_adaptive)" if released_only else ""
    cur.execute(f"UPDATE {table} SET status = CASE status WHEN ? THEN ? WHEN ? THEN ? ELSE status END, "
                f"claimed_at = NULL, job_id = NULL "
                f"WHERE status IN (?, ?) AND claimed_at IS NOT NULL AND claimed_at < ?{extra}",
                (STATUS['RUNNING'], STATUS['NOOPT'], STATUS['RETRYING'], STATUS['PENDING'],
                 STATUS['RUNNING'], STATUS['RETRYING'], time.time() - float(timeout)))
    n = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    if n:
        logging.warning(f"reclaimed {n} pair(s) claimed more than {float(timeout):.0f} s ago and never finished")
    return n


# -- couple.cascade.adaptive, model: costaware -----------------------------------
#
# massivora.adaptive_cascade.CostAwareThreshold, driven from the plmDCA DB:
#   cascade_adaptive      (seq, batch = block, name, role): role 'explore' for the
#                         stratified random convergence pairs, 'final' for the
#                         pairs released after the freeze
#   cascade_adaptive_obs  (name, k): the estimate whose check_every results
#                         included the pair (-1: finished without a result)
#   cascade_adaptive_log  one row per estimate k (k = 0: the start), with the
#                         pairs observed / released / in flight, t, s*, the
#                         relative change, the consecutive passes, frozen, theta
#                         and beta; the frozen row also carries the release
#                         pointer (final_next) into the frozen s order
# Each step runs under BEGIN IMMEDIATE and is safe from any worker at any time:
#   1. claims older than running_timeout are reclaimed (they count as in flight
#      until then);
#   2. new results are taken in release order in chunks of exactly check_every;
#      one estimate per chunk, warm-started from the previous estimate's beta
#      (from the log), logged;
#   3. on freeze, unclaimed convergence pairs with s < s* are withdrawn and the
#      pairs with s >= s* are released, highest s first, release_block at a time;
#   4. otherwise blocks of check_every stratified random pairs are released
#      until `lead` released pairs are claimable (pairs in flight count as
#      processed: never drawn again, counted in the stratum allocation).
# A frozen state is read back from the log (beta, s*, pointer), never recomputed
# from new data, so a restarted worker carries on with the same selection.

_COSTAWARE_DDL = (
    "CREATE TABLE IF NOT EXISTS cascade_adaptive (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
    "batch INTEGER NOT NULL, name TEXT NOT NULL UNIQUE, role TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS cascade_adaptive_obs (name TEXT PRIMARY KEY, k INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS cascade_adaptive_log (k INTEGER PRIMARY KEY, at REAL, mode TEXT, "
    "n_pop INTEGER, n_obs INTEGER, n_released INTEGER, n_inflight INTEGER, n_failed INTEGER, t REAL, "
    "t_prev REAL, rel_change REAL, check_ok INTEGER, streak INTEGER, frozen INTEGER, s_star REAL, "
    "theta REAL, found REAL, total_hat REAL, missed_hat REAL, beta TEXT, est_s REAL, released INTEGER, "
    "withdrawn INTEGER, reclaimed INTEGER, n_selected INTEGER, final_next INTEGER, reason TEXT)",
)
_COSTAWARE_LOG_COLS = ('k', 'at', 'mode', 'n_pop', 'n_obs', 'n_released', 'n_inflight', 'n_failed', 't', 't_prev',
                       'rel_change', 'check_ok', 'streak', 'frozen', 's_star', 'theta', 'found', 'total_hat',
                       'missed_hat', 'beta', 'est_s', 'released', 'withdrawn', 'reclaimed', 'n_selected',
                       'final_next', 'reason')

_COSTAWARE_CACHE = {}


def _costaware_model(pop, c):
    """CostAwareThreshold, the name index and the frozen order, cached per
    population and configuration."""
    import json
    from massivora.adaptive_cascade import CostAwareThreshold
    key = (id(pop), json.dumps(c, sort_keys=True, default=str))
    hit = _COSTAWARE_CACHE.get(key)
    if hit is None or hit['pop'] is not pop:
        _COSTAWARE_CACHE.clear()
        model = CostAwareThreshold(pop['rank'], pop['cost'], pop['neffoverL'], pop['N'], pop['B'], c)
        hit = _COSTAWARE_CACHE[key] = dict(pop=pop, model=model, index={n: i for i, n in enumerate(pop['names'])},
                                           order={})
    return hit


def _costaware_log(cur, row):
    cols = [k for k in _COSTAWARE_LOG_COLS if k in row]
    cur.execute(f"INSERT INTO cascade_adaptive_log ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[k] for k in cols])


def _costaware_last(cur, frozen=False):
    where = " WHERE frozen = 1" if frozen else ""
    cur.execute(f"SELECT {', '.join(_COSTAWARE_LOG_COLS)} FROM cascade_adaptive_log{where} ORDER BY k DESC LIMIT 1")
    r = cur.fetchone()
    return dict(zip(_COSTAWARE_LOG_COLS, r)) if r else None


def _costaware_claimable(cur, table):
    cur.execute(f"SELECT COUNT(*) FROM cascade_adaptive a JOIN {table} t ON t.name = a.name "
                f"WHERE t.status IN (?, ?) AND t.claimed_at IS NULL", (STATUS['NOOPT'], STATUS['PENDING']))
    return int(cur.fetchone()[0])


def _costaware_release(cur, table, rows):
    """Insert (batch, name, role) rows into cascade_adaptive and make them claimable."""
    cur.executemany("INSERT OR IGNORE INTO cascade_adaptive (batch, name, role) VALUES (?, ?, ?)", rows)
    cur.executemany(f"UPDATE {table} SET status = ? WHERE name = ? AND status = ?",
                    [(STATUS['NOOPT'], r[1], STATUS['SKIPPED']) for r in rows])
    return len(rows)


def _costaware_release_final(cur, table, ca, frozen, c):
    """Release the next pairs of the frozen selection (highest s first) until
    release_block released pairs are claimable. Returns (released, exhausted)."""
    import json
    import numpy as np
    model, names = ca['model'], ca['pop']['names']
    okey = (frozen['beta'], frozen['s_star'], frozen['mode'])
    order = ca['order'].get(okey)
    if order is None:
        if frozen['mode'] == 'full':
            order = model.order_rank
        else:
            order = model.selection(json.loads(frozen['beta']), float(frozen['s_star']))[1]
        ca['order'] = {okey: order}
    ptr = int(frozen['final_next'] or 0)
    if ptr >= len(order):
        return 0, True
    need = c['release_block'] - _costaware_claimable(cur, table)
    if need <= 0:
        return 0, False
    cur.execute("SELECT name FROM cascade_adaptive WHERE role != 'final'")
    conv = {r[0] for r in cur.fetchall()}
    cur.execute("SELECT COALESCE(MAX(batch), 0) FROM cascade_adaptive")
    block = int(cur.fetchone()[0]) + 1
    new = []
    while len(new) < need and ptr < len(order):
        name = names[int(order[ptr])]
        ptr += 1
        if name not in conv:
            new.append((block, name, 'final'))
    _costaware_release(cur, table, new)
    cur.execute("UPDATE cascade_adaptive_log SET final_next = ?, released = COALESCE(released, 0) + ? WHERE k = ?",
                (ptr, len(new), frozen['k']))
    return len(new), ptr >= len(order)


def _adaptive_costaware_step(cfg, c, pop, full=False):
    """
    One step of `couple.cascade.adaptive` with `model: costaware` (see the
    comment above). `full`: the small-population fallback that runs every gated
    pair (frozen at the start, in rank order). Returns dict(status = released |
    wait | frozen | done, released, estimates, t, frozen, mode).
    """
    import json
    import time
    import numpy as np
    names = pop['names']
    n = len(names)
    mode = 'full' if full else 'costaware'
    if n == 0:
        return dict(status='done', released=0, estimates=0, t=None, frozen=True, mode=mode)
    if not full and not pop.get('has_depth', True):
        raise ValueError("couple.cascade.adaptive (model costaware) needs the source's neffoverL column")
    ca = _costaware_model(pop, c)
    model, index = ca['model'], ca['index']
    table = quote_identifier(get_table_names(cfg)['couplings'])
    conn = connect_db_rw(cfg)
    conn.isolation_level = None
    n_est, released, withdrawn, frozen_now, est, exhausted = 0, 0, 0, False, None, False
    try:
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        for ddl in _COSTAWARE_DDL:
            cur.execute(ddl)
        cur.execute("SELECT * FROM cascade_adaptive_log LIMIT 0")
        if 'frozen' not in {d[0] for d in cur.description}:
            raise ValueError("couple.cascade.adaptive: this DB holds a run of the batch mode; "
                             "model: costaware needs a fresh plmDCA DB")
        last = _costaware_last(cur)
        if last is None:                                            # k = 0: the start
            last = dict(k=0, at=time.time(), mode=mode, n_pop=n, n_obs=0, n_released=0, t=c['start_fraction'],
                        streak=0, frozen=int(full), beta=None, released=0, final_next=0,
                        s_star=-math.inf if full else None, n_selected=n if full else None,
                        reason='small population: every gated pair' if full else 'start')
            _costaware_log(cur, last)
        elif last['mode'] != mode:
            raise ValueError(f"couple.cascade.adaptive: this DB holds a {last['mode']} run, the configuration "
                             f"now asks for {mode} (min_population / small_population_mode changed?)")
        reclaimed = reclaim_stale_claims(cur, table, c['running_timeout'], released_only=True)

        if not last['frozen']:
            # released pairs, their state and the results not used yet
            cur.execute(f"SELECT a.name, a.role, t.status, t.log_lr_raw, t.claimed_at, o.k "
                        f"FROM cascade_adaptive a LEFT JOIN {table} t ON t.name = a.name "
                        f"LEFT JOIN cascade_adaptive_obs o ON o.name = a.name ORDER BY a.seq")
            rel = cur.fetchall()
            rel_m = np.zeros(n, bool)
            explored = np.zeros(n, bool)
            failed = np.zeros(n, bool)
            new_ok, new_bad, unclaimed, claimable = [], [], [], 0
            for name, role, status, llr, claimed, k in rel:
                i = index.get(name)
                if i is None:
                    continue
                rel_m[i] = True
                explored[i] = role == 'explore'
                if k == -1:
                    failed[i] = True
                elif k is None:
                    if status == STATUS['DONE'] and llr is not None:
                        new_ok.append((i, name, float(llr)))
                    elif status in (STATUS['DONE'], STATUS['FAILED'], STATUS['SKIPPED'], None):
                        new_bad.append(name)
                        failed[i] = True
                if status in (STATUS['NOOPT'], STATUS['PENDING']) and claimed is None:
                    claimable += 1
                    if status == STATUS['NOOPT'] and role == 'explore':
                        unclaimed.append(i)
            cur.executemany("INSERT OR IGNORE INTO cascade_adaptive_obs (name, k) VALUES (?, -1)",
                            [(nm,) for nm in new_bad])
            cur.execute(f"SELECT o.name, t.log_lr_raw FROM cascade_adaptive_obs o JOIN {table} t ON t.name = o.name "
                        f"WHERE o.k > 0 ORDER BY o.k, o.rowid")
            obs, llr_o = [], []
            for name, llr in cur.fetchall():
                i = index.get(name)
                if i is not None and llr is not None:
                    obs.append(i)
                    llr_o.append(float(llr))

            # one estimate per check_every new results
            B = c['check_every']
            k, t_prev, streak = int(last['k']), float(last['t']), int(last['streak'] or 0)
            beta = json.loads(last['beta']) if last['beta'] else None
            while len(new_ok) >= B and not frozen_now:
                chunk, new_ok = new_ok[:B], new_ok[B:]
                k += 1
                cur.executemany("INSERT INTO cascade_adaptive_obs (name, k) VALUES (?, ?)",
                                [(nm, k) for _, nm, _ in chunk])
                for i, _, l in chunk:
                    obs.append(i)
                    llr_o.append(l)
                t0 = time.perf_counter()
                est = model.estimate(obs, llr_o, rel_m, failed, beta)
                est_s = time.perf_counter() - t0
                ok, rel_change = model.check(t_prev, est['t'])
                streak = streak + 1 if ok else 0
                frozen_now = streak >= c['freeze_consecutive']
                row = dict(k=k, at=time.time(), mode=mode, n_pop=n, n_obs=est['n_obs'], n_released=int(rel_m.sum()),
                           n_inflight=est['n_inflight'], n_failed=est['n_failed'], t=est['t'], t_prev=t_prev,
                           rel_change=rel_change, check_ok=int(ok), streak=streak, frozen=int(frozen_now),
                           s_star=est['s_star'], theta=est['theta'], found=est['found'], total_hat=est['total_hat'],
                           missed_hat=est['missed_hat'], beta=json.dumps(est['beta']), est_s=est_s,
                           reclaimed=reclaimed, released=0, final_next=0,
                           reason='converged' if frozen_now else None)
                if frozen_now:
                    sel = model.selection(est['beta'], est['s_star'])[0]
                    out = [names[i] for i in unclaimed if not sel[i]]
                    if out:
                        cur.executemany(f"UPDATE {table} SET status = ? WHERE name = ? AND status = ? "
                                        f"AND claimed_at IS NULL",
                                        [(STATUS['SKIPPED'], nm, STATUS['NOOPT']) for nm in out])
                        cur.executemany("DELETE FROM cascade_adaptive WHERE name = ?", [(nm,) for nm in out])
                        for nm in out:
                            rel_m[index[nm]] = False
                    withdrawn = len(out)
                    row.update(withdrawn=withdrawn, n_selected=int(sel.sum()))
                _costaware_log(cur, row)
                t_prev, beta = est['t'], est['beta']
                n_est += 1
            if not frozen_now:
                # convergence blocks until `lead` released pairs are claimable
                cur.execute("SELECT COALESCE(MAX(batch), 0) FROM cascade_adaptive")
                block = int(cur.fetchone()[0])
                new = []
                while claimable + len(new) < c['lead']:
                    block += 1
                    idx = model.block(rel_m, explored, np.random.default_rng([c['seed'], block]), B)
                    if not len(idx):
                        exhausted = True
                        break
                    rel_m[idx] = True
                    explored[idx] = True
                    new += [(block, names[i], 'explore') for i in idx]
                released = _costaware_release(cur, table, new)
                exhausted = exhausted or bool(rel_m.all())
                if n_est and released:
                    cur.execute("UPDATE cascade_adaptive_log SET released = ? WHERE k = ?", (released, k))
            last = _costaware_last(cur)
        if last['frozen']:
            released_f, exhausted = _costaware_release_final(cur, table, ca, last, c)
            released += released_f
        cur.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()
    frozen = bool(last['frozen'])
    if n_est or released or withdrawn:
        logging.info(f"cascade.adaptive (costaware): {n} gated pairs; {n_est} estimate(s), last k = {last['k']}: "
                     f"t = {float(last['t']):.4f}, s* = {last['s_star']}, {last['n_obs']} observed; "
                     + (f"FROZEN: released {released} (pairs with s >= s*, highest s first), withdrew {withdrawn}"
                        if frozen else f"released {released} convergence pairs"))
    status = 'released' if released else ('done' if exhausted else ('frozen' if frozen else 'wait'))
    return dict(status=status, released=released, estimates=n_est, t=float(last['t']), frozen=frozen, mode=mode,
                estimate=est)
