import hashlib
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
    'FAILED': 4
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

    # Covering indexes for protein interaction-network lookups
    couple_name = tables['couplings']
    for suffix, columns in (
        ('net1', 'pid1, pcontact DESC, pid2, neffoverL'),
        ('net2', 'pid2, pcontact DESC, pid1, neffoverL'),
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

    # Ensure all columns exist
    ensure_columns(cfg)
