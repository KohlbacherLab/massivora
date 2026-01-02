import sqlite3

STATUS = {
    'NOOPT': -1,
    'DONE': 0,
    'RUNNING': 1,
    'PENDING': 2,
    'RETRYING': 3,
    'FAILED': 4
}

def connect_db(db_path):
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute('PRAGMA busy_timeout=60000;')
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def connect_db_ro(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    conn.execute('PRAGMA busy_timeout=60000;')
    return conn


def connect_db_rw(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True, timeout=60)
    conn.execute('PRAGMA busy_timeout=60000;')
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def ensure_columns(db_path):
    conn = connect_db(db_path)
    cur = conn.cursor()

    def add_col(table, col_def):
        try:
            cur.execute(f'ALTER TABLE {table} ADD COLUMN {col_def}')
            conn.commit()
        except sqlite3.OperationalError:
            pass

    add_col('couplings', 'claimed_at REAL')
    add_col('couplings', 'job_id TEXT')
    add_col('alignments', 'job_id TEXT')

    conn.close()


def create_job_db(db_path):
    conn = connect_db(db_path)
    cursor = conn.cursor()

    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS alignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pid TEXT NOT NULL,
        start INTEGER,
        end INTEGER,
        align_nseqs INTEGER,
        filtered_nseqs INTEGER,
        status INTEGER NOT NULL,
        job_id TEXT
    );
    '''
    )

    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS couplings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pid1 TEXT NOT NULL,
        pid2 TEXT NOT NULL,
        length INTEGER,
        number INTEGER,
        effnumber INTEGER,
        status INTEGER NOT NULL,
        job_id TEXT,
        claimed_at REAL,
        UNIQUE (pid1, pid2)
    );
    '''
    )

    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        status INTEGER NOT NULL
    );
    '''
    )

    conn.commit()
    conn.close()

    # Ensure migrations for older DBs
    ensure_columns(db_path)
