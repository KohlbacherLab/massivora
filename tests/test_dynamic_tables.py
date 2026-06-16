"""
Unit tests for dynamic database table name functionality.

Tests the ability to configure custom table names via the db_table field
in the project configuration, including:
- get_table_names() helper function
- quote_identifier() SQL injection prevention
- create_job_db() with custom table names
- Foreign key constraints with dynamic tables
- Backwards compatibility with default table names
"""

import os
import sqlite3
import tempfile

from massivora.db import (
    create_job_db,
    get_table_names,
    quote_identifier,
    connect_db,
    STATUS,
)


def test_get_table_names_with_custom_config():
    """Test get_table_names extracts custom table names from config."""
    cfg = {
        'align': {'db_table': 'my_alignments'},
        'couple': {'db_table': 'my_couplings'}
    }

    result = get_table_names(cfg)

    assert result['alignments'] == 'my_alignments', \
        f"Expected 'my_alignments', got {result['alignments']}"
    assert result['couplings'] == 'my_couplings', \
        f"Expected 'my_couplings', got {result['couplings']}"


def test_get_table_names_with_defaults():
    """Test get_table_names returns defaults when db_table not specified."""
    cfg = {
        'align': {},
        'couple': {}
    }

    result = get_table_names(cfg)

    assert result['alignments'] == 'alignments', \
        f"Expected default 'alignments', got {result['alignments']}"
    assert result['couplings'] == 'couplings', \
        f"Expected default 'couplings', got {result['couplings']}"


def test_get_table_names_with_partial_config():
    """Test get_table_names with only one custom table name."""
    cfg = {
        'align': {'db_table': 'custom_align'},
        'couple': {}
    }

    result = get_table_names(cfg)

    assert result['alignments'] == 'custom_align', \
        f"Expected 'custom_align', got {result['alignments']}"
    assert result['couplings'] == 'couplings', \
        f"Expected default 'couplings', got {result['couplings']}"


def test_quote_identifier_normal():
    """Test quote_identifier with normal table names."""
    assert quote_identifier('alignments') == '"alignments"', \
        "Failed to quote normal identifier"
    assert quote_identifier('my_table') == '"my_table"', \
        "Failed to quote identifier with underscore"
    assert quote_identifier('table123') == '"table123"', \
        "Failed to quote identifier with numbers"


def test_quote_identifier_sql_injection():
    """Test quote_identifier prevents SQL injection attempts."""
    # Test various SQL injection attempts
    malicious_inputs = [
        'table"; DROP TABLE users; --',
        "table' OR '1'='1",
        'table; DELETE FROM data',
        'table\'; --',
    ]

    for malicious in malicious_inputs:
        quoted = quote_identifier(malicious)
        assert quoted.startswith('"') and quoted.endswith('"'), \
            f"Failed to quote malicious input: {malicious}"
        # Ensure the quoted result doesn't have unescaped quotes
        inner = quoted[1:-1]
        # Count unescaped quotes (quotes not followed by another quote)
        unescaped = 0
        i = 0
        while i < len(inner):
            if inner[i] == '"':
                if i + 1 < len(inner) and inner[i + 1] == '"':
                    i += 2  # Skip escaped quote
                else:
                    unescaped += 1
                    i += 1
            else:
                i += 1
        assert unescaped == 0, \
            f"Found unescaped quotes in: {quoted}"


def test_quote_identifier_with_quotes():
    """Test quote_identifier properly escapes existing quotes."""
    result = quote_identifier('my"table')
    expected = '"my""table"'
    assert result == expected, \
        f"Expected {expected}, got {result}"

    result = quote_identifier('table"with"many"quotes')
    expected = '"table""with""many""quotes"'
    assert result == expected, \
        f"Expected {expected}, got {result}"


def test_create_job_db_with_custom_tables():
    """Test create_job_db creates tables with custom names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        cfg = {
            'align': {'db_table': 'test_alignments'},
            'couple': {'db_table': 'test_couplings'}
        }

        create_job_db(db_path, cfg)

        # Verify database exists
        assert os.path.exists(db_path), "Database file not created"

        # Check table names
        conn = connect_db(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert 'test_alignments' in tables, \
            f"Custom alignments table not found. Tables: {tables}"
        assert 'test_couplings' in tables, \
            f"Custom couplings table not found. Tables: {tables}"
        assert 'jobs' in tables, \
            f"Jobs table not found. Tables: {tables}"

        conn.close()


def test_create_job_db_with_default_tables():
    """Test create_job_db creates default tables when config is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        create_job_db(db_path, None)

        # Check table names
        conn = connect_db(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert 'alignments' in tables, \
            f"Default alignments table not found. Tables: {tables}"
        assert 'couplings' in tables, \
            f"Default couplings table not found. Tables: {tables}"

        conn.close()


def test_custom_tables_schema():
    """Test that custom tables have the correct schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        cfg = {
            'align': {'db_table': 'my_align'},
            'couple': {'db_table': 'my_couple'}
        }

        create_job_db(db_path, cfg)

        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Check alignments table schema
        cursor.execute("PRAGMA table_info(my_align)")
        align_columns = {row[1] for row in cursor.fetchall()}

        expected_align_cols = {'id', 'pid', 'start', 'end', 'align_nseqs',
                               'filtered_nseqs', 'status', 'job_id', 'claimed_at'}
        assert expected_align_cols.issubset(align_columns), \
            f"Missing columns in alignments table. Expected {expected_align_cols}, got {align_columns}"

        # Check couplings table schema
        cursor.execute("PRAGMA table_info(my_couple)")
        couple_columns = {row[1] for row in cursor.fetchall()}

        expected_couple_cols = {'id', 'pair', 'pid1', 'pid2', 'length',
                               'number', 'effnumber', 'status', 'job_id', 'claimed_at'}
        assert expected_couple_cols.issubset(couple_columns), \
            f"Missing columns in couplings table. Expected {expected_couple_cols}, got {couple_columns}"

        conn.close()


def test_foreign_key_constraints_with_custom_tables():
    """Test that foreign key constraints work with custom table names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        cfg = {
            'align': {'db_table': 'custom_align'},
            'couple': {'db_table': 'custom_couple'}
        }

        create_job_db(db_path, cfg)

        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Insert a protein into alignments
        cursor.execute("INSERT INTO custom_align (pid, status) VALUES (?, ?)",
                      ('PROT1', STATUS['DONE']))
        cursor.execute("INSERT INTO custom_align (pid, status) VALUES (?, ?)",
                      ('PROT2', STATUS['DONE']))
        conn.commit()

        # Insert a coupling referencing the proteins - this should succeed
        cursor.execute(
            "INSERT INTO custom_couple (pair, pid1, pid2, status) VALUES (?, ?, ?, ?)",
            ('PROT1-PROT2', 'PROT1', 'PROT2', STATUS['NOOPT'])
        )
        conn.commit()

        # Try to insert a coupling with non-existent protein - should fail with FK constraint
        try:
            cursor.execute(
                "INSERT INTO custom_couple (pair, pid1, pid2, status) VALUES (?, ?, ?, ?)",
                ('PROT1-PROT3', 'PROT1', 'PROT3', STATUS['NOOPT'])
            )
            conn.commit()
            # If we get here, FK constraint didn't work
            assert False, "Foreign key constraint should have been violated"
        except sqlite3.IntegrityError as e:
            # This is expected - FK constraint violated
            assert 'FOREIGN KEY constraint failed' in str(e), \
                f"Wrong error message: {e}"

        conn.close()


def test_table_operations_with_custom_names():
    """Test basic CRUD operations with custom table names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        cfg = {
            'align': {'db_table': 'step1_alignments'},
            'couple': {'db_table': 'step1_couplings'}
        }

        create_job_db(db_path, cfg)

        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Insert proteins
        proteins = ['PROT_A', 'PROT_B', 'PROT_C']
        for prot in proteins:
            cursor.execute(
                "INSERT INTO step1_alignments (pid, status) VALUES (?, ?)",
                (prot, STATUS['NOOPT'])
            )
        conn.commit()

        # Read back
        cursor.execute("SELECT pid FROM step1_alignments ORDER BY pid")
        results = [row[0] for row in cursor.fetchall()]
        assert results == proteins, \
            f"Expected {proteins}, got {results}"

        # Update
        cursor.execute(
            "UPDATE step1_alignments SET status = ? WHERE pid = ?",
            (STATUS['DONE'], 'PROT_A')
        )
        conn.commit()

        cursor.execute("SELECT status FROM step1_alignments WHERE pid = ?", ('PROT_A',))
        status = cursor.fetchone()[0]
        assert status == STATUS['DONE'], \
            f"Expected status {STATUS['DONE']}, got {status}"

        conn.close()


def test_multiple_table_sets():
    """Test creating database with multiple sets of table names (cascade simulation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        # Create first set of tables
        cfg1 = {
            'align': {'db_table': 'step1_alignments'},
            'couple': {'db_table': 'step1_couplings'}
        }
        create_job_db(db_path, cfg1)

        # Manually create second set of tables (simulating a cascade)
        cfg2 = {
            'align': {'db_table': 'step2_alignments'},
            'couple': {'db_table': 'step2_couplings'}
        }

        conn = connect_db(db_path)
        cursor = conn.cursor()

        # Create second set manually
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "step2_alignments" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pid TEXT NOT NULL,
                status INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS "step2_couplings" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                pid1 TEXT NOT NULL,
                pid2 TEXT NOT NULL,
                status INTEGER,
                FOREIGN KEY (pid1) REFERENCES "step2_alignments"(pid),
                FOREIGN KEY (pid2) REFERENCES "step2_alignments"(pid)
            )
        ''')
        conn.commit()

        # Verify all tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            'step1_alignments', 'step1_couplings',
            'step2_alignments', 'step2_couplings',
            'jobs'
        }
        assert expected_tables.issubset(tables), \
            f"Missing tables. Expected {expected_tables}, got {tables}"

        # Insert data into both sets
        cursor.execute("INSERT INTO step1_alignments (pid, status) VALUES (?, ?)",
                      ('PROT1', STATUS['DONE']))
        cursor.execute("INSERT INTO step2_alignments (pid, status) VALUES (?, ?)",
                      ('PROT2', STATUS['DONE']))
        conn.commit()

        # Verify data isolation
        cursor.execute("SELECT COUNT(*) FROM step1_alignments")
        count1 = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM step2_alignments")
        count2 = cursor.fetchone()[0]

        assert count1 == 1 and count2 == 1, \
            "Tables should have independent data"

        conn.close()


def test_special_characters_in_table_names():
    """Test table names with special characters (but valid SQL identifiers)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        cfg = {
            'align': {'db_table': 'align_2024_03_18'},
            'couple': {'db_table': 'couple_v2_final'}
        }

        create_job_db(db_path, cfg)

        conn = connect_db(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert 'align_2024_03_18' in tables, \
            f"Table with underscores and numbers not found. Tables: {tables}"
        assert 'couple_v2_final' in tables, \
            f"Table with mixed naming not found. Tables: {tables}"

        # Verify we can insert into these tables
        cursor.execute("INSERT INTO align_2024_03_18 (pid, status) VALUES (?, ?)",
                      ('TEST', STATUS['NOOPT']))
        conn.commit()

        cursor.execute("SELECT pid FROM align_2024_03_18")
        result = cursor.fetchone()[0]
        assert result == 'TEST', f"Expected 'TEST', got {result}"

        conn.close()


if __name__ == '__main__':
    # Run all tests
    test_functions = [
        test_get_table_names_with_custom_config,
        test_get_table_names_with_defaults,
        test_get_table_names_with_partial_config,
        test_quote_identifier_normal,
        test_quote_identifier_sql_injection,
        test_quote_identifier_with_quotes,
        test_create_job_db_with_custom_tables,
        test_create_job_db_with_default_tables,
        test_custom_tables_schema,
        test_foreign_key_constraints_with_custom_tables,
        test_table_operations_with_custom_names,
        test_multiple_table_sets,
        test_special_characters_in_table_names,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")

    if failed > 0:
        exit(1)
