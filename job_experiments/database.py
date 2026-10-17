"""Explicit read-only PostgreSQL export and exact counting, no DB provisioning."""
import hashlib
import json
from .protocol import COLUMNS, require


def connect(service, timeout_ms):
    import psycopg2
    require(service and all(c.isalnum() or c in '_-.' for c in service), 'use a pg_service.conf service name')
    conn = psycopg2.connect(service=service, connect_timeout=10)
    conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(timeout_ms),))
        cur.execute('SHOW standard_conforming_strings')
        require(cur.fetchone()[0] == 'on', 'standard_conforming_strings must be on')
    return conn


def identity(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT version(), current_database(), pg_database.datcollate, pg_database.datctype '
                    'FROM pg_database WHERE datname=current_database()')
        version, database, collate, ctype = cur.fetchone()
        return dict(version=version, database=database, datcollate=collate, datctype=ctype)


def row_bytes(row):
    return (json.dumps(dict(id=row[0], value=row[1]), ensure_ascii=False, separators=(',', ':'))+'\n').encode()


def export_column(conn, column, destination=None):
    from psycopg2 import sql
    require(column in COLUMNS, 'unknown column')
    table, col = column.split('.')
    h, total, nulls, empty = hashlib.sha256(), 0, 0, 0
    stream = open(destination, 'wb') if destination else None
    try:
        with conn.cursor(name='job_export_'+table) as cur:
            cur.itersize = 10000
            cur.execute(sql.SQL('SELECT id, {} FROM {} ORDER BY id').format(sql.Identifier(col), sql.Identifier(table)))
            for row in cur:
                b = row_bytes(row); h.update(b)
                total += 1; nulls += row[1] is None; empty += row[1] == ''
                if stream: stream.write(b)
    finally:
        if stream: stream.close()
    require(total > 0, 'empty JOB table: '+table)
    return dict(rows=total, nulls=nulls, empty_strings=empty, nonnull=total-nulls, sha256=h.hexdigest())


def count_patterns(conn, column, patterns):
    from psycopg2 import sql
    table, col = column.split('.')
    require(column in COLUMNS, 'unknown column')
    query = sql.SQL('SELECT count(*) FROM {} WHERE {} LIKE %s').format(sql.Identifier(table), sql.Identifier(col))
    result = {}
    with conn.cursor() as cur:
        for pattern in patterns:
            cur.execute(query, (pattern,))
            result[pattern] = cur.fetchone()[0]
    return result
