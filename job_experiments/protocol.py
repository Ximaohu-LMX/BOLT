"""Strict extraction for the repository's flat JOB SQL workload (not a SQL planner)."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ('movie_companies.note', 'title.title', 'name.name', 'link_type.link')
TOKEN = re.compile(r"\s+|--[^\n]*|/\*[\s\S]*?\*/|'(?:''|[^'])*'|[A-Za-z_][A-Za-z_0-9]*|<>|!=|<=|>=|[().,;=<>*+-]|[0-9]+")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def normalize(pattern):
    # Same wildcard-language equivalence as canonicalize_like_query, without
    # importing torch for SQL-only preparation. Escaped LIKE is not supported.
    require('\\' not in pattern, 'escaped LIKE requires a separate encoder contract')
    return re.sub(r'[%_]+', lambda m: ('%' if '%' in m[0] else '') + '_' * m[0].count('_'), pattern)


def tokens(sql):
    out, pos = [], 0
    for m in TOKEN.finditer(sql):
        require(m.start() == pos, 'unsupported SQL token at offset %d' % pos)
        pos = m.end()
        v = m[0]
        if not v.isspace() and not v.startswith(('--', '/*')):
            out.append(v)
    require(pos == len(sql), 'unsupported trailing SQL syntax')
    return out


def extract_sql(sql, sql_id):
    t = tokens(sql)
    upper = [v.upper() if not v.startswith("'") else v for v in t]
    require(upper.count('SELECT') == 1 and 'FROM' in upper and 'WHERE' in upper,
            'only flat SELECT/FROM/WHERE JOB queries are admitted')
    require('ESCAPE' not in upper, 'explicit ESCAPE not supported')
    a, b = upper.index('FROM'), upper.index('WHERE')
    aliases = {}
    # Frozen JOB FROM syntax: table AS alias, ...; reject a changed grammar.
    groups, current = [], []
    for v in t[a+1:b] + [',']:
        if v == ',':
            groups.append(current); current = []
        else:
            current.append(v)
    for g in groups:
        require(len(g) == 3 and g[1].upper() == 'AS', 'unsupported FROM clause')
        require(g[2] not in aliases, 'duplicate table alias')
        aliases[g[2]] = g[0]
    result = []
    for i in range(b+1, len(t)-3):
        if t[i+1] != '.' or t[i] not in aliases:
            continue
        column = aliases[t[i]] + '.' + t[i+2]
        if column not in COLUMNS:
            continue
        j = i+3
        neg = upper[j] == 'NOT'
        if neg:
            j += 1
        op = upper[j]
        if op == 'IS':
            rest = upper[j+1:j+3]
            require(rest[0] == 'NULL' or rest == ['NOT', 'NULL'], 'unsupported IS predicate')
            continue  # NULL tests remain PostgreSQL-native.
        require(op in ('LIKE', '=', '<>', '!=', 'IN'), 'unsupported target-column predicate: '+op)
        require(not neg or op in ('LIKE', 'IN'), 'unsupported NOT predicate')
        if op == 'IN':
            require(t[j+1] == '(', 'IN requires literal list')
            k, values = j+2, []
            while True:
                require(k < len(t) and t[k].startswith("'"), 'IN requires string literals')
                values.append(t[k]); k += 1
                require(k < len(t), 'unterminated IN')
                if t[k] == ')': break
                require(t[k] == ',', 'invalid IN list'); k += 1
        else:
            require(t[j+1].startswith("'"), 'target comparison requires plain string literal')
            values = [t[j+1]]
        for raw in values:
            value = raw[1:-1].replace("''", "'")
            require(not any(c in value for c in '\\\r\n'), 'unsupported literal escape/newline')
            if op != 'LIKE':
                require('%' not in value and '_' not in value,
                        'literal wildcard in equality requires separate encoding')
            result.append(dict(sql_id=sql_id, column=column, alias=t[i],
                               operator=('NOT ' if neg else '')+op,
                               literal=value, pattern=normalize(value), token_offset=i))
    return result


def workload(path):
    queries = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    require(len(queries) == 59, 'expected frozen 59-query JOB subset')
    records = [r for i, q in enumerate(queries, 1) for r in extract_sql(q, i)]
    require(set(r['column'] for r in records) == set(COLUMNS), 'four modeled columns required')
    return dict(sql_sha256=sha(path), sql_count=len(queries), predicates=records,
                test_patterns={c: sorted({r['pattern'] for r in records if r['column'] == c}) for c in COLUMNS},
                note='token offsets identify SQL predicates, NOT PostgreSQL injection slots')


def seal(folder, name, payload):
    payload = dict(payload)
    payload['files'] = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
                        if p.is_file() and p.name != name and not p.name.endswith('.tmp')}
    write_json(folder / name, payload)


def verify(folder, name):
    info = read_json(folder / name)
    for rel, digest in info['files'].items():
        require(sha(folder / rel) == digest, 'changed frozen input: '+rel)
    return info
