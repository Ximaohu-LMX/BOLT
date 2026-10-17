"""Apply reviewed keyed-hook source to the bundled PostgreSQL 14.5 source."""
import argparse
from pathlib import Path
from .protocol import ROOT,require,sha,write_json

def patch(source):
    source=Path(source).resolve()
    path=source/'src/backend/optimizer/path/clausesel.c'
    text=path.read_text()
    require('job_replace' not in text,'patch already applied')
    start=text.index('typedef char TYPCATEGORY;')
    end=text.index('char** splitArrayString',start)
    text=text[:start]+'typedef char TYPCATEGORY;\n#include "pg_job_hook.h"\n\n'+text[end:]
    start=text.index('            if(!is_join_condition(opclause))')
    end=text.index('\n\t\t}',start)
    text=text[:start]+'            if(!is_join_condition(opclause))\n                s1 = job_replace(root, clause, s1);'+text[end:]
    start=text.index('else if (IsA(clause, ScalarArrayOpExpr))')
    end=text.index('\n\t}',start)
    text=text[:end]+'\n        if (!treat_as_join_clause(root, clause, rinfo, varRelid, sjinfo))\n            s1 = job_replace(root, clause, s1);'+text[end:]
    path.write_text(text)
    hook=path.parent/'pg_job_hook.h';hook.write_bytes((ROOT/'job_experiments/pg_job_hook.h').read_bytes())
    # File names are dedicated cluster-relative short names; inherited GUC remains.
    write_json(source.parent/'PATCH.json',dict(archive_sha256=sha(ROOT/'likeBench/postgresql-13.1.zip'),files={str(p):sha(p) for p in (path,hook)},mode='keyed four-column restrictions plus equality arrays; joins native'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');patch(p.parse_args().source)
