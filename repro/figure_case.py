"""Independently check the fixed Figure 1 illustration against the raw AUTHOR column."""
import re
from .common import ROOT, rows, read_json, digest, atomic_json
from .datasets import ensure


def like_pattern(pattern):
    result=[];escaped=False
    for char in pattern:
        if escaped:
            result.append(re.escape(char));escaped=False
        elif char=='\\':escaped=True
        elif char=='%':result.append('.*')
        elif char=='_':result.append('.')
        else:result.append(re.escape(char))
    if escaped:raise ValueError('Malformed trailing LIKE escape')
    return re.compile(''.join(result),re.DOTALL)


def main():
    ensure(['AUTHOR'],[],raw=True)
    folder=ROOT/'paper_results/reference/figure1_filled_case'
    if digest(folder/'nodes.csv') != read_json(folder/'provenance.json')['nodes_sha256']:
        raise ValueError('Figure 1 source changed')
    nodes=list(rows(folder/'nodes.csv'))
    with (ROOT/'data/AUTHOR/AUTHOR.txt').open() as f:db=[s.rstrip('\n') for s in f]
    previous=set();checks=[]
    for row in nodes:
        pattern=like_pattern(row['query'])
        matches={i for i,s in enumerate(db) if pattern.fullmatch(s)}
        count=len(matches)
        if count != int(row['true_count']) or not previous <= matches or not float(row['lower_bound']) <= count <= float(row['upper_bound']):
            raise ValueError('Figure 1 containment/count/bound mismatch')
        checks.append(dict(query=row['query'],count=count,contained=True))
        previous=matches
    atomic_json(ROOT/'paper_results/generated/figure1_audit.json',dict(status='PASS',checks=checks,source_sha256=digest(folder/'nodes.csv')))
    print('PASS Figure 1 exact AUTHOR counts, bounds, and containment')


if __name__=='__main__':main()
