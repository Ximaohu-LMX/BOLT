"""Verified HTTP byte-range download, resuming an existing immutable prefix."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import time
import urllib.request
from .protocol import require, write_json, sha

URL='https://event.cwi.nl/da/job/imdb.tgz'
SIZE=1263193115


def download(prefix,output,status_dir):
    prefix,output,status_dir=Path(prefix),Path(output),Path(status_dir)
    require(not output.exists(),'archive already exists')
    offset=prefix.stat().st_size
    require(0 < offset <= SIZE,'invalid prefix size')
    ranges=[]
    for i in range(4):
        start=offset+(SIZE-offset)*i//4
        end=offset+(SIZE-offset)*(i+1)//4-1
        if start<=end: ranges.append((start,end))
    def part(bounds):
        start,end=bounds; path=prefix.with_name('imdb.range.%d-%d' % bounds)
        for retry in range(6):
            size=path.stat().st_size if path.exists() else 0
            if size==end-start+1:return path
            require(size<end-start+1,'oversized range')
            current=start+size
            try:
                req=urllib.request.Request(URL,headers={'Range':'bytes=%d-%d'%(current,end)})
                with urllib.request.urlopen(req,timeout=60) as r:
                    require(r.status==206 and r.headers['Content-Range']=='bytes %d-%d/%d'%(current,end,SIZE),'unexpected range response')
                    with path.open('ab') as f:
                        while True:
                            block=r.read(1024*1024)
                            if not block:break
                            f.write(block)
                require(path.stat().st_size==end-start+1,'short range')
                return path
            except Exception:
                if retry==5:raise
                time.sleep(2)
    write_json(status_dir/'download_status.json',dict(status='DOWNLOADING',prefix_bytes=offset,ranges=ranges,pid=os.getpid()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(part,ranges))
    target=output.with_suffix('.assembling')
    with target.open('wb') as f:
        for source in [prefix,*paths]:
            with source.open('rb') as s:shutil.copyfileobj(s,f,1024*1024)
    require(target.stat().st_size==SIZE,'assembled size mismatch')
    target.rename(output)
    write_json(status_dir/'DOWNLOAD.json',dict(status='DOWNLOADED',url=URL,bytes=SIZE,sha256=sha(output),pid=os.getpid()))
    print('DOWNLOAD_COMPLETE',str(output),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--output',required=True);p.add_argument('--status-dir',required=True)
    a=p.parse_args();download(a.prefix,a.output,a.status_dir)
