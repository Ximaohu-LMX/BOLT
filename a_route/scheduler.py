"""File locks shared by sequential training and timing runs."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack, contextmanager
import fcntl
import json
from pathlib import Path
import time
from .io import atomic_json, read_json, require, fingerprint

GLOBAL_LOCKS=('clique_b_route_parallel','clique_b_route_multiseed','clique_a_route')
TIMING_LOCKS=('clique_time_efficiency_gpu0_confirm','clique_time_efficiency_gpu_v1','clique_p4_launch')

@contextmanager
def locks(names, directory=Path('/tmp')):
    with ExitStack() as stack:
        for name in sorted(set(names)):
            f=stack.enter_context((Path(directory)/(name+'.lock')).open('a'))
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('resource lock held: '+name)
        yield
