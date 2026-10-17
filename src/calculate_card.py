"""
calculate_cardinality.py
计算LIKE模式的真实基数（PostgreSQL版本）

使用方法:
    # 单文件处理
    python calculate_cardinality.py \
        --query_file train_patterns.txt \
        --output_file train.txt \
        --db_name testdb \
        --table_name author

    # 批量处理 train/val/test（自动匹配 *_patterns.txt）
    python calculate_cardinality.py \
        --input_dir ./stage2_train_test \
        --output_dir ./stage2_train_test \
        --db_name string_db \
        --table_name author

输出格式 (tab分隔): pattern\tcardinality
    例如: %Joyce%ttman%\t1
    使用 tab 分隔避免模式中包含逗号导致解析错位
"""

import argparse
import csv
import hashlib
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import psycopg2
    from psycopg2 import pool as pg_pool
except ImportError:
    print("请安装 psycopg2: pip install psycopg2-binary")
    exit(1)

def sanitize_table_name(name: str) -> str:
    # 转为小写
    name = name.lower()
    # 将非法字符（非字母数字下划线）替换为下划线
    name = re.sub(r'[^a-z0-9_]', '_', name)
    # 确保不以数字开头（在开头加上下划线）
    if name and name[0].isdigit():
        name = '_' + name
    return name

def create_connection_pool(db_config, min_conn=2, max_conn=8):
    """创建数据库连接池"""
    return pg_pool.ThreadedConnectionPool(
        min_conn, max_conn, **db_config
    )


def calculate_cardinality(conn, pattern, table_name, column_name='content'):
    """计算单个LIKE模式的基数"""
    cursor = conn.cursor()
    query = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} LIKE %s"
    cursor.execute(query, (pattern,))
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def process_one_pattern(conn_pool, pattern, table_name, column_name):
    """从连接池取连接，查询一个模式的基数，归还连接"""
    started = time.perf_counter()
    conn = conn_pool.getconn()
    try:
        count = calculate_cardinality(conn, pattern, table_name, column_name)
        return pattern, count, time.perf_counter() - started, ""
    except Exception as exc:
        return pattern, -1, time.perf_counter() - started, repr(exc)
    finally:
        conn_pool.putconn(conn)


class DetailTimingSink:
    """Chunked per-query timing writer; only the coordinator writes CSV."""

    fields = [
        "run_id", "repeat_id", "cache_protocol", "query_index",
        "pattern_sha256", "elapsed_seconds", "status", "cardinality",
    ]

    def __init__(self, path, run_id="", repeat_id="", cache_protocol=""):
        self.path = path
        self.common = {
            "run_id": run_id,
            "repeat_id": repeat_id,
            "cache_protocol": cache_protocol,
        }

    def append(self, rows):
        if not self.path or not rows:
            return
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else '.', exist_ok=True)
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({**self.common, **row})
            handle.flush()
            os.fsync(handle.fileno())


def detail_row(query_index, pattern, elapsed, cardinality, error=""):
    return {
        "query_index": query_index,
        "pattern_sha256": hashlib.sha256(pattern.encode("utf-8")).hexdigest(),
        "elapsed_seconds": f"{elapsed:.9f}",
        "status": "ERROR" if error else "DONE",
        "cardinality": cardinality,
    }


def iter_patterns(query_file):
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            pattern = line.rstrip('\n\r')
            if pattern != "":
                yield pattern


def count_patterns(query_file):
    return sum(1 for _ in iter_patterns(query_file))


def inspect_existing_output(output_file):
    completed = 0
    existing_errors = 0
    valid_bytes = 0
    malformed = 0

    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return completed, existing_errors, malformed, valid_bytes

    with open(output_file, 'rb') as f:
        for raw_line in f:
            try:
                line = raw_line.decode('utf-8').rstrip('\n\r')
            except UnicodeDecodeError:
                malformed += 1
                break
            if not line:
                valid_bytes = f.tell()
                continue
            if '\t' not in line:
                malformed += 1
                break
            _, card = line.rsplit('\t', 1)
            try:
                card = int(card)
            except ValueError:
                malformed += 1
                break
            completed += 1
            if card < 0:
                existing_errors += 1
            valid_bytes = f.tell()

    return completed, existing_errors, malformed, valid_bytes


def prepare_output_file(output_file, total_patterns, resume):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    if not resume:
        return 0, 0, 'w'

    completed, existing_errors, malformed, valid_bytes = inspect_existing_output(output_file)
    if malformed:
        print(f"  Truncating malformed tail in {output_file} at byte {valid_bytes}")
        with open(output_file, 'ab') as f:
            f.truncate(valid_bytes)

    if completed > total_patterns:
        raise ValueError(
            f"{output_file} has {completed} completed rows, "
            f"but the query file only has {total_patterns} patterns"
        )

    if completed:
        print(f"  Resume enabled: {completed} rows already written in {output_file}")
    return completed, existing_errors, 'a'


def batched(iterator, chunk_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def skip_patterns(iterator, count):
    for _ in range(count):
        try:
            next(iterator)
        except StopIteration:
            break


def empty_stats(total_patterns, skipped=0, existing_errors=0):
    return {
        "total_patterns": total_patterns,
        "skipped": skipped,
        "processed": 0,
        "errors": 0,
        "existing_errors": existing_errors,
        "zero": 0,
        "min": "",
        "max": "",
        "avg": "",
        "elapsed": 0.0,
    }


def update_stats(stats, cardinality):
    if cardinality < 0:
        stats["errors"] += 1
        return
    stats["zero"] += int(cardinality == 0)
    stats["_sum"] = stats.get("_sum", 0) + cardinality
    stats["min"] = cardinality if stats["min"] == "" else min(stats["min"], cardinality)
    stats["max"] = cardinality if stats["max"] == "" else max(stats["max"], cardinality)


def finish_stats(stats):
    valid_count = stats["processed"] - stats["errors"]
    if valid_count > 0:
        stats["avg"] = stats.get("_sum", 0) / valid_count
    stats.pop("_sum", None)
    return stats


def print_stats(stats):
    elapsed = stats["elapsed"]
    rate = stats["processed"] / elapsed if elapsed > 0 else 0
    print(f"  Processed {stats['processed']} new patterns in {elapsed:.2f}s "
          f"({rate:.1f} queries/sec)")
    print(f"\n  {'─' * 50}")
    print("  Statistics:")
    print(f"    Input total: {stats['total_patterns']}")
    print(f"    Skipped existing: {stats['skipped']}")
    print(f"    Newly processed: {stats['processed']}")
    print(f"    New zero: {stats['zero']}")
    total_errors = stats["existing_errors"] + stats["errors"]
    if total_errors > 0:
        print(f"    Errors: {total_errors} "
              f"(existing={stats['existing_errors']}, new={stats['errors']})")
    if stats["avg"] != "":
        print(f"    New min: {stats['min']}, Max: {stats['max']}, "
              f"Avg: {stats['avg']:.2f}")
    print(f"  {'─' * 50}")


def process_queries_parallel(query_file, output_file, conn_pool,
                             table_name, column_name='content',
                             num_workers=4, resume=False,
                             chunk_size=256, detail_sink=None):
    """Process queries in bounded chunks and write each completed chunk."""

    print(f"  Counting queries in {query_file}...")
    total_patterns = count_patterns(query_file)
    print(f"  Loaded {total_patterns} patterns")

    if not total_patterns:
        print(f"  No patterns to process, skipping.")
        return empty_stats(total_patterns)

    skipped, existing_errors, output_mode = prepare_output_file(
        output_file, total_patterns, resume
    )
    stats = empty_stats(total_patterns, skipped, existing_errors)
    if skipped >= total_patterns:
        print(f"  All patterns already processed: {output_file}")
        return finish_stats(stats)

    print(f"  Calculating cardinalities (workers={num_workers}, chunk_size={chunk_size})...")
    start_time = time.time()

    pattern_iter = iter_patterns(query_file)
    skip_patterns(pattern_iter, skipped)

    with ThreadPoolExecutor(max_workers=num_workers) as executor, \
            open(output_file, output_mode, encoding='utf-8') as out:
        with tqdm(total=total_patterns, initial=skipped, desc="  Processing") as pbar:
            for batch in batched(pattern_iter, chunk_size):
                results = [None] * len(batch)
                future_to_idx = {
                    executor.submit(process_one_pattern, conn_pool, pat, table_name, column_name): i
                    for i, pat in enumerate(batch)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        pattern, cardinality, query_elapsed, error = future.result()
                        results[idx] = (pattern, cardinality, query_elapsed, error)
                    except Exception as e:
                        results[idx] = (batch[idx], -1, 0.0, repr(e))
                        print(f"  ⚠️  Error on pattern '{batch[idx][:50]}...': {e}")
                    pbar.update(1)

                detail_rows = []
                for idx, (pattern, cardinality, query_elapsed, error) in enumerate(results):
                    out.write(f"{pattern}\t{cardinality}\n")
                    stats["processed"] += 1
                    update_stats(stats, cardinality)
                    detail_rows.append(detail_row(
                        skipped + stats["processed"] - 1,
                        pattern, query_elapsed, cardinality, error,
                    ))
                out.flush()
                if detail_sink:
                    detail_sink.append(detail_rows)

    stats["elapsed"] = time.time() - start_time
    finish_stats(stats)
    print_stats(stats)
    return stats


def process_queries_serial(query_file, output_file, conn,
                           table_name, column_name='content',
                           resume=False, chunk_size=256, detail_sink=None):
    """串行处理（fallback，不需要连接池）"""

    print(f"  Counting queries in {query_file}...")
    total_patterns = count_patterns(query_file)
    print(f"  Loaded {total_patterns} patterns")

    if not total_patterns:
        print(f"  No patterns to process, skipping.")
        return empty_stats(total_patterns)

    skipped, existing_errors, output_mode = prepare_output_file(
        output_file, total_patterns, resume
    )
    stats = empty_stats(total_patterns, skipped, existing_errors)
    if skipped >= total_patterns:
        print(f"  All patterns already processed: {output_file}")
        return finish_stats(stats)

    print(f"  Calculating cardinalities (serial)...")
    start_time = time.time()

    pattern_iter = iter_patterns(query_file)
    skip_patterns(pattern_iter, skipped)
    since_flush = 0
    detail_rows = []

    with open(output_file, output_mode, encoding='utf-8') as out:
        with tqdm(total=total_patterns, initial=skipped, desc="  Processing") as pbar:
            for pattern in pattern_iter:
                query_started = time.perf_counter()
                error = ""
                try:
                    cardinality = calculate_cardinality(conn, pattern, table_name, column_name)
                except Exception as exc:
                    cardinality = -1
                    error = repr(exc)
                out.write(f"{pattern}\t{cardinality}\n")
                stats["processed"] += 1
                update_stats(stats, cardinality)
                detail_rows.append(detail_row(
                    skipped + stats["processed"] - 1, pattern,
                    time.perf_counter() - query_started, cardinality, error,
                ))
                since_flush += 1
                if since_flush >= chunk_size:
                    out.flush()
                    if detail_sink:
                        detail_sink.append(detail_rows)
                    detail_rows = []
                    since_flush = 0
                pbar.update(1)
        out.flush()
        if detail_sink:
            detail_sink.append(detail_rows)

    stats["elapsed"] = time.time() - start_time
    finish_stats(stats)
    print_stats(stats)
    return stats


def append_timing_row(timing_file, row):
    if not timing_file:
        return
    os.makedirs(os.path.dirname(timing_file) if os.path.dirname(timing_file) else '.', exist_ok=True)
    fields = [
        "timestamp", "status", "query_file", "output_file", "table_name",
        "column_name", "workers", "chunk_size", "resume",
        "total_patterns", "skipped", "processed", "errors",
        "existing_errors", "elapsed_seconds", "run_id", "repeat_id",
        "cache_protocol", "query_list_sha256",
    ]
    write_header = not os.path.exists(timing_file) or os.path.getsize(timing_file) == 0
    with open(timing_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
    print(f"  Timing appended: {timing_file}")


def main():
    dataset = "IMDB"

    parser = argparse.ArgumentParser(description='Calculate cardinality for LIKE patterns (PostgreSQL)')

    # 文件参数
    parser.add_argument('--query_file', type=str, default=f'./models_single_dataset/{dataset}_query.txt', help='单个查询文件')
    parser.add_argument('--output_file', type=str, default=f'./models_single_dataset/{dataset}_card.txt', help='单个查询的输出文件')
    parser.add_argument('--input_dir', type=str, help='包含 *_patterns.txt 的目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')

    # 数据库参数
    parser.add_argument('--db_name', type=str, default='string_db', help='数据库名')
    parser.add_argument('--table_name', type=str, default=sanitize_table_name(dataset), help='表名')
    parser.add_argument('--column_name', type=str, default='content', help='列名')
    parser.add_argument('--host', type=str, default='localhost', help='主机')
    parser.add_argument('--port', type=int, default=5432, help='端口')
    parser.add_argument('--user', type=str, default='postgres', help='用户名')
    parser.add_argument('--password', type=str, default=os.getenv('PGPASSWORD'),
                        help='密码；默认交给 PGPASSWORD/.pgpass/libpq，不写入命令日志')

    # 并行参数
    parser.add_argument('--workers', type=int, default=4,
                        help='并行查询的线程数（0=串行模式）')
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='每批提交和落盘的查询数；越小，中断后丢失的未写结果越少')
    parser.add_argument('--resume', action='store_true',
                        help='从已有输出文件续跑：跳过已经写出的前缀行，并继续追加')
    parser.add_argument('--timing_file', type=str,
                        help='追加写入计数耗时和续跑统计的CSV文件')
    parser.add_argument('--detail_timing_file', type=str,
                        help='逐查询写入pattern哈希、耗时和状态的CSV文件')
    parser.add_argument('--run_id', type=str, default='')
    parser.add_argument('--repeat_id', type=str, default='')
    parser.add_argument('--cache_protocol', type=str, default='')

    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be positive")

    run_start = time.time()
    run_status = "FAILED"
    run_stats = {}

    # 数据库配置
    db_config = {
        'dbname': args.db_name,
        'user': args.user,
        'host': args.host,
        'port': args.port,
    }
    if args.password:
        db_config['password'] = args.password

    detail_sink = DetailTimingSink(
        args.detail_timing_file, args.run_id, args.repeat_id, args.cache_protocol
    )

    # 获取总行数（用单连接）
    print(f"Connecting to PostgreSQL: {args.db_name}...")
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {args.table_name}")
    total_rows = cursor.fetchone()[0]
    cursor.close()
    print(f"  Connected. Total rows in {args.table_name}: {total_rows:,}")

    use_parallel = args.workers > 0

    # 创建连接池（并行模式）
    conn_pool = None
    if use_parallel:
        conn_pool = create_connection_pool(db_config, min_conn=2, max_conn=args.workers + 2)
        print(f"  Connection pool created (max_conn={args.workers + 2})")

    try:
        if args.query_file and args.output_file:
            # === 单文件处理 ===
            print(f"\n{'=' * 60}")
            print(f"Processing: {args.query_file}")
            print(f"{'=' * 60}")

            if use_parallel:
                run_stats = process_queries_parallel(
                    args.query_file, args.output_file, conn_pool,
                    args.table_name, args.column_name, args.workers,
                    args.resume, args.chunk_size, detail_sink
                )
            else:
                run_stats = process_queries_serial(
                    args.query_file, args.output_file, conn,
                    args.table_name, args.column_name,
                    args.resume, args.chunk_size, detail_sink
                )
            if run_stats.get("errors", 0):
                run_status = "FAILED_QUERY_ERRORS"
                raise RuntimeError(
                    f"{run_stats['errors']} cardinality queries failed; refusing a partial timing run"
                )
            run_status = "DONE"

        elif args.input_dir and args.output_dir:
            # === 批量处理：自动发现 *_patterns.txt ===
            os.makedirs(args.output_dir, exist_ok=True)

            # 匹配 train_patterns.txt, val_patterns.txt, test_patterns.txt
            splits = ['train', 'valid', 'test']
            found_any = False

            for split in splits:
                query_file = os.path.join(args.input_dir, f'{split}_patterns.txt')
                output_file = os.path.join(args.output_dir, f'{split}.txt')

                if os.path.exists(query_file):
                    found_any = True
                    print(f"\n{'=' * 60}")
                    print(f"Processing {split}: {query_file}")
                    print(f"{'=' * 60}")

                    if use_parallel:
                        run_stats = process_queries_parallel(
                            query_file, output_file, conn_pool,
                            args.table_name, args.column_name,
                            args.workers, args.resume, args.chunk_size, detail_sink
                        )
                    else:
                        run_stats = process_queries_serial(
                            query_file, output_file, conn,
                            args.table_name, args.column_name,
                            args.resume, args.chunk_size, detail_sink
                        )
                else:
                    print(f"  Skipping {split}: {query_file} not found")

            if found_any:
                print(f"\n✅ All done! Results saved to {args.output_dir}/")
                run_status = "DONE"
            else:
                print(f"\n❌ No *_patterns.txt files found in {args.input_dir}/")
                print(f"   Expected: train_patterns.txt, val_patterns.txt, test_patterns.txt")
                run_status = "NO_INPUT"

        else:
            parser.print_help()
            print("\n示例:")
            print("  # 批量处理（推荐）")
            print("  python calculate_cardinality.py \\")
            print("      --input_dir ./stage2_train_test --output_dir ./stage2_train_test \\")
            print("      --db_name string_db --table_name author --workers 8")
            print()
            print("  # 单文件处理")
            print("  python calculate_cardinality.py \\")
            print("      --query_file train_patterns.txt --output_file train.txt \\")
            print("      --db_name string_db --table_name author")

    finally:
        elapsed = time.time() - run_start
        append_timing_row(args.timing_file, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": run_status,
            "query_file": args.query_file or args.input_dir,
            "output_file": args.output_file or args.output_dir,
            "table_name": args.table_name,
            "column_name": args.column_name,
            "workers": args.workers,
            "chunk_size": args.chunk_size,
            "resume": int(args.resume),
            "total_patterns": run_stats.get("total_patterns", ""),
            "skipped": run_stats.get("skipped", ""),
            "processed": run_stats.get("processed", ""),
            "errors": run_stats.get("errors", ""),
            "existing_errors": run_stats.get("existing_errors", ""),
            "elapsed_seconds": f"{elapsed:.3f}",
            "run_id": args.run_id,
            "repeat_id": args.repeat_id,
            "cache_protocol": args.cache_protocol,
            "query_list_sha256": (
                hashlib.sha256(open(args.query_file, "rb").read()).hexdigest()
                if args.query_file and os.path.isfile(args.query_file) else ""
            ),
        })
        conn.close()
        if conn_pool:
            conn_pool.closeall()


if __name__ == "__main__":
    main()
