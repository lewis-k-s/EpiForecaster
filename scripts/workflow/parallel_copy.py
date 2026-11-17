"""Copy the large files of a directory tree with concurrent block readers.

One sequential stream reads about 0.4 GB/s from GPFS (the 223 GB n400 corpus
took 9 minutes to stage). Reading 64 MiB blocks on several threads keeps
several GPFS requests in flight. Runs on the host, before the container
starts, so it uses only the standard library of the system Python 3.

Node-local disk is shared with other jobs on ACC nodes, whose leftovers can
fill it (job 48640552: 165 of 436 GB free for a 204 GB dataset, found only
after four minutes of copying). The free space is therefore checked before
any copy: the large files plus --reserve-bytes must fit, or the step fails.

A free-space check alone does not hold the space. Two jobs that start together
on one node both pass it and then fill the disk mid-copy (jobs 49064430 and
49064484: two 204 GB copies on one 480 GB disk, ENOSPC after ten minutes). The
check and the block reservation of every destination file (posix_fallocate)
therefore run under the node-wide --lock file before any byte is copied, so
the copy itself cannot run out of space. A job that loses the race on space
fails within seconds and releases what it reserved.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import os
import shutil
import sys

BLOCK_BYTES = 64 << 20


def copy_file(source, destination, streams):
    size = os.stat(source).st_size
    source_fd = os.open(source, os.O_RDONLY)
    try:
        # The destination already holds its reserved blocks; see reserve().
        destination_fd = os.open(destination, os.O_WRONLY)
        try:

            def copy_block(offset):
                end = min(offset + BLOCK_BYTES, size)
                while offset < end:
                    data = os.pread(source_fd, end - offset, offset)
                    if not data:
                        raise OSError("Unexpected end of file in " + source)
                    view = memoryview(data)
                    while view:
                        written = os.pwrite(destination_fd, view, offset)
                        offset += written
                        view = view[written:]

            with ThreadPoolExecutor(max_workers=streams) as pool:
                list(pool.map(copy_block, range(0, size, BLOCK_BYTES)))
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)
    if os.stat(destination).st_size != size:
        raise OSError("Incomplete copy of " + source)
    shutil.copystat(source, destination)


def reserve(copies, destination_root, reserve_bytes):
    """Allocate every destination file, or remove them all and exit."""
    required = sum(os.stat(source).st_size for source, _ in copies)
    disk = os.statvfs(destination_root)
    free = disk.f_bavail * disk.f_frsize
    if free < required + reserve_bytes:
        sys.exit(
            "Node-local disk has %.1f GB free at %s; staging needs %.1f GB plus a "
            "%.1f GB reserve"
            % (free / 1e9, destination_root, required / 1e9, reserve_bytes / 1e9)
        )
    reserved = []
    try:
        for source, destination in copies:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            reserved.append(destination)
            try:
                os.posix_fallocate(fd, 0, os.stat(source).st_size)
            finally:
                os.close(fd)
    except OSError as error:
        for path in reserved:
            os.unlink(path)
        if error.errno != errno.ENOSPC:
            raise
        sys.exit(
            "Node-local disk at %s filled while reserving %.1f GB for staging; "
            "another job is writing to this node's disk"
            % (destination_root, required / 1e9)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--min-bytes", type=int, required=True)
    parser.add_argument("--reserve-bytes", type=int, required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--streams", type=int, default=16)
    args = parser.parse_args()
    copies = []
    for root, _, files in os.walk(args.source, followlinks=True):
        relative = os.path.relpath(root, args.source)
        target_dir = os.path.normpath(os.path.join(args.destination, relative))
        for name in sorted(files):
            source = os.path.join(root, name)
            if os.stat(source).st_size >= args.min_bytes:
                copies.append((source, os.path.join(target_dir, name)))
    if not copies:
        return
    with open(args.lock, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reserve(copies, args.destination, args.reserve_bytes)
    for source, destination in copies:
        copy_file(source, destination, args.streams)


if __name__ == "__main__":
    main()
