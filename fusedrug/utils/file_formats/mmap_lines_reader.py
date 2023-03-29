import mmap
from typing import Union


def mmap_lines_reader(filename: str, decode: bool = False, verbose: int = 0) -> Union[str,bytes]:
    """
    yields line by line, and uses mmap to map the file to memory.
    This is useful when reading big file (and having enough RAM to support it)

    Experienced drastic performance improvement compared to plain reading of massive text files.

    example usage:

    for line in mmap_lines_reader('some_large_file.txt', decode=True):
        if len(line)==123:
            print('omg!')
    """

    with open(filename, "rt") as f:
        mm_read = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # useful for massive files
        linenum = 0
        line = None
        while True:
            line = mm_read.readline()
            if line == b"":
                break

            if decode:
                line = line.decode()

            yield line

            if verbose > 0 and linenum > 0:
                if not linenum % 10**5:
                    print("line num=", linenum)
