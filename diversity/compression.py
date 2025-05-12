from typing import List, Optional
from pathlib import Path

import tempfile
import gzip
import os
import lzma as xz


def compression_ratio(
        data: List[str],
        algorithm: str = 'gzip',
        verbose: bool = False,
        path: Optional[str] = None
) -> float:
    """ Calculates the compression ratio for a collection of text.
     Args:
         path (str): Path to store temporarily zipped files.
         data (List[str]): Strings to compress.
         algorithm (str, optional): Either 'gzip' or 'xz'. Defaults to 'gzip'.
         verbose (bool, optional): Print out the original and compressed size separately. Defaults to False.
     Returns:
         float: Compression ratio (original size / compressed size)
     """
     
    temp_dir = None
    if not path:
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name)
    else:
        path = Path(path)

    with (path / 'original.txt').open('w+') as f:
        f.write(' '.join(data))

    original_size = os.path.getsize(os.path.join(path, "original.txt"))

    if algorithm == 'gzip':

        with gzip.GzipFile(str(path / 'compressed.gz'), 'w+') as f:
            f.write(gzip.compress(' '.join(data).encode('utf-8')))

        compressed_size = os.path.getsize(os.path.join(path, "compressed.gz"))

    elif algorithm == 'xz': 

        with xz.open(str(path / 'compressed.gz'), 'wb') as f:
            f.write(' '.join(data).encode('utf-8'))

        compressed_size = (path / "compressed.gz").stat().st_size

    if verbose: 
        print(f"Original Size: {original_size}\nCompressed Size: {compressed_size}")

    if temp_dir:
        temp_dir.cleanup()

    return round(original_size / compressed_size, 3)
    
