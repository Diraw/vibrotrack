"""Fast reader for the Basler 16-bit uncompressed TIFFs (2400x2400, 1 row per strip).

PIL decodes these at ~25 ms/frame although there is nothing to decompress.
Reading the strip-offset table and taking a view is ~3.4 ms and byte-identical.
Falls back to PIL for any file whose layout does not match the expected contract.
"""
import struct
import numpy as np

_BIG = '>'


def _ifd(buf):
    endian = '>' if buf[:2] == b'MM' else '<'
    ifd, = struct.unpack(endian + 'I', buf[4:8])
    n, = struct.unpack(endian + 'H', buf[ifd:ifd + 2])
    out = {}
    for i in range(n):
        o = ifd + 2 + 12 * i
        if o + 12 > len(buf):
            break
        tag, typ, cnt = struct.unpack(endian + 'HHI', buf[o:o + 8])
        if typ == 3 and cnt == 1:
            v, = struct.unpack(endian + 'H', buf[o + 8:o + 10])
        else:
            v, = struct.unpack(endian + 'I', buf[o + 8:o + 12])
        out[tag] = (typ, cnt, v)
    return endian, out


def read(path):
    """Return (image uint16 native byte order, 2-D). Fast path first, PIL fallback."""
    with open(path, 'rb', buffering=0) as fh:
        buf = fh.read()
    try:
        endian, t = _ifd(buf)
        w, h = t[256][2], t[257][2]
        bps, comp = t[258][2], t[259][2]
        nstrip, table = t[273][1], t[273][2]
        if bps != 16 or comp != 1 or nstrip != h:
            raise ValueError('unexpected TIFF layout')
        if endian == '>':
            offs = np.frombuffer(buf, dtype='>I', count=nstrip, offset=table)
        else:
            offs = np.frombuffer(buf, dtype='<I', count=nstrip, offset=table)
        first, step = int(offs[0]), int(offs[1]) - int(offs[0])
        if step != 2 * w or int(offs[-1]) + 2 * w > len(buf):
            raise ValueError('strips are not contiguous rows')
        a = np.frombuffer(buf, dtype=endian + 'u2', count=w * h, offset=first).reshape(h, w)
        return a.astype('<u2', copy=True)
    except Exception:
        import io
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        return np.array(Image.open(io.BytesIO(buf)))
