"""Caminus's own GGUF reader: the header, the metadata and the tensor table of a GGUF file, and memory-mapped views of
the tensors' raw bytes. Written from the file format (docs/specs/spec-own-quant.md: the format is the data contract; this
module replaces gguf-py on the model path, gguf-py stays a test oracle).

Layout (little-endian): magic b'GGUF', u32 version (2 or 3), u64 tensor count, u64 metadata count; then the metadata
(key: u64 length + UTF-8 bytes, u32 value type, value) and the tensor table (name, u32 dims, u64 dims[n] innermost first,
u32 type, u64 offset relative to the data section); the data section starts at the next multiple of general.alignment
(default 32) after the table. Value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array
(u32 element type, u64 count, elements), 10 u64, 11 i64, 12 f64.

Block sizes per tensor type: (values per block, bytes per block), the ggml type ids the files use."""
import mmap
import os
import struct

import numpy as np

# ggml type id -> (name, values per block, bytes per block)
TYPES = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24),
    8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36), 10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144),
    13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292), 16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74),
    18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110), 22: ('IQ2_S', 256, 82),
    23: ('IQ4_XS', 256, 136), 24: ('I8', 1, 1), 25: ('I16', 1, 2), 26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8),
    29: ('IQ1_M', 256, 56), 30: ('BF16', 1, 2), 34: ('TQ1_0', 256, 54), 35: ('TQ2_0', 256, 66), 39: ('MXFP4', 32, 17),
    40: ('NVFP4', 64, 36), 41: ('Q1_0', 128, 18), 42: ('Q2_0', 64, 18),   # (42 Q2_0: the GSQ checkpoints' 2-bit type)
}
_SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
_NP = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32, 7: np.bool_,
       10: np.uint64, 11: np.int64, 12: np.float64}


class GGUFError(ValueError):
    pass


class Tensor:
    """one tensor: name, type (name), type_id, shape (innermost first, as in the file), offset (absolute), n_bytes, and
    data(): a read-only uint8 view of its bytes (memory-mapped)"""
    __slots__ = ('name', 'type', 'type_id', 'shape', 'offset', 'n_bytes', 'n_values', '_f')

    def __init__(self, f, name, type_id, shape, offset):
        if type_id not in TYPES: raise GGUFError(f'{name}: unknown tensor type {type_id}')
        tname, bs, ts = TYPES[type_id]
        n = 1
        for d in shape: n *= int(d)
        if n % bs: raise GGUFError(f'{name}: {n} values is not a whole number of {tname} blocks ({bs})')
        self.name, self.type, self.type_id, self.shape, self.offset = name, tname, type_id, tuple(int(d) for d in shape), offset
        self.n_values, self.n_bytes, self._f = n, n // bs * ts, f

    def data(self):
        return np.frombuffer(self._f._mm, dtype=np.uint8, count=self.n_bytes, offset=self.offset)

    def pread(self, a, b):
        """bytes a .. b of the tensor as a uint8 array in ordinary memory (os.preadv, not the mapping): the source for
        a copy to a card from a file that does not fit in free RAM. A card copying straight from the mapping's pages
        stalls once the page cache is full (measured: a 32.5 GB file on a 29 GiB host, no progress after 23 GB)"""
        if not 0 <= a <= b <= self.n_bytes: raise GGUFError(f'{self.name}: bytes {a} .. {b} outside its {self.n_bytes}')
        out = bytearray(b - a); mv = memoryview(out); n = 0
        while n < b - a:
            k = os.preadv(self._f._fd.fileno(), [mv[n:n + (1 << 30)]], self.offset + a + n)
            if k <= 0: raise GGUFError(f'{self.name}: short read at byte {a + n}')
            n += k
        return np.frombuffer(out, dtype=np.uint8)

    def rows(self):
        """(rows, bytes per row) of a matrix: rows = the product of every dim but the innermost"""
        _, bs, ts = TYPES[self.type_id]
        return self.n_values // self.shape[0], self.shape[0] // bs * ts

    def __repr__(self):
        return f'Tensor({self.name}, {self.type}, {list(self.shape)}, {self.n_bytes} B)'


class GGUFFile:
    """GGUFFile(path): .version, .meta (key -> value: int / float / bool / str / list / numpy array for numeric arrays),
    .tensors (in file order), .T (name -> Tensor), .table_start, .data_start, .alignment; the file stays memory-mapped
    (close())"""

    def __init__(self, path):
        self.path = os.fspath(path)
        self._fd = open(self.path, 'rb')
        size = os.fstat(self._fd.fileno()).st_size
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self._p = 0
        if self._read(4) != b'GGUF': raise GGUFError(f'{self.path}: not a GGUF file (magic)')
        self.version = self._u('<I')
        if self.version not in (2, 3): raise GGUFError(f'{self.path}: GGUF version {self.version} (2 and 3 are read)')
        nt, nkv = self._u('<Q'), self._u('<Q')
        if nt > 1 << 20 or nkv > 1 << 20: raise GGUFError(f'{self.path}: implausible counts ({nt} tensors, {nkv} keys)')
        self.meta = {}
        for _ in range(nkv):
            k = self._str()
            self.meta[k] = self._value(self._u('<I'))
        self.table_start = self._p   # (the tensor table's first byte: the header and the metadata end here)
        info = []
        for _ in range(nt):
            name = self._str(); nd = self._u('<I')
            if nd > 8: raise GGUFError(f'{name}: {nd} dims')
            dims = [self._u('<Q') for _ in range(nd)]
            info.append((name, dims, self._u('<I'), self._u('<Q')))
        self.alignment = int(self.meta.get('general.alignment', 32))
        if self.alignment < 1 or self.alignment & (self.alignment - 1): raise GGUFError(f'alignment {self.alignment}')
        self.data_start = (self._p + self.alignment - 1) // self.alignment * self.alignment
        self.tensors = []
        for name, dims, tid, off in info:
            t = Tensor(self, name, tid, dims, self.data_start + off)
            if off % self.alignment or t.offset + t.n_bytes > size:
                raise GGUFError(f'{name}: data at {t.offset} + {t.n_bytes} B outside the file ({size} B) or misaligned')
            self.tensors.append(t)
        self.T = {t.name: t for t in self.tensors}
        if len(self.T) != len(self.tensors): raise GGUFError(f'{self.path}: duplicate tensor names')

    # -- the parser -------------------------------------------------------------------------------------------------------
    def _read(self, n):
        if self._p + n > len(self._mm): raise GGUFError(f'{self.path}: truncated header at byte {self._p}')
        b = self._mm[self._p:self._p + n]; self._p += n
        return b

    def _u(self, fmt):
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]

    def _str(self):
        n = self._u('<Q')
        if n > 1 << 24: raise GGUFError(f'{self.path}: string of {n} bytes')
        return self._read(n).decode('utf-8')

    def _value(self, vt):
        if vt in _SCALAR: return self._u(_SCALAR[vt])
        if vt == 8: return self._str()
        if vt == 9:
            et, n = self._u('<I'), self._u('<Q')
            if et in _NP:
                a = np.frombuffer(self._read(n * np.dtype(_NP[et]).itemsize), dtype=_NP[et]).copy()
                return a
            if et == 8: return [self._str() for _ in range(n)]
            raise GGUFError(f'{self.path}: array of type {et}')
        raise GGUFError(f'{self.path}: value type {vt}')

    # -- conveniences -----------------------------------------------------------------------------------------------------
    @property
    def arch(self):
        return self.meta.get('general.architecture')

    def hp(self, key, default=None):
        """an architecture hyperparameter: meta['<arch>.<key>']"""
        return self.meta.get(f'{self.arch}.{key}', default)

    def close(self):
        """unmap the file; while tensor views still exist the mapping stays until they are gone (BufferError otherwise)"""
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                pass
            self._fd.close(); self._mm = None
