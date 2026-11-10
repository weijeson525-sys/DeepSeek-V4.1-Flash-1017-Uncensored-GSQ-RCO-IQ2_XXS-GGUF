#!/usr/bin/env python3
"""Convert the ISTA-DASLab DeepSeek-V4.1-Flash-1017 GSQ-RCO GGUFs to llama.cpp's own deepseek41 layout.

The files published before October 27, 2026 were written for ISTA's llama.cpp branch. llama.cpp's DeepSeek-V4.1
support (ggml-org/llama.cpp#31086, merged October 25, 2026) reads the layout of its converter
(conversion/deepseek41.py). This script writes that layout from the old files:

  - metadata under the upstream key names; the Engram hash constants (token map, bucket counts, multipliers) move
    from tensors into metadata, with absolute row offsets into the concatenated table
  - the Engram tables of all Engram layers become one tensor, per_layer_token_embd. They stay in the shared
    Engram file, which is converted in place: only its 416-byte header is rewritten, the 110 GB of table data stay
    where they are. llama.cpp opens it as shard 6 of each build, through a link <build>-00006-of-00006.gguf
  - blk.N.engram_wkv is renamed blk.N.engram_kv; blk.N.engram_q and blk.N.engram_k are folded into the F32
    blk.N.engram_gate = q * k (exact: a product of two BF16 values fits in F32)
  - the vision router bias blk.N.exp_probs_b_vl is added (the repository's exp_probs_b_vl.f32, F32 from the
    original checkpoint); llama.cpp routes image tokens with it
  - blk.N.attn_output_a as the 2D matrix of the checkpoint (it was stored as [4096, 1024, 8]; same bytes)
  - tensors in converter order, split like llama-gguf-split --split-max-size 40G --no-tensor-first-split: shard 1
    holds only metadata, so later metadata fixes replace a 6 MB file
  - the chat template of the repository's chat_template.jinja, so downloads that were never patched get the
    current one too
  - mmproj: projector type deepseek41v (V4.1 image token layout) instead of deepseek4v

Quantized weights are copied byte for byte. Standard library only (Python 3.8+). The model is not downloaded again;
the script only fetches chat_template.jinja and exp_probs_b_vl.f32 (80 KB) from the repository, unless their current
versions (checked by SHA-256) lie next to it or in the model folder (then it runs offline).

    python3 convert_ds41_upstream.py MODEL_FOLDER [--out NEW_FOLDER]

MODEL_FOLDER holds the old shards (*-0000N-of-00004.gguf), DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf and optionally the
mmproj. Every build found there is converted: its new shards 1 to 5 are written next to the old ones (or to --out)
and need as much space as its old shards (121 to 155 GB). With --delete-old, the old shards of a build are deleted as
soon as its new ones are verified, so free space for one build suffices. On a folder of new builds the script only
adds missing links to the Engram file.
"""
import argparse
import hashlib
import os
import re
import struct
import sys
import urllib.request

REPO = 'ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF'  # chat_template.jinja and exp_probs_b_vl.f32 come from here
ARCH = 'deepseek41'
TABLE = 'per_layer_token_embd.weight'
ENGRAM_FILE = 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
N_SPLIT = 6  # shards per build; the last is the shared Engram file
SPLIT_MAX_SIZE = 40 * 1000 ** 3  # llama-gguf-split --split-max-size 40G
ALIGNMENT = 32
SPLIT_RE = re.compile(r'^(?P<prefix>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$')
COPY_CHUNK = 64 << 20

# ---------------------------------------------------------------------------------------------------------------
# GGUF header reading and writing

GGUF_MAGIC = 0x46554747
T_U16, T_U32, T_I32, T_STRING, T_ARRAY, T_U64 = 2, 4, 5, 8, 9, 10
SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
GGML_F32, GGML_IQ4_NL, GGML_I32, GGML_I64, GGML_BF16 = 0, 20, 26, 27, 30
# GGML type -> (elements per block, bytes per block), for the types these files contain
GGML_BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
              14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
              21: (256, 110), 22: (256, 82), 23: (256, 136), 26: (1, 4), 27: (1, 8), 29: (256, 56), 30: (1, 2),
              39: (32, 17)}


def align(n, a=ALIGNMENT):
    return (n + a - 1) // a * a



# --- ABIRAY PATCH: Ensure all GGML quant types (including Q6_K = 14) are registered ---
GGML_BLOCK.update({
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K (Used by Abiray's abliterated blk.0..39.attn_output_b.weight!)
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
})

def tensor_nbytes(dims, ggml_type):
    per_block, block_bytes = GGML_BLOCK[ggml_type]
    n = 1
    for d in dims:
        n *= d
    if dims[0] % per_block:
        raise ValueError('row of %d elements is not a whole number of type %d blocks' % (dims[0], ggml_type))
    return n // per_block * block_bytes


class _Reader:
    def __init__(self, buf):
        self.buf, self.p = buf, 0

    def take(self, n):
        if self.p + n > len(self.buf):
            raise EOFError
        b = self.buf[self.p:self.p + n]
        self.p += n
        return b

    def u(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.u('<Q')).decode('utf-8')

    def skip_value(self, t):
        if t == T_STRING:
            self.take(self.u('<Q'))
        elif t == T_ARRAY:
            at, n = self.u('<I'), self.u('<Q')
            if at in (T_STRING, T_ARRAY):
                for _ in range(n):
                    self.skip_value(at)
            else:
                self.take(n * struct.calcsize(SCALAR[at]))
        else:
            self.take(struct.calcsize(SCALAR[t]))


def decode_value(t, raw):
    r = _Reader(raw)
    if t == T_STRING:
        return r.string()
    if t == T_ARRAY:
        at, n = r.u('<I'), r.u('<Q')
        if at == T_STRING:
            return [r.string() for _ in range(n)]
        fmt = SCALAR[at]
        return list(struct.unpack('<%d%s' % (n, fmt[1]), r.take(n * struct.calcsize(fmt))))
    return struct.unpack(SCALAR[t], raw)[0]


def encode_value(t, value, array_type=None):
    if t == T_STRING:
        b = value.encode('utf-8')
        return struct.pack('<Q', len(b)) + b
    if t == T_ARRAY:
        head = struct.pack('<IQ', array_type, len(value))
        if array_type == T_STRING:
            return head + b''.join(encode_value(T_STRING, v) for v in value)
        return head + struct.pack('<%d%s' % (len(value), SCALAR[array_type][1]), *value)
    return struct.pack(SCALAR[t], value)


class GGUFFile:
    """Header of a GGUF file: kv is a list of [key, type, raw value bytes], tensors maps a name to
    (absolute file offset, dims, ggml type, byte size)."""

    def __init__(self, path):
        self.path = path
        size = 1 << 20
        while True:
            with open(path, 'rb') as f:
                buf = f.read(size)
            try:
                self._parse(buf)
                break
            except EOFError:
                if len(buf) < size:
                    raise ValueError('%s: truncated GGUF header' % path)
                size *= 4
        self.size = os.path.getsize(path)

    def _parse(self, buf):
        r = _Reader(buf)
        if r.u('<I') != GGUF_MAGIC:
            raise ValueError('%s is not a GGUF file' % self.path)
        self.version = r.u('<I')
        n_tensors, n_kv = r.u('<Q'), r.u('<Q')
        self.kv = []
        for _ in range(n_kv):
            key = r.string()
            t = r.u('<I')
            start = r.p
            r.skip_value(t)
            self.kv.append([key, t, buf[start:r.p]])
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u('<Q') for _ in range(r.u('<I'))]
            infos.append((name, dims, r.u('<I'), r.u('<Q')))
        self.header_size = r.p
        self.data_start = align(r.p, self.get('general.alignment', ALIGNMENT))
        self.tensors = {}
        self.order = []
        for name, dims, gt, off in infos:
            self.tensors[name] = (self.data_start + off, dims, gt, tensor_nbytes(dims, gt))
            self.order.append(name)

    def get(self, key, default=None):
        for k, t, raw in self.kv:
            if k == key:
                return decode_value(t, raw)
        return default

    def read_tensor(self, name):
        off, _, _, n = self.tensors[name]
        with open(self.path, 'rb') as f:
            f.seek(off)
            data = f.read(n)
        if len(data) != n:
            raise ValueError('%s: tensor %s is truncated' % (self.path, name))
        return data


def serialize_header(kv, infos):
    """kv: [key, type, raw]; infos: (name, dims, ggml type, relative offset). Padded to the data section."""
    out = [struct.pack('<IIQQ', GGUF_MAGIC, 3, len(infos), len(kv))]
    for key, t, raw in kv:
        out += [encode_value(T_STRING, key), struct.pack('<I', t), raw]
    for name, dims, gt, off in infos:
        out += [encode_value(T_STRING, name), struct.pack('<I', len(dims)),
                struct.pack('<%dQ' % len(dims), *dims), struct.pack('<IQ', gt, off)]
    header = b''.join(out)
    return header + b'\0' * (align(len(header)) - len(header))


# ---------------------------------------------------------------------------------------------------------------
# conversion


class Tensor:
    """An output tensor: bytes from a file range or computed here."""

    def __init__(self, name, dims, ggml_type, path=None, offset=None, data=None):
        self.name, self.dims, self.type = name, list(dims), ggml_type
        self.nbytes = tensor_nbytes(self.dims, ggml_type)
        self.path, self.offset, self.data = path, offset, data
        if data is not None and len(data) != self.nbytes:
            raise ValueError('%s: %d bytes for a tensor of %d' % (name, len(data), self.nbytes))

    def source_hash(self):
        if self.data is not None:
            return hashlib.sha256(self.data).hexdigest()
        return hash_range(self.path, self.offset, self.nbytes)


def hash_range(path, offset, n):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        f.seek(offset)
        while n:
            b = f.read(min(n, COPY_CHUNK))
            if not b:
                raise ValueError('%s ends early' % path)
            h.update(b)
            n -= len(b)
    return h.hexdigest()


# the old branch-specific metadata, replaced by the upstream keys
OLD_KEYS = ['attention.candidate_block_size', 'attention.candidate_topk_blocks', 'attention.candidate_source_layer_id',
            'kv_source_layer_ids', 'index_source_layer_ids', 'dspark_target_layer_ids', 'expert_scoring_func',
            'engram.present', 'engram.layer_ids', 'engram.num_embeddings', 'engram_head_dim', 'engram_n_heads',
            'engram_max_ngram_size', 'engram_vocab_size', 'engram_compressed_vocab_size', 'engram_pad_token_id',
            'engram.source', 'engram.file']
CONST_TENSORS = ['engram_token_map.weight', 'engram_primes.weight', 'engram_offsets.weight',
                 'engram_multipliers.weight']


def unpack(data, fmt):
    return list(struct.unpack('<%d%s' % (len(data) // struct.calcsize(fmt), fmt[1]), data))


def convert_metadata(meta, tensor_source, chat_template):
    """meta: GGUFFile with the old first-shard metadata. Returns the new kv list (without split keys)."""
    A = ARCH + '.'
    get = lambda k: meta.get(A + k)
    if meta.get('general.architecture') != ARCH:
        raise ValueError('%s: architecture %r, expected %s' % (meta.path, meta.get('general.architecture'), ARCH))
    missing = [A + k for k in OLD_KEYS if meta.get(A + k) is None and k not in ('dspark_target_layer_ids',
                                                                              'expert_scoring_func')]
    if missing:
        raise ValueError('%s: missing metadata %s' % (meta.path, ', '.join(missing)))

    layers = get('engram.layer_ids')
    n_gram, n_heads, head_dim = get('engram_max_ngram_size'), get('engram_n_heads'), get('engram_head_dim')
    n_cols = (n_gram - 1) * n_heads
    token_map = unpack(tensor_source('engram_token_map.weight'), '<i')
    primes = unpack(tensor_source('engram_primes.weight'), '<q')
    local_offsets = unpack(tensor_source('engram_offsets.weight'), '<q')
    multipliers = unpack(tensor_source('engram_multipliers.weight'), '<q')
    if len(primes) != len(layers) * n_cols or len(local_offsets) != len(primes) or \
            len(multipliers) != len(layers) * n_gram:
        raise ValueError('Engram constants do not match %d layers x %d heads' % (len(layers), n_cols))
    for e, rows in enumerate(get('engram.num_embeddings')):
        layer_primes = primes[e * n_cols:(e + 1) * n_cols]
        if sum(layer_primes) != rows:
            raise ValueError('Engram layer %d: bucket counts do not add up to its table rows' % layers[e])
        if local_offsets[e * n_cols:(e + 1) * n_cols] != [sum(layer_primes[:i]) for i in range(n_cols)]:
            raise ValueError('Engram layer %d: unexpected row offsets' % layers[e])
    if max(token_map) + 1 != get('engram_compressed_vocab_size') or \
            len(token_map) != len(meta.get('tokenizer.ggml.tokens')):
        raise ValueError('Engram token map does not match the vocabulary')
    if any(m <= 0 or m % 2 == 0 for m in multipliers):
        raise ValueError('Engram multipliers must be odd and positive')

    new = [
        ('attention.kv_source_layers', T_ARRAY, get('kv_source_layer_ids'), T_U32),
        ('attention.indexer.source_layers', T_ARRAY, get('index_source_layer_ids'), T_U32),
        ('attention.indexer.candidate_source_layer', T_I32, get('attention.candidate_source_layer_id'), None),
        ('attention.indexer.candidate_top_k_blocks', T_U32, get('attention.candidate_topk_blocks'), None),
        ('attention.indexer.candidate_block_size', T_U32, get('attention.candidate_block_size'), None),
        ('ple.layers', T_ARRAY, layers, T_U32),
        ('ple.ngram_size', T_U32, n_gram, None),
        ('ple.heads_per_ngram', T_U32, n_heads, None),
        ('embedding_length_per_layer_input', T_U32, head_dim, None),
        ('ple.layer_multipliers', T_ARRAY, multipliers, T_U64),
        ('ple.head_offsets', T_ARRAY, [sum(primes[:i]) for i in range(len(primes))], T_U64),
        ('ple.head_vocab_sizes', T_ARRAY, primes, T_U64),
        ('engram.pad_token_id', T_U32, get('engram_pad_token_id'), None),
        ('engram.token_map', T_ARRAY, token_map, T_I32),
    ]
    old = {A + k for k in OLD_KEYS}
    kv, inserted = [], False
    for key, t, raw in meta.kv:
        if key.startswith('split.'):
            continue
        if key in old:
            if not inserted:
                kv += [[A + k, vt, encode_value(vt, v, at)] for k, vt, v, at in new]
                inserted = True
            continue
        if key == 'tokenizer.chat_template':
            raw = encode_value(T_STRING, chat_template)  # the repository's, whether or not the old one was patched
        kv.append([key, t, raw])
    return kv


def bf16_to_f32(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]


def engram_gate(q, k):
    """F32 q * k from two BF16 tensors (the upstream converter's data_torch.float() * k().float())."""
    n = len(q) // 2
    qv, kv = struct.unpack('<%dH' % n, q), struct.unpack('<%dH' % n, k)
    return struct.pack('<%df' % n, *[bf16_to_f32(a) * bf16_to_f32(b) for a, b in zip(qv, kv)])


def canonical_key(name):
    """Converter order: global tensors, then the blocks in order, the Engram table last; names sorted."""
    if name == 'per_layer_token_embd.weight':
        return (2, 0, name)
    m = re.match(r'blk\.(\d+)\.', name)
    return (1, int(m.group(1)), name) if m else (0, 0, name)


def plan_tensors(meta, sources, engram, bias_vl):
    """sources: GGUFFiles holding the old backbone tensors; engram: the old Engram GGUFFile."""
    index = {}
    for src in sources:
        for name, (off, dims, gt, _) in src.tensors.items():
            if name in index:
                raise ValueError('tensor %s found in %s and %s' % (name, index[name][0].path, src.path))
            index[name] = (src, off, dims, gt)
    A = ARCH + '.'
    n_layer, n_expert = meta.get(A + 'block_count'), meta.get(A + 'expert_count')
    layers = meta.get(A + 'engram.layer_ids')
    out = []
    for name, (src, off, dims, gt) in index.items():
        if name in CONST_TENSORS or re.match(r'blk\.\d+\.engram_[qk]\.weight$', name):
            continue
        if name.endswith('.attn_output_a.weight') and len(dims) == 3:
            dims = [dims[0], dims[1] * dims[2]]  # the checkpoint's 2D wo_a, as the converter writes it; same bytes
        out.append(Tensor(name.replace('.engram_wkv.', '.engram_kv.'), dims, gt, src.path, off))

    for layer in layers:
        q, k = ('blk.%d.engram_%s.weight' % (layer, w) for w in 'qk')
        if index[q][3] != GGML_BF16 or index[k][3] != GGML_BF16 or index[q][2] != index[k][2]:
            raise ValueError('blk.%d engram_q / engram_k must be BF16 of the same shape' % layer)
        out.append(Tensor('blk.%d.engram_gate.weight' % layer, index[q][2], GGML_F32,
                          data=engram_gate(index[q][0].read_tensor(q), index[k][0].read_tensor(k))))

    if len(bias_vl) != n_layer * n_expert * 4:
        raise ValueError('exp_probs_b_vl.f32 is for a different model')
    for layer in range(n_layer):
        if 'blk.%d.exp_probs_b.bias' % layer not in index:
            raise ValueError('blk.%d.exp_probs_b.bias is missing' % layer)
        row = bias_vl[layer * n_expert * 4:(layer + 1) * n_expert * 4]
        out.append(Tensor('blk.%d.exp_probs_b_vl.bias' % layer, [n_expert], GGML_F32, data=row))

    table, table_layers = engram_table(engram)
    if table.dims[1] != sum(meta.get(A + 'engram.num_embeddings')) or table_layers not in (None, layers) or \
            meta.get(A + 'engram.source') != engram.get(A + 'engram.source'):
        raise ValueError('%s does not belong to this model (other tables or checkpoint revision)' % engram.path)
    out.append(table)
    out.sort(key=lambda t: canonical_key(t.name))
    return out


def engram_table(engram):
    """The Engram table as one tensor, and the layers of the old per-layer tables (None if already converted).
    The old file holds the per-layer tables back to back, so their concatenation is one byte range."""
    if TABLE in engram.tensors:
        off, dims, gt, _ = engram.tensors[TABLE]
        return Tensor(TABLE, dims, gt, engram.path, off), None
    names = sorted((n for n in engram.tensors if re.match(r'blk\.\d+\.engram_embd\.weight$', n)),
                   key=lambda n: engram.tensors[n][0])
    if not names or len(names) != len(engram.tensors):
        raise ValueError('%s: unexpected tensors in the Engram file' % engram.path)
    first_off, dims, gt, _ = engram.tensors[names[0]]
    end, rows = first_off, 0
    for name in names:
        off, d, g, n = engram.tensors[name]
        if off != end or g != gt or d[0] != dims[0]:
            raise ValueError('%s: the Engram tables are not contiguous' % engram.path)
        end, rows = off + n, rows + d[1]
    layers = [int(re.match(r'blk\.(\d+)\.', n).group(1)) for n in names]
    return Tensor(TABLE, [dims[0], rows], gt, engram.path, first_off), layers


def plan_splits(tensors, max_size):
    """llama-gguf-split --split-max-size with --no-tensor-first-split: the first file holds only metadata."""
    splits, cur = [[], []], 0
    for t in tensors:
        n = align(t.nbytes)
        if cur + n > max_size:
            if not splits[-1]:
                raise ValueError('tensor %s is larger than the split size and comes first' % t.name)
            splits.append([])
            cur = n
        else:
            cur += n
        splits[-1].append(t)
    return splits


def split_kv(no, count, n_tensors):
    return [['split.no', T_U16, encode_value(T_U16, no)], ['split.count', T_U16, encode_value(T_U16, count)],
            ['split.tensors.count', T_I32, encode_value(T_I32, n_tensors)]]


def copy_range(src_path, offset, n, fout, buf):
    """Plain reads and writes through buf, as on every file system (os.copy_file_range leaves the copy to the
    file system's own implementation)."""
    with open(src_path, 'rb') as fin:
        fin.seek(offset)
        while n:
            got = fin.readinto(buf[:min(n, len(buf))])
            if not got:
                raise ValueError('%s ends early' % src_path)
            fout.write(buf[:got])
            n -= got


def write_gguf(path, kv, tensors):
    infos, off = [], 0
    for t in tensors:
        infos.append((t.name, t.dims, t.type, off))
        off += align(t.nbytes)
    tmp = path + '.partial'
    buf = memoryview(bytearray(COPY_CHUNK))
    with open(tmp, 'wb') as f:
        f.write(serialize_header(kv, infos))
        for t in tensors:
            if t.data is not None:
                f.write(t.data)
            else:
                copy_range(t.path, t.offset, t.nbytes, f, buf)
            f.write(b'\0' * (align(t.nbytes) - t.nbytes))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_gguf(path, kv, tensors):
    """Re-read the written file: same metadata, tensor table and tensor bytes as planned."""
    g = GGUFFile(path)
    if g.kv != kv or g.order != [t.name for t in tensors]:
        raise ValueError('%s: header differs from the plan' % path)
    for t in tensors:
        off, dims, gt, n = g.tensors[t.name]
        if dims != t.dims or gt != t.type or hash_range(path, off, n) != t.source_hash():
            raise ValueError('%s: tensor %s differs from its source%s. The copy or a read failed on this machine; '
                             'a damaged download would have been copied as it is'
                             % (path, t.name, '' if t.path is None else ' (%s at %d)' % (t.path, t.offset)))
    if g.size != g.data_start + sum(align(t.nbytes) for t in tensors):
        raise ValueError('%s: unexpected file size' % path)


def plan_model(meta, sources, engram, chat_template, bias_vl):
    """Metadata, all tensors (the Engram table last) and the backbone split of one build."""
    kv = convert_metadata(meta, lambda t: next(s for s in sources if t in s.tensors).read_tensor(t), chat_template)
    tensors = plan_tensors(meta, sources, engram, bias_vl)
    splits = plan_splits(tensors[:-1], SPLIT_MAX_SIZE)
    if tensors[-1].name != TABLE or len(splits) != N_SPLIT - 1:
        raise ValueError('the backbone needs %d shards; the shared Engram file is shard %d of %d'
                         % (len(splits), N_SPLIT, N_SPLIT))
    return kv, tensors, splits


def convert_model(meta, sources, engram, chat_template, bias_vl, name, out_dir, verify, log):
    """Writes shards 1 to N_SPLIT - 1 of a build; shard N_SPLIT is the shared Engram file."""
    kv, tensors, splits = plan_model(meta, sources, engram, chat_template, bias_vl)
    paths = []
    for i, part in enumerate(splits):
        path = os.path.join(out_dir, '%s-%05d-of-%05d.gguf' % (name, i + 1, N_SPLIT))
        part_kv = (kv if i == 0 else []) + split_kv(i, N_SPLIT, len(tensors))
        log('  %s: %d tensors, %.2f GB' % (os.path.basename(path), len(part), sum(t.nbytes for t in part) / 1e9))
        write_gguf(path, part_kv, part)
        if verify:
            verify_gguf(path, part_kv, part)
        paths.append(path)
    return paths, tensors


def engram_header(table, source, n_tensors):
    kv = [['general.name', T_STRING, encode_value(T_STRING, 'DeepSeek-V4.1-Flash-1017 Engram tables')],
          ['general.description', T_STRING, encode_value(T_STRING, 'Shard 6 of 6 of every GSQ-RCO build')],
          [ARCH + '.engram.source', T_STRING, encode_value(T_STRING, source)]]
    kv += split_kv(N_SPLIT - 1, N_SPLIT, n_tensors)
    return kv, serialize_header(kv, [(TABLE, table.dims, table.type, 0)])


def convert_engram(engram, table, n_tensors, log):
    """Rewrite the old Engram file's header in place. The new header ends where the old one did, so the 110 GB of
    table data stay where they are; the old header is kept in a side file until the new one is checked."""
    path = engram.path
    backup = path + '.old-header'
    kv, header = engram_header(table, engram.get(ARCH + '.engram.source'), n_tensors)
    if TABLE in engram.tensors:
        if engram.kv != kv or engram.data_start != len(header):
            raise ValueError('%s: unexpected header' % path)
        if os.path.exists(backup):
            os.remove(backup)
        return False
    if len(header) != engram.data_start or table.offset != engram.data_start:
        raise ValueError('%s: the new header does not fit the old one' % path)
    with open(path, 'rb') as f:
        old = f.read(engram.data_start)
    with open(backup, 'wb') as f:
        f.write(old)
        os.fsync(f.fileno())
    with open(path, 'r+b') as f:
        f.write(header)
        f.flush()
        os.fsync(f.fileno())
    g = GGUFFile(path)
    if g.kv != kv or g.order != [TABLE] or g.tensors[TABLE][:3] != (table.offset, table.dims, table.type) or \
            g.size != engram.size:
        with open(path, 'r+b') as f:
            f.write(old)
        raise ValueError('%s: the rewritten header did not check out; the old one was restored' % path)
    os.remove(backup)
    log('%s: header rewritten in place (%d bytes), the table data are unchanged' % (os.path.basename(path), len(old)))
    return True


def link_engram(engram_path, name, out_dir, log):
    """llama.cpp opens <name>-0000N-of-0000N.gguf as the last shard: a link to the shared Engram file."""
    path = os.path.join(out_dir, '%s-%05d-of-%05d.gguf' % (name, N_SPLIT, N_SPLIT))
    if os.path.lexists(path):
        if os.path.exists(path) and os.path.samefile(path, engram_path):
            return
        raise ValueError('%s exists and is not the Engram file' % path)
    try:
        os.symlink(os.path.relpath(engram_path, out_dir), path)
        kind = 'link'
    except (OSError, NotImplementedError):
        os.link(engram_path, path)  # e.g. Windows without symlink rights
        kind = 'hard link'
    log('  %s: %s to %s' % (os.path.basename(path), kind, os.path.basename(engram_path)))


def convert_mmproj(src, out_path, log):
    if src.get('clip.projector_type') != 'deepseek4v':
        raise ValueError('%s: projector type %r' % (src.path, src.get('clip.projector_type')))
    kv = []
    for key, t, raw in src.kv:
        if key == 'clip.projector_type':
            raw = encode_value(T_STRING, 'deepseek41v')
        elif key == 'clip.vision.image_max_pixels':
            continue  # deepseek41v fixes the image budget at 1024 tokens; the upstream converter writes no key
        kv.append([key, t, raw])
    tensors = [Tensor(n, src.tensors[n][1], src.tensors[n][2], src.path, src.tensors[n][0]) for n in src.order]
    log('  %s: projector type deepseek41v' % os.path.basename(out_path))
    write_gguf(out_path, kv, tensors)
    verify_gguf(out_path, kv, tensors)


# ---------------------------------------------------------------------------------------------------------------
# discovery


def find_sets(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith('.gguf')]
        else:
            files.append(p)
    models, shards, engrams, mmprojs, converted = [], {}, {}, [], []
    for path in files:
        g = GGUFFile(path)
        arch = g.get('general.architecture')
        m = SPLIT_RE.match(os.path.basename(path))
        if arch == ARCH:
            if g.get(ARCH + '.attention.kv_source_layers') is not None:
                if m:  # already in the upstream layout; it may still lack the link to the Engram file
                    converted.append((os.path.dirname(os.path.abspath(path)), m.group('prefix'), g))
                continue
            models.append(g)
        elif arch == ARCH + '-engram' or TABLE in g.tensors:
            engrams.setdefault(os.path.realpath(path), g)  # links to the Engram file count once
        elif g.get('clip.projector_type') == 'deepseek4v':
            mmprojs.append(g)
        elif m and g.get('split.no') is not None:
            shards[(os.path.dirname(path), m.group('prefix'), int(m.group('no')))] = g
    sets = []
    for g in models:
        m = SPLIT_RE.match(os.path.basename(g.path))
        if m is None:
            sets.append((g, [g], os.path.basename(g.path)[:-5]))
            continue
        prefix, count = m.group('prefix'), g.get('split.count')
        parts = [g] + [shards.get((os.path.dirname(g.path), prefix, i)) for i in range(2, count + 1)]
        if None in parts:
            raise ValueError('%s: not all %d shards are present' % (g.path, count))
        sets.append((g, parts, prefix))
    return sets, list(engrams.values()), mmprojs, converted


def repo_file(name, path, inputs, sha256, log):
    """A small file of the repository with the given SHA-256: the given path, else a copy next to this script or in
    the model folder (other versions, such as an older download, are skipped), else downloaded from the repository."""
    if path is not None:
        candidates = [path]
    else:
        folders = [os.path.dirname(os.path.abspath(__file__))]
        folders += [p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p)) for p in inputs]
        candidates = list(dict.fromkeys(f for f in (os.path.join(d, name) for d in folders) if os.path.isfile(f)))
    for f in candidates:
        with open(f, 'rb') as fin:
            data = fin.read()
        if hashlib.sha256(data).hexdigest() == sha256:
            return data
        if path is not None:
            raise ValueError('%s is not the current %s of %s (checksum)' % (path, name, REPO))
        log('%s: not the current version (checksum), skipped' % f)
    url = 'https://huggingface.co/%s/resolve/main/%s' % (REPO, name)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except OSError as e:
        raise ValueError('%s could not be downloaded from %s (%s); put it next to this script' % (name, url, e))
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError('%s of %s has changed since this script was written; download the current '
                         'convert_ds41_upstream.py' % (name, REPO))
    log('%s: downloaded from %s' % (name, REPO))
    return data


def read_inputs(args, log):
    """chat_template.jinja (embedded in the new first shards) and exp_probs_b_vl.f32 (the vision router bias)."""
    template = repo_file('chat_template.jinja', args.chat_template, args.inputs, TEMPLATE_SHA256, log).decode('utf-8')
    bias_vl = repo_file('exp_probs_b_vl.f32', args.vision_bias, args.inputs, BIAS_VL_SHA256, log)
    return template, bias_vl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('inputs', nargs='+', help='model folder(s) or GGUF files')
    ap.add_argument('--out', help='folder for the new shards (default: next to the old ones)')
    ap.add_argument('--engram', help='the Engram GGUF (default: the one found among the inputs)')
    ap.add_argument('--chat-template', help='chat_template.jinja of the repository (default: next to this script or '
                    'in the model folder)')
    ap.add_argument('--vision-bias', help='exp_probs_b_vl.f32 of the repository (default: as --chat-template)')
    ap.add_argument('--metadata', help=argparse.SUPPRESS)  # header of another first shard (release builds)
    ap.add_argument('--name', help='output name prefix for a single unsplit input')
    ap.add_argument('--no-verify', action='store_true', help='skip re-reading and hashing the written shards')
    ap.add_argument('--delete-old', action='store_true', help='delete the old shards after a verified conversion')
    ap.add_argument('--dry-run', action='store_true', help='only print what would be done')
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)

    sets, engrams, mmprojs, converted = find_sets(args.inputs)
    if args.engram:
        engrams = [GGUFFile(args.engram)]
    if (sets or converted) and len(engrams) != 1:
        raise SystemExit('expected one Engram file (%s), found %d; pass it with --engram' % (ENGRAM_FILE, len(engrams)))
    if not sets and not mmprojs and not converted:
        raise SystemExit('nothing to convert (no DeepSeek-V4.1 GSQ-RCO files found)')
    engram = engrams[0] if engrams else None
    chat_template = bias_vl = None
    if sets:
        chat_template, bias_vl = read_inputs(args, log)
    written, old = [], []
    builds = [(d, prefix) for d, prefix, _ in converted]
    n_tensors = {g.get('split.tensors.count') for _, _, g in converted}
    for first, parts, prefix in sets:
        meta = GGUFFile(args.metadata) if args.metadata else first
        name = args.name or prefix
        out_dir = os.path.abspath(args.out or os.path.dirname(os.path.abspath(first.path)))
        log('%s -> %s' % (prefix, os.path.join(out_dir, name + '-*.gguf')))
        if args.dry_run:
            _, tensors, splits = plan_model(meta, parts, engram, chat_template, bias_vl)
            for i, part in enumerate(splits):
                log('  shard %d: %d tensors, %.2f GB' % (i + 1, len(part), sum(t.nbytes for t in part) / 1e9))
            log('  shard %d: %s' % (N_SPLIT, os.path.basename(engram.path)))
            n_tensors.add(len(tensors))
            continue
        os.makedirs(out_dir, exist_ok=True)
        paths, tensors = convert_model(meta, parts, engram, chat_template, bias_vl, name, out_dir, not args.no_verify, log)
        written += paths
        builds.append((out_dir, name))
        n_tensors.add(len(tensors))
        if args.delete_old:
            for p in parts:
                os.remove(p.path)
            log('  deleted the %d old shards' % len(parts))
        else:
            old += [p.path for p in parts]

    if engram is not None and (builds or sets):
        if len(n_tensors) != 1:
            raise ValueError('the builds differ in tensor count, so they cannot share one Engram file')
        table, _ = engram_table(engram)
        if args.dry_run:
            if TABLE not in engram.tensors:
                log('%s: header rewritten in place, the table data stay' % os.path.basename(engram.path))
        else:
            convert_engram(engram, table, n_tensors.pop(), log)
            for out_dir, name in builds:
                link_engram(engram.path, name, out_dir, log)
    for src in mmprojs:
        out_dir = args.out or os.path.dirname(os.path.abspath(src.path))
        out_path = os.path.join(out_dir, os.path.basename(src.path))
        if args.dry_run:
            log('%s: projector type deepseek4v -> deepseek41v' % src.path)
            continue
        if os.path.abspath(out_path) == os.path.abspath(src.path):
            convert_mmproj(src, out_path + '.new', log)
            os.replace(out_path + '.new', out_path)
        else:
            convert_mmproj(src, out_path, log)
    if old:
        log('done. llama.cpp no longer needs the old shards:\n  ' + '\n  '.join(old))


# exp_probs_b_vl.f32: blk.0..39.exp_probs_b_vl = layers.N.ffn.gate.bias_vl of deepseek-ai/DeepSeek-V4.1-Flash-1017
# (revision 1bae0e1088993b9b82de08f59de5be5a34e42170), 40 x 384 little-endian F32
BIAS_VL_SHA256 = '710b11267b4db392d1fdb558f36ba8e0e91a91f4e8180c953c912e18eacf843d'
# chat_template.jinja of the repository since October 25, 2026 (commit 1c100b9), as in the published first shards
TEMPLATE_SHA256 = '4d61d22c90ea0cdf15c7e298f7215f23eeffd20f8eed8fb517fda5fb64693d99'

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as e:
        sys.exit('error: %s' % e)
