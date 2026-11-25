"""keylime: extracts every shader the game ships from shaders.rpf.

RPF9 packs encrypt their table of contents, but each shader in shaders.rpf is a plain record:
  u8 len | name[len-1] | 0 | u8 flag | u32 csize | u32 ? | u32 csize | u32 usize | raw deflate[csize]
(the layout XCL documented). Each record inflates to the container the game passes to D3D12, so a
file named by its FNV-1a 64 hash is the same shader the runtime keys its caches by.

  python tools/extract_shaders.py <shaders.rpf> <out dir>
"""
import mmap, os, re, struct, sys, zlib

def fnv(b):
    h = 0xcbf29ce484222325
    for c in b:
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def main(src, out):
    os.makedirs(out, exist_ok=True)
    f = open(src, 'rb')
    d = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    name_ok = re.compile(rb'[A-Za-z_][A-Za-z0-9_:]*\Z')
    i, n, names = 0, 0, []
    end = len(d) - 32
    while i < end:
        ln = d[i]
        p = i + ln
        if ln >= 4 and p + 18 <= len(d) and d[p] == 0 and name_ok.match(d[i + 1:p]):
            c1, c2, u = struct.unpack_from('<I', d, p + 2)[0], struct.unpack_from('<I', d, p + 10)[0], struct.unpack_from('<I', d, p + 14)[0]
            if c1 == c2 and 0 < c1 < 16 << 20 and 0 < u < 64 << 20 and p + 18 + c1 <= len(d):
                try:
                    blob = zlib.decompress(d[p + 18:p + 18 + c1], -15)
                except zlib.error:
                    blob = None
                if blob and len(blob) == u:
                    h = fnv(blob)
                    path = os.path.join(out, '%016x.bin' % h)
                    if not os.path.exists(path):
                        with open(path, 'wb') as o:
                            o.write(blob)
                    names.append('%016x %s' % (h, d[i + 1:p].decode('ascii', 'replace')))
                    n += 1
                    i = p + 18 + c1
                    continue
        i += 1
    with open(os.path.join(out, 'names.txt'), 'w') as o:
        o.write('\n'.join(names) + '\n')
    print('%d shader records extracted to %s' % (n, out))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
