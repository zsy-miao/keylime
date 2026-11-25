"""keylime: extracts shader containers (DXBC) embedded in the game's executable and DLLs.

Some shaders the game runs are not in shaders.rpf (the AMD upscaler, the BVH builder and a few in
the executable). Each is a DXBC container: 'DXBC', a 16-byte digest, u32 1, u32 total size. Files
are named by their FNV-1a 64 hash, as tools/extract_shaders.py names them.

  python tools/extract_embedded_shaders.py <game dir> <out dir>
"""
import mmap, os, struct, sys

def fnv(b):
    h = 0xcbf29ce484222325
    for c in b:
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def main(game, out):
    os.makedirs(out, exist_ok=True)
    total = 0
    for name in sorted(os.listdir(game)):
        if not name.lower().endswith(('.exe', '.dll')):
            continue
        with open(os.path.join(game, name), 'rb') as f:
            d = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n, i = 0, d.find(b'DXBC')
            while i >= 0:
                if i + 32 <= len(d):
                    one, size = struct.unpack_from('<II', d, i + 20)
                    if one == 1 and 32 < size < 64 << 20 and i + size <= len(d):
                        blob = d[i:i + size]
                        path = os.path.join(out, '%016x.bin' % fnv(blob))
                        if not os.path.exists(path):
                            open(path, 'wb').write(blob)
                            n += 1
                        i = d.find(b'DXBC', i + size)
                        continue
                i = d.find(b'DXBC', i + 4)
        if n:
            print('%-40s %d shaders' % (name, n))
        total += n
    print('%d new shader containers in %s' % (total, out))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
