## curans: streaming compressor with CUDA
![AI Assisted](https://shields.io)

A small experimental, self-contained streaming compressor that runs its entropy coder 
on the GPU. Data is split into 64 KiB chunks, each chunk into 128 interleaved rANS 
streams, and one thread encodes or decodes one stream. A frame (default 16 MiB) shares  
a single static frequency table, so the table costs 512 bytes per frame rather than per chunk. 

### Build

```sh
git clone https://github.com/xxx/curans.git
mkdir build && cd build
cmake ../curans

ctest --output-on-failure   # CPU unit tests
./curans --self-test  # GPU vs CPU

cmake -DCURANS_CUDA_ARCH=sm_86 # architecture (CUDA >= 11.5 or "native")
```

Requires CMake 3.16+ and CUDA 10.1+.
( uses `__match_any_sync` on CUDA 11+ with compute capability )

### Usage

```sh
./curans -c input.bin input.crns
./curans -d input.crns output.bin

./curans -c --cpu input.bin out.crns # force CPU

./curans -c -f 1 input.bin out.crns # smaller frames, lower latency, worse ratio

# -f    sets the frame size (1 - 256 MiB, default 16)
```

### Details

```
frame (<= 16 MiB)
 . histogram kernel     one 256-bin histogram
 . host: normalise      counts -> freq[256], sum(freq) == 4096
 . encode kernel        1 block per chunk, 1 thread per lane
 . host: offsets        exclusive prefix sum of the 128*N
 . gather kernel        pack the fixed-size scratch slots
 . record               header | lens[] | payload
```

Decoding is the mirror image: one block per chunk, each thread decodes its own
stream, so writes are coalesced.

Incompressible data is detected per frame and stored verbatim,
so the output never grows by more than the 524-byte frame header.

### Format (little-endian)

```
StreamHdr  { u32 magic "CRNS"; u32 version; u32 chunk_size;
             u32 lanes; u32 frame_size; }                    # 20 bytes

FrameHdr   { u32 nchunks; u32 raw_len; u32 flags; u16 freq[256]; }  # 524 bytes
           u16 lens[nchunks * lanes]                         
           u8  payload[]                                     

flags bit0 = stored: nchunks == 0 # payload is raw_len plaintext bytes
```

An all-zero terminates the stream. A stream that ends at a frame boundary
without that marker is rejected as truncated, so silent data loss
on a cut transfer is not possible.

### Performance notes

- This is an ORDER-0 coder: ~2.3x on English text, ~4.6x on low-entropy
  data, 1.0x on random. It will not compete with zstd/LZ4.
- For a pipeline that reads from disk and writes to disk over PCIe, a fast CPU
  LZ4/zstd usually wins.


### Inspired by

NVIDIA nvCOMP - high-speed data compression and decompression library optimized for NVIDIA GPUs

### License

This project is open-source and available under the MIT License. Feel free to modify and adapt it.
