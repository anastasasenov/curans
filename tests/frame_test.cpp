/** CPU round-trip test */
#include "../src/frame.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>

static uint32_t rng = 987654321u;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static bool test(const std::vector<uint8_t>& src, const char* name) {
    size_t cap = frame_bound(src.size());
    std::vector<uint8_t> comp(cap), back(src.size() + 64);
    size_t clen = cpu_compress_frame(src.data(), src.size(), comp.data(), cap);
    if (!clen) { printf("%-16s encode failed\n", name); return false; }
    size_t used = cpu_decompress_frame(comp.data(), clen, back.data(), back.size());
    bool ok = used == clen && back.size() >= src.size() &&
              memcmp(back.data(), src.data(), src.size()) == 0;
    printf("%-16s raw=%9zu comp=%9zu  %5.2fx  %s\n", name, src.size(), clen,
           src.empty() ? 0.0 : (double)src.size() / clen, ok ? "OK" : "FAIL");
    return ok;
}

static bool test_assemble(const std::vector<uint8_t>& src, const char* name) {
    size_t cap = frame_bound(src.size());
    std::vector<uint8_t> a(cap), b(cap);
    size_t m = cpu_compress_frame(src.data(), src.size(), a.data(), cap);
    if (!m) { printf("%-16s encode failed\n", name); return false; }
    FrameHdr h;
    memcpy(&h, a.data(), sizeof(h));
    size_t m2;
    if (h.flags & kFlagStored) {
        m2 = frame_assemble(b.data(), cap, src.data(), src.size(), h.freq, nullptr,
                            src.size(), a.data());
    } else {
        const uint16_t* lens = (const uint16_t*)(a.data() + sizeof(h));
        size_t poff = frame_payload_off(h.nchunks);
        m2 = frame_assemble(b.data(), cap, src.data(), src.size(), h.freq, lens,
                            m - poff, a.data() + poff);
    }
    bool ok = m == m2 && memcmp(a.data(), b.data(), m) == 0;
    printf("%-16s n=%8zu  %s\n", name, src.size(), ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    std::vector<uint8_t> v;

    v.clear();
    for (int i = 0; i < 3000000; i++) v.push_back("abcdefghij"[rnd() % 10]);
    ok &= test(v, "low entropy");

    v.resize(2000000);
    for (auto& b : v) b = (uint8_t)rnd();
    ok &= test(v, "random");

    v.resize(1000000);
    for (size_t i = 0; i < v.size(); i++)
        v[i] = (uint8_t)((i / 1000) * 3 + (rnd() % 5));
    ok &= test(v, "structured");

    v.assign(500000, 0);
    ok &= test(v, "all zero");

    for (size_t n : {(size_t)1, (size_t)63, (size_t)65535, (size_t)65536, (size_t)65537,
                     (size_t)131072, (size_t)131073}) {
        v.resize(n);
        for (auto& b : v) b = (uint8_t)(rnd() % 3 + 'x');
        ok &= test(v, "size sweep");
    }

    for (size_t n : {(size_t)1, (size_t)100, (size_t)65536, (size_t)300000}) {
        v.resize(n);
        for (auto& b : v) b = (uint8_t)(rnd() % 7);
        ok &= test_assemble(v, "assemble");
    }

    v.resize(100000); /** @note: trunc */
    for (auto& b : v) b = (uint8_t)rnd();
    std::vector<uint8_t> comp(frame_bound(v.size())), back(v.size());
    size_t clen = cpu_compress_frame(v.data(), v.size(), comp.data(), comp.size());
    bool rejected = true;
    for (size_t cut = 0; cut < clen; cut += 997)
        if (cpu_decompress_frame(comp.data(), cut, back.data(), back.size()) != 0)
            rejected = false;
    printf("%-16s %s\n", "truncation", rejected ? "OK" : "FAIL");
    ok &= rejected;

    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "TESTS FAILED");
    return ok ? 0 : 1;
}
