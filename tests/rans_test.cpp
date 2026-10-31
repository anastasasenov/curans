/** Standalone CPU test of the rANS */
#include "../src/rans.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>

static uint32_t rng = 12345u;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static bool roundtrip(const std::vector<uint8_t>& syms, const char* name) {
    const int n = (int)syms.size();
    uint32_t counts[256] = {0};
    for (int i = 0; i < n; i++) counts[syms[i]]++;
    uint16_t freq[256], start[256];
    rans_normalize(counts, freq, start);

    std::vector<uint8_t> buf((size_t)n * 2 + 64);
    uint8_t* p = buf.data() + buf.size();
    uint32_t x = RANS_L;
    for (int i = n - 1; i >= 0; i--) {
        uint8_t s = syms[i];
        x = rans_put(x, &p, start[s], freq[s]);
    }
    rans_flush(x, &p);
    size_t clen = (size_t)(buf.data() + buf.size() - p);

    uint8_t tab[4096];
    rans_build_table(freq, start, tab);
    std::vector<uint8_t> out(n);
    const uint8_t* q = p;
    uint32_t d = rans_ld32(q);
    q += 4;
    for (int i = 0; i < n; i++) out[i] = rans_get(&d, &q, tab, freq, start);

    bool ok = (q == buf.data() + buf.size()) && (out == syms);
    double H = 0;
    for (int s = 0; s < 256; s++)
        if (freq[s]) H -= (freq[s] / 4096.0) * (freq[s] ? __builtin_log2(freq[s] / 4096.0) : 0.0);
    double expect = (H / 8.0) * n;
    printf("%-18s n=%7d  in=%8d  out=%8d  %.3f b/sym  expected~%8.0f  %s\n",
           name, n, n, (int)clen, 8.0 * clen / n, expect,
           ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    std::vector<uint8_t> v;

    v.resize(200000);
    for (auto& b : v) b = (uint8_t)rnd();
    ok &= roundtrip(v, "uniform random");

    v.resize(200000);
    const char* text = "the quick brown fox jumps over the lazy dog 0123456789 ";
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t r = rnd();
        v[i] = (r % 4) ? (uint8_t)text[r % 55] : (uint8_t)(r & 0x7f);
    }
    ok &= roundtrip(v, "text-like");

    v.assign(100000, 0x41);
    ok &= roundtrip(v, "single symbol");

    v.resize(100000);
    for (size_t i = 0; i < v.size(); i++) v[i] = (uint8_t)((i * 7) & 0xff);
    ok &= roundtrip(v, "sawtooth");

    v.resize(3);
    v[0] = 'a'; v[1] = 'b'; v[2] = 'a';
    ok &= roundtrip(v, "tiny");

    v.resize(1);
    v[0] = 0xFF;
    ok &= roundtrip(v, "one byte");

    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "TESTS FAILED");
    return ok ? 0 : 1;
}
