#pragma once
#include <cstdlib>

static int kv_source(int layer) {
    const char * value = std::getenv("BENCH_SHARE_MASK");
    const int mask = value ? std::atoi(value) : 0;
    GGML_ASSERT(mask >= 0 && mask <= 15);
    return layer % 8 == 7 && (mask & (1 << (layer/8))) ? layer-4 : layer;
}
