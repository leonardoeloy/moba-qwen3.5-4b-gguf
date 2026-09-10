#pragma once
#include "ggml-backend.h"
#include <cstdio>
#include <fstream>
#include <vector>

static ggml_tensor * kv_adapter(ggml_context * ctx, ggml_tensor * cur, int layer) {
    const char * path = std::getenv("BENCH_ADAPTER");
    if (!path || layer != 7) return cur;
    GGML_ASSERT(kv_source(layer) == 3 && cur->ne[0] == 4096);
    struct weights {
        ggml_context * ctx;
        ggml_backend_buffer_t buffer;
        ggml_tensor * matrix, * bias;
        weights(const char * path) {
            ctx = ggml_init({ggml_tensor_overhead()*4, nullptr, true});
            matrix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 256, 16);
            bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 16);
            ggml_set_name(matrix, "kv_adapter_matrix");
            ggml_set_name(bias, "kv_adapter_bias");
            auto device = ggml_backend_dev_by_name("Vulkan0");
            GGML_ASSERT(device);
            buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(device));
            GGML_ASSERT(buffer);
            std::fprintf(stderr, "kv_adapter: backend=%s bytes=%zu\n", ggml_backend_dev_name(device), ggml_backend_buffer_get_size(buffer));
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::ifstream file(path, std::ios::binary);
            for (auto t : {matrix, bias}) {
                std::vector<float> data(ggml_nelements(t));
                GGML_ASSERT(file.read(reinterpret_cast<char *>(data.data()), ggml_nbytes(t)));
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            }
            GGML_ASSERT(file.peek() == std::char_traits<char>::eof());
        }
        ~weights() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
    };
    static weights w(path);
    auto x = ggml_permute(ctx, ggml_reshape_3d(ctx, cur, 256, 16, cur->ne[1]), 0, 2, 1, 3);
    auto y = ggml_add(ctx, ggml_mul_mat(ctx, w.matrix, x), w.bias);
    return ggml_cont_2d(ctx, ggml_permute(ctx, y, 0, 2, 1, 3), 4096, cur->ne[1]);
}
