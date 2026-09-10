#include <cstdlib>

static int moba_option(const char * key, int fallback) {
    const char * value = std::getenv(key);
    return value ? std::atoi(value) : fallback;
}

static ggml_tensor * moba_mask(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * mask, int layer) {
    const int top = moba_option("MOBA_TOPK", 16), dense = moba_option("MOBA_DENSE_LAYERS", 2);
    if (!moba_option("MOBA", 0) || layer % 4 != 3 || layer >= 32 - 4*dense || !mask) return mask;
    const int64_t d = k->ne[0], nk = k->ne[1], nq = q->ne[1], nh = q->ne[2], nb = nk/64;
    if (nq < 256 || nq % 256 || nk % 256 || nb <= top || q->ne[3] != 1 || mask->ne[2] != 1) return mask;
    GGML_ASSERT(top >= 2 && d == 256 && nh == 16 && k->ne[2] == 4);
    auto keys = ggml_view_2d(ctx, k, d*k->ne[2], nk, k->nb[1], 0);
    if (keys->type != GGML_TYPE_F16) keys = ggml_cast(ctx, keys, GGML_TYPE_F32);
    keys = ggml_pool_2d(ctx, keys, GGML_OP_POOL_AVG, 1, 64, 1, 64, 0, 0);
    keys = ggml_permute(ctx, ggml_reshape_4d(ctx, keys, d, k->ne[2], nb, 1), 0, 2, 1, 3);
    auto scores = ggml_mul_mat(ctx, keys, q);
    auto causal = ggml_cast(ctx, ggml_cont(ctx, ggml_view_2d(ctx, mask, nk, nq, mask->nb[1], 0)), GGML_TYPE_F32);
    auto current = ggml_floor(ctx, ggml_scale_bias(ctx, ggml_sum_rows(ctx, ggml_exp(ctx, causal)), 1.0f/64, -1.0f/64));
    auto blocks = ggml_repeat_4d(ctx, ggml_arange(ctx, 0, nb, 1), nb, nq, 1, 1);
    auto delta = ggml_sub(ctx, blocks, ggml_repeat(ctx, current, blocks));
    auto local = ggml_scale_bias(ctx, ggml_step(ctx, ggml_abs(ctx, delta)), -1, 1);
    auto bias = ggml_scale(ctx, ggml_sub(ctx, local, ggml_step(ctx, delta)), 1e9f);
    bias = ggml_add(ctx, bias, ggml_scale_bias(ctx, ggml_step(ctx, blocks), -1e9f, 1e9f));
    scores = ggml_add(ctx, scores, bias);
    auto indices = ggml_reshape_2d(ctx, ggml_top_k(ctx, scores, top), top, nq*nh);
    auto zeros = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nb, nq*nh), 0);
    auto ones = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, top, nq*nh), 1);
    auto selected = ggml_reshape_4d(ctx, ggml_set_rows(ctx, zeros, ones, indices), nb, nq, nh, 1);
    if (moba_option("MOBA_COMPACT", 1)) {
        auto compact = ggml_cast(ctx, ggml_log(ctx, selected), GGML_TYPE_F16);
        ggml_set_name(compact, "moba_compact_64");
        return compact;
    }
    selected = ggml_interpolate(ctx, selected, nk, nq, nh, 1, GGML_SCALE_MODE_NEAREST);
    auto sparse = ggml_add(ctx, ggml_log(ctx, selected), causal);
    sparse = ggml_pad(ctx, sparse, 0, mask->ne[1] - nq, 0, 0);
    return ggml_cast(ctx, sparse, GGML_TYPE_F16);
}
