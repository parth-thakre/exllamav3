#pragma once

#include <cstdint>
#include <vector>

class Graph;

void gated_delta_net_fused_op
(
    const at::Tensor& mixed_qkvz,
    const at::Tensor& mixed_ba,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    at::Tensor& mixed_qkv,
    at::Tensor& z,
    at::Tensor& beta,
    at::Tensor& g,
    size_t num_k_heads,
    size_t num_v_heads,
    size_t k_head_dim,
    size_t v_head_dim,
    const float beta_scale
);

void gated_delta_net_fused_op_2
(
    const at::Tensor& b,
    const at::Tensor& a,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    at::Tensor& beta,
    at::Tensor& g,
    const float beta_scale
);

// save_state=false reads committed state and produces outputs without changing state or history.
void cuda_recurrent_gated_delta_rule
(
    const at::Tensor& mixed_qkv,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    bool save_state = true
);

void cuda_recurrent_gated_delta_rule_gr
(
    const at::Tensor& mixed_qkv,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    Graph* graph,
    bool save_state = true
);

// Mamba2 discretization: dt = clamp(softplus(dt_raw + dt_bias), dt_min, dt_max), g = -exp(a_log) * dt
void mamba2_dt_op
(
    const at::Tensor& dt_raw,       // [B,S,H] float
    const at::Tensor& dt_bias,      // [H] float
    const at::Tensor& a_log,        // [H] float
    at::Tensor& dt,                 // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H] float
    float dt_min,
    float dt_max
);

// Mamba2 BC helper: bf16 conv input + discretized dt/g + a contiguous gate copy, from the
// in_proj output [z, xBC, dt]
void mamba2_fused_op_gr
(
    const at::Tensor& proj,         // [B,S,>= v_dim + F + dt_first + H] float
    at::Tensor& xbc,                // out [B, F, S] bfloat16
    at::Tensor& dt,                 // out [B, S, H] bfloat16
    at::Tensor& g,                  // out [B, S, H] float
    at::Tensor& z_gate,             // out [B, S, v_dim] float, contiguous
    const at::Tensor& dt_bias,      // [H] float
    const at::Tensor& a_log,        // [H] float
    int v_dim,
    int dt_first,
    float dt_min,
    float dt_max,
    class Graph* graph
);

// Mamba2 (SSD) recurrence: gated delta rule without the correction term, over conv channel
// order [x, B, C], with dt as input scale and per-head skip y += D * x
void cuda_recurrent_mamba2
(
    const at::Tensor& mixed_xbc,
    const at::Tensor& g,
    const at::Tensor& dt,
    const at::Tensor& D,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history
);

void cuda_recurrent_mamba2_gr
(
    const at::Tensor& mixed_xbc,
    const at::Tensor& g,
    const at::Tensor& dt,
    const at::Tensor& D,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    Graph* graph
);

// save_state=false reads the committed window and leaves conv_state unchanged (requires !history)
void cuda_causal_conv1d_update
(
    const at::Tensor& x,
    at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    bool activation,
    bool history,
    bool save_state = true
);

void cuda_causal_conv1d_update_gr
(
    const at::Tensor& x,
    at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    bool activation,
    bool history,
    Graph* graph,
    bool save_state = true
);

// Split-projection (Qwen3.5) helper: cast/transpose qkv to bf16 mixed_qkv and compute beta/g from
// the packed ba projection
void gated_delta_net_fused_op_3_gr
(
    const at::Tensor& qkv,          // [B,S,F] float
    const at::Tensor& ba,           // [B,S,2H] float
    const at::Tensor& dt_bias,      // [H] bfloat16
    const at::Tensor& a_log,        // [H] float or bfloat16
    at::Tensor& mixed_qkv,          // out [B,F,S] bfloat16
    at::Tensor& beta,               // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H] float
    const float beta_scale,
    Graph* graph
);

// Small fp16 GEMV with fp32 accumulation/output for the merged b/a projections. Avoids cublas so
// the x pointer can be patched in captured graphs
void gdn_ba_gemv
(
    const at::Tensor& x,            // [.., k] half
    const at::Tensor& w_t,          // [n, k] half
    const c10::optional<at::Tensor>& bias,  // [n] half
    at::Tensor& y                   // [.., n] float
);

void gdn_ba_gemv_gr
(
    const at::Tensor& x,            // [.., k] half
    const at::Tensor& w_t,          // [n, k] half
    const c10::optional<at::Tensor>& bias,  // [n] half
    at::Tensor& y,                  // [.., n] float
    Graph* graph
);

// Float-input fp16-weight GEMV for the KDA low-rank second stages (graph statics, no patching)
void gdn_lowrank_gemv_f_gr
(
    const at::Tensor& x,            // [.., k] float
    const at::Tensor& w_t,          // [n, k] half
    at::Tensor& y,                  // [.., n] float
    Graph* graph
);

// KDA (GLM5.3) helper: cast/transpose qkv to bf16 mixed_qkv, beta = sigmoid(b), per-k-channel
// log decay from the low-rank forget path (safe-gate when lower_bound != 0, else softplus)
void kda_gate_op_gr
(
    const at::Tensor& qkv,          // [B,S,F] float
    const at::Tensor& b,            // [B,S,H] float
    const at::Tensor& f,            // [B,S,H*Dk] float
    const at::Tensor& dt_bias,      // [H*Dk] bfloat16
    const at::Tensor& a_log,        // [H] float or bfloat16
    at::Tensor& mixed_qkv,          // out [B,F,S] bfloat16
    at::Tensor& beta,               // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H,Dk] float
    const float lower_bound,
    const float beta_scale,
    Graph* graph
);

// Batched recurrent-state rewind (speculative decoding draft rejection/commit)

// conv_state shift: conv_state[slot, :, :cdim] <- conv_state[slot, :, p-cdim:p]. `dim` independent
// per-channel copies of `cdim` elements, `stride` elements apart in both src and dst (same
// tensor, same per-channel stride). src/dst can overlap when num_tokens < conv_kernel_size.
struct ConvRewindJob
{
    uintptr_t src;
    uintptr_t dst;
    int dim;
    int cdim;
    int stride;

    ConvRewindJob() = default;
    ConvRewindJob(uintptr_t _src, uintptr_t _dst, int _dim, int _cdim, int _stride) :
        src(_src), dst(_dst), dim(_dim), cdim(_cdim), stride(_stride) {}
};

// recurrent_state rewind: recurrent_state[slot, 0] <- recurrent_state[slot, last_history+1-num_tokens].
// Flat fp32 copy of num_elements contiguous elements; src/dst never overlap (num_tokens >= 1
// forces the source history index to differ from destination index 0).
struct StateRewindJob
{
    uintptr_t src;
    uintptr_t dst;
    int64_t num_elements;

    StateRewindJob() = default;
    StateRewindJob(uintptr_t _src, uintptr_t _dst, int64_t _num_elements) :
        src(_src), dst(_dst), num_elements(_num_elements) {}
};

void batched_conv_rewind(std::vector<ConvRewindJob> const& jobs, int device_index);
void batched_state_rewind(std::vector<StateRewindJob> const& jobs, int device_index);

// Accepted-input replay commit for one state slot of one layer (EXL3_GDN_REPLAY). Addresses are
// row/slot bases: recurrent_state = the slot's committed plane (float [Nv, 128, 128]), conv_state =
// the slot's window (bf16 [F, conv_stride]); the recorded verify row holds conv_input (bf16
// [F, seqlen], channel-major), conv_out (bf16 [seqlen, F]), beta (bf16 [seqlen, Nv]) and g (float
// [seqlen, Nv], or [seqlen, Nv, 128] channelwise)
struct GDNReplayJob
{
    uintptr_t recurrent_state;
    uintptr_t conv_state;
    uintptr_t conv_input;
    uintptr_t conv_out;
    uintptr_t beta;
    uintptr_t g;

    GDNReplayJob() = default;
    GDNReplayJob(uintptr_t _recurrent_state, uintptr_t _conv_state, uintptr_t _conv_input,
                 uintptr_t _conv_out, uintptr_t _beta, uintptr_t _g) :
        recurrent_state(_recurrent_state), conv_state(_conv_state), conv_input(_conv_input),
        conv_out(_conv_out), beta(_beta), g(_g) {}
};

// Commit the first `accepted` recorded tokens of every job: one recurrent-scan launch and one
// conv-window launch per 64 jobs. All jobs share the geometry and record length `seqlen`
void batched_gdn_replay
(
    std::vector<GDNReplayJob> const& jobs,
    int device_index,
    int accepted,
    int seqlen,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    int conv_dim,
    int conv_k,
    int conv_stride,
    bool channelwise
);
