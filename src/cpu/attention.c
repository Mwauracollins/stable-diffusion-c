#include <stdlib.h>
#include <utils/softmax.h>
#include <cmath>

void init_query_key_value(
    float** query_proj, 
    float** key_proj, 
    float** value_proj, 
    int D_MODEL
) {
    *query_proj = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
    *key_proj = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
    *value_proj = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));

    if (*query_proj == NULL || *key_proj == NULL || *value_proj == NULL) {
        printf("Failed to allocate memory for query, key, and value weights\n");
        exit(1);
    }

    for (int i = 0; i < D_MODEL * D_MODEL; i++) {
        (*query_proj)[i] = random_normal();
        (*key_proj)[i] = random_normal();
        (*value_proj)[i] = random_normal();
    }
}

typedef struct {
    float* qkv_w; // (C, 3 *C)
    float* qkv_b; // (3 * C)
    float* o_w;
    float* o_b;
} AttentionParameters;

typedef struct {
    float* Q; // B, T, n_heads, d_k
    float* K;
    float* V;
    float* scores; // B, n_heads, T, T
    float* attn_weights;
} AttentionActivations;

void compute_qkv(
    float* inputs, // Post-layernorm input, shape (B, T, C)
    float* Q,      // Query activations, shape (B, T, C)
    float* K,      // Key activations, shape (B, T, C)
    float* V,      // Value activations, shape (B, T, C)
    float* qkv_w,  // Weight matrix, shape (C, 3 * C)
    float* qkv_b,  // Bias vector, shape (3 * C)
    int B, int T, int C
) {
    int C3 = 3 * C;

    for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            float q = qkv_b[c];
            float k = qkv_b[C + c];
            float v = qkv_b[2 * C + c];
            for (int i = 0; i < C; i++) {
                float input_val = inputs[b * T * C + t * C + i];
                q += input_val * qkv_w[i * C3 + c];
                k += input_val * qkv_w[i * C3 + C + c];
                v += input_val * qkv_w[i * C3 + 2 * C + c];
            }
            Q[b * T * C + t * C + c] = q;
            K[b * T * C + t * C + c] = k;
            V[b * T * C + t * C + c] = v;
        }
    }
}

}

void compute_qkv_backward(
    float* dqkv_w,
    float* dqkv_b,
    float* dQ, float* dK, float* dV,
) {
    // Gradients w.r.t. QKV weights (dqkv_w, dqkv_b)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                dqkv_b[c] = dQ[b * T * C + t * C + c] + dK[b * T * C + t * C + c] + dV[b * T * C + t * C + c];

                for (int i = 0; i < C; i++) {
                    float dqkv_w_q = dQ[b * T * C + t * C + c] * Q[b * T * C + t * C + c];
                    float dqkv_w_k = dK[b * T * C + t * C + c] * K[b * T * C + t * C + c];
                    float dqkv_w_v = dV[b * T * C + t * C + c] * V[b * T * C + t * C + c];

                    dqkv_w[i * 3 * C + c] = dqkv_w_q;
                    dqkv_w[i * 3 * C + C + c] = dqkv_w_k;
                    dqkv_w[i * 3 * C + 2 * C + c] = dqkv_w_v;
                }
            }
        }
    }
}

void self_attention_forward(
    float* output, // B, T, C
    float* o_w, // (C, C)
    float* o_b, // (C)
    float* Q, float* K, float* V, // (B, T, C)
    float* scores, float* attn_weights, //(B, n_heads, T, T)
    float* head_output, // (B, T, C)
    int B, int T, int C,
    int n_heads
) {
    int d_k = C / n_heads;

    //calc attention for each head
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int t1 = 0; t1 < T; t1++) {
                float max_score = -INFINITY;
                
                for (int t2 = 0; t2 < T; t2++) {
                    float score = 0.0f;

                    // attention scores Q.K^T
                    for (int d = 0; d < d_k; d++) {
                        score += Q[b * T * C + t1 * C + h * d_k + d] * 
                                 K[b * T * C + t2 * C + h * d_k + d];
                    }
                    // scale the score by the square root of d_k
                    score /= sqrtf(d_k);
                    max_score = fmaxf(max_score, score);
                    scores[b * n_heads * T * T + h * T * T + t1 * T + t2] = score;
                }

                // Apply softmax with max norm and store the normalized attention scores
                float sum = 0.0f;
                for (int t2 = 0; t2 < T; t2++) {
                    float score = scores[b * n_heads * T * T + h * T * T + t1 * T + t2];
                    attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2] = expf(score - max_score);
                    sum += attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2];
                }
                // Normalize the attention weights
                for (int t2 = 0; t2 < T; t2++) {
                    attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2] /= sum;
                }
            }
            
            // Get the weighted sum of V for the output of this head
            for (int t1 = 0; t1 < T; t1++) {
                for (int d = 0; d < d_k; d++) {
                    float weighted_sum = 0.0f;
                    for (int t2 = 0; t2 < T; t2++) {
                        float attn = attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2];
                        float val = V[b * T * C + t2 * C + h * d_k + d];
                        weighted_sum += attn * val;
                    }
                    head_output[b * T * C + t1 * C + h * d_k + d] = weighted_sum;
                }
            }
        }

        // Concat all the heads' outputs and apply the linear transform
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                // Init output with bias
                output[b * T * C + t * C + c] = o_b[c];
                
                // Add weighted sum of all heads
                for (int h = 0; h < n_heads; h++) {
                    int head_offset = h * d_k + (c % d_k); // Ensure proper head offset mapping
                    output[b * T * C + t * C + c] += 
                        head_output[b * T * C + t * C + head_offset] * o_w[head_offset * C + c];
                }
            }
        }
    }
}

void self_attention_backward(
    float* doutput, // Gradient of the output (B, T, C)
    float* o_w, float* o_b, // Output weights and biases (C, C)
    float* Q, float* K, float* V, // Query, Key, Value activations (B, T, C)
    float* scores, float* attn_weights, // Scores and attention weights (B, n_heads, T, T)
    float* head_output, // Output of each attention head (B, T, C)
    float* dQ, float* dK, float* dV, // Gradients of Q, K, V (B, T, C)
    float* dqkv_w, float* dqkv_b, // Gradients of QKV weights and biases (C, 3 * C)
    int B, int T, int C,
    int n_heads
) {
    int d_k = C / n_heads;

    // Gradients for output weights (o_w, o_b)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                // Gradient with respect to o_b
                float do_b = 0.0f;
                for (int h = 0; h < n_heads; h++) {
                    do_b += doutput[b * T * C + t * C + c];
                }

                // Gradient with respect to o_w
                for (int h = 0; h < n_heads; h++) {
                    int head_offset = h * d_k + (c % d_k);
                    float do_w = 0.0f;
                    for (int t1 = 0; t1 < T; t1++) {
                        do_w += head_output[b * T * C + t1 * C + head_offset] * doutput[b * T * C + t * C + c];
                    }
                }
            }
        }
    }

    // Gradients for head_output (weighted sum of V)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int t1 = 0; t1 < T; t1++) {
                for (int d = 0; d < d_k; d++) {
                    float dhead_output = 0.0f;
                    for (int c = 0; c < C; c++) {
                        dhead_output += o_w[d * C + c] * doutput[b * T * C + t1 * C + c];
                    }
                }
            }
        }
    }

    // Gradients for attention weights and scores (backprop through softmax)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int t1 = 0; t1 < T; t1++) {
                for (int t2 = 0; t2 < T; t2++) {
                    float dscore = 0.0f;

                    // Gradient of softmax (chain rule for softmax operation)
                    dscore += doutput[b * T * C + t1 * C + h] * attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2];

                }
            }
        }
    }

    // Gradients for Q, K, V (backpropagate through the attention mechanism)
    for (int b = 0; b < B; b++) {
        for (int t1 = 0; t1 < T; t1++) {
            for (int t2 = 0; t2 < T; t2++) {
                for (int h = 0; h < n_heads; h++) {
                    for (int d = 0; d < d_k; d++) {
                        // Compute gradients w.r.t. Q, K, V
                        dQ[b * T * C + t1 * C + h * d_k + d] += attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2] * scores[b * n_heads * T * T + h * T * T + t1 * T + t2];
                        dK[b * T * C + t2 * C + h * d_k + d] += attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2] * scores[b * n_heads * T * T + h * T * T + t1 * T + t2];
                        dV[b * T * C + t2 * C + h * d_k + d] += attn_weights[b * n_heads * T * T + h * T * T + t1 * T + t2] * scores[b * n_heads * T * T + h * T * T + t1 * T + t2];
                    }
                }
            }
        }
    }
}


