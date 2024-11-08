#include <stdlib.h>

float* init_token_embeddings(
    int vocab_size,
    int d_model
) {
    float* wte = (float*) malloc(vocab_size * d_model * sizeof(float));

    if (wte == NULL) {
        printf("Failed to allocate memory for token embedding");
        exit(1);
    }

    for (int i = 0; i < vocab_size * d_model; i++) {
        wte[i] = rand_normal();
    }
    return wte;
}

float* init_position_embeddings(
    int sequence_length,
    int d_model
) {
    float* wpe = (float*) malloc(sequence_length * d_model * sizeof(float));

    if (wpe == NULL) {
        printf("Failed to allocate memory for positinal embedding");
        exit(1);
    }

    for (int i = 0; i < sequence_length * d_model; i++) {
        wpe[i] = random_normal();
    }
    return wpe;
}

void clip_embedding(
    float* tokens, // B, T
    float* output, //B, T, C where C is the d_model
    float* wte,
    float* wpe,
    int B,
    int T, // where T is the sequence_length
) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = tokens[b * T + t]; // get the id for the current position

            if (token_id >= 0 && token_id < N_VOCABS) {
                for (int c = 0; c < D_MODEL; c++){
                    float token_value = wte[token_id * D_MODEL + c];
                    float position_value = wpe[t * D_MODEL + c];
                    output[b * T * D_MODEL + t * D_MODEL + D_MODEL] = token_value + position_value;
                }
            } else {
                for (int c = 0; c < D_MODEL; c++) {
                    output[b *  T * D_MODEL + t * D_MODEL + c] = 0.0f;
                }
            }
        }
    }
}

