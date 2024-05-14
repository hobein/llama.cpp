#include <vector>

#include "simple-llama.h"
#include "common/common.h"

static std::tuple<struct llama_model *, struct llama_context *, simple_llama_status> simple_llama_init_from_gpt_params(gpt_params & params);

struct simple_llama {
    gpt_params params;
    llama_model * model;
    llama_context * ctx;
};

// simple_llama_inference_state keep track of the inference state
struct simple_llama_inference_state {
    llama_sampling_context * ctx_sampling;
    std::vector<llama_token> embd_inp; // input prompt tokens
    std::vector<llama_token> embd;  // currently processing tokens
    std::vector<llama_token> antiprompt_ids;
    int n_remain;
    int n_consumed;
    int n_past;
};

struct simple_llama* simple_llama_new() {
    simple_llama * simple_llm = new simple_llama;

    simple_llm->model = nullptr;
    simple_llm->ctx = nullptr;

    return simple_llm;
}

simple_llama_status simple_llama_init_model(struct simple_llama* simple_llm, const char* model_file) {
    // forced settings
    simple_llm->params.logits_all = false;
    simple_llm->params.embedding = false;
    simple_llm->params.rope_freq_base = 0.0;
    simple_llm->params.rope_freq_scale = 0.0;
    simple_llm->params.n_keep = 0;
    simple_llm->params.interactive_first = false; // unused, no interactive mode
    simple_llm->params.verbose_prompt = false;
    simple_llm->params.grp_attn_n = 1;     // group-attention factor
    simple_llm->params.grp_attn_w = 512;   // group-attention width

    simple_llm->params.n_predict = -2; // number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
    // Use the model context size.
    // See llama_new_context_with_model
    simple_llm->params.n_ctx = 0;

    simple_llm->params.sparams.cfg_scale = 1.f;

    simple_llm->params.model = model_file;

    simple_llama_status status;
    std::tie(simple_llm->model, simple_llm->ctx, status) = simple_llama_init_from_gpt_params(simple_llm->params);

    return status;
}

void simple_llama_free(struct simple_llama* simple_llm) {
    if (simple_llm == nullptr) {
        return;
    }

    if (simple_llm->ctx != nullptr) {
        llama_free(simple_llm->ctx);
        simple_llm->ctx = nullptr;
    }

    if (simple_llm->model != nullptr) {
        llama_free_model(simple_llm->model);
        simple_llm->model = nullptr;
    }


    delete simple_llm;
}

struct simple_llama_inference_state* simple_llama_inference_state_new() {
    simple_llama_inference_state * state = new simple_llama_inference_state;

    state->ctx_sampling = nullptr;
    state->n_remain = -1;   // infinite generation
    state->n_consumed = 0;
    state->n_past = 0;

    return state;
}

void simple_llama_inference_state_free(struct simple_llama_inference_state* state) {
    if (state->ctx_sampling != nullptr) {
        llama_sampling_free(state->ctx_sampling);
        state->ctx_sampling = nullptr;
    }

    delete state;
}

simple_llama_status simple_llama_set_prompt(struct simple_llama* simple_llm, struct simple_llama_inference_state* state, const char* prompt) {
    if ((prompt == nullptr) || strlen(prompt) == 0) {
        return SIMPLE_LLAMA_STATUS_EMPTY_PROMPT;
    }

    if (state->ctx_sampling == nullptr) {
        state->ctx_sampling = llama_sampling_init(simple_llm->params.sparams);
        if (state->ctx_sampling == nullptr) {
            return SIMPLE_LLAMA_STATUS_CONTEXT_SAMPLING_FAILURE;
        }
    } else {
        // reset the context sampling
        llama_sampling_reset(state->ctx_sampling);
    }

    // reset state
    state->embd = {};
    state->n_remain = -1;   // infinite generation
    state->n_consumed = 0;
    state->n_past = 0;
    // Assign the BOS and EOS token to the anti prompt
    state->antiprompt_ids = {llama_token_bos(simple_llm->model), llama_token_eos(simple_llm->model)};

    state->embd_inp = ::llama_tokenize(simple_llm->ctx, prompt, true, true);
    if (state->embd_inp.size() > llama_n_ctx(simple_llm->ctx) - 4) {
        state->embd_inp = {};
        return SIMPLE_LLAMA_STATUS_PROMPT_TOO_LONG;
    }


    return SIMPLE_LLAMA_STATUS_SUCCESS;
}

int simple_llama_inference_state_get_input_prompt_token_count(struct simple_llama*, struct simple_llama_inference_state* state) {
    return state->embd_inp.size();
}

simple_llama_inference_status simple_llama_inference_state_get_next_token(struct simple_llama* simple_llm, struct simple_llama_inference_state* state, llama_token* generated_token) {
    *generated_token = -1;

    if (state->n_remain == 0) {
        return SIMPLE_LLAMA_INFERENCE_STATUS_DONE;
    }

    int n_ctx = llama_n_ctx(simple_llm->ctx);

    while (state->n_remain != 0) {
        // predict
        if (!state->embd.empty()) {
             // Always stop when the context is full.
            if (state->n_past + (int) state->embd.size() >= n_ctx) {
                // Always stop when the context is full.
                return SIMPLE_LLAMA_INFERENCE_STATUS_CONTEXT_FULL;
            }

            for (int i = 0; i < (int) state->embd.size(); i += simple_llm->params.n_batch) {
                int n_eval = (int) state->embd.size() - i;
                if (n_eval > simple_llm->params.n_batch) {
                    n_eval = simple_llm->params.n_batch;
                }

                if (llama_decode(simple_llm->ctx, llama_batch_get_one(&state->embd[i], n_eval, state->n_past, 0))) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return SIMPLE_LLAMA_INFERENCE_STATUS_LLAMA_DECODE_FAILURE;
                }

                state->n_past += n_eval;
            }
        }

        state->embd.clear();

        if ((int) state->embd_inp.size() <= state->n_consumed) {
            const llama_token id = llama_sampling_sample(state->ctx_sampling, simple_llm->ctx, nullptr);

            llama_sampling_accept(state->ctx_sampling, simple_llm->ctx, id, true);

            state->embd.push_back(id);

            // decrement remaining sampling budget
            --state->n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) state->embd_inp.size() > state->n_consumed) {
                state->embd.push_back(state->embd_inp[state->n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(state->ctx_sampling, simple_llm->ctx, state->embd_inp[state->n_consumed], false);

                ++state->n_consumed;
                if ((int) state->embd.size() >= simple_llm->params.n_batch) {
                    break;
                }
            }
        }

        // end of generation
        if (!state->embd.empty()) {
            if (llama_token_is_eog(simple_llm->model, state->embd.back())) {
                std::cerr << "\t\t\tstate->embd.back(): " << state->embd.back() << " llama_token_is_eog is true" << std::endl;
                return SIMPLE_LLAMA_INFERENCE_STATUS_DONE;
            }

            // check for reverse prompt ids against the last token
            const auto last_token = state->embd.back();
            for (auto token_id : state->antiprompt_ids) {
                if (token_id == last_token) {
                    std::cerr << "\t\t\tlast_token: " << last_token << " equals anti prompt " << last_token << std::endl;
                    return SIMPLE_LLAMA_INFERENCE_STATUS_DONE;
                }
            }
        }

        // here embd size should always be zero or one.
        if (state->embd.size() == 1) {
            *generated_token = state->embd[0];
            return SIMPLE_LLAMA_INFERENCE_STATUS_SUCCESS;
        }
    }

    return SIMPLE_LLAMA_INFERENCE_STATUS_SUCCESS;
}

int32_t simple_llama_token_to_piece(struct simple_llama* simple_llm, llama_token token, char * buf, int32_t length) {
    return llama_token_to_piece(simple_llm->model, token, buf, length, true);
}

static std::tuple<struct llama_model *, struct llama_context *, simple_llama_status> simple_llama_init_from_gpt_params(gpt_params & params) {
    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model = nullptr;

    model = llama_load_model_from_file(params.model.c_str(), mparams);

    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return std::make_tuple(nullptr, nullptr, SIMPLE_LLAMA_STATUS_LOAD_MODEL_FAILURE);
    }

    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return std::make_tuple(nullptr, nullptr, SIMPLE_LLAMA_STATUS_CONTEXT_CREATION_FAILURE);
    }

    for (unsigned int i = 0; i < params.lora_adapter.size(); ++i) {
        const std::string & lora_adapter = std::get<0>(params.lora_adapter[i]);
        float lora_scale = std::get<1>(params.lora_adapter[i]);
        int err = llama_model_apply_lora_from_file(model,
                                             lora_adapter.c_str(),
                                             lora_scale,
                                             ((i > 0) || params.lora_base.empty())
                                                ? NULL
                                                : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            llama_free(lctx);
            llama_free_model(model);
            return std::make_tuple(nullptr, nullptr, SIMPLE_LLAMA_STATUS_LORA_FAILURE);
        }
    }

    if (params.ignore_eos) {
        params.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
    }

    if (params.warmup) {
        LOG("warming up the model with an empty run\n");

        std::vector<llama_token> tmp = { llama_token_bos(model), llama_token_eos(model), };
        llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch), 0, 0));
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_reset_timings(lctx);
    }

    return std::make_tuple(model, lctx, SIMPLE_LLAMA_STATUS_SUCCESS);
}
