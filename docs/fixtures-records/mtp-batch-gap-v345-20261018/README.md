# V345: actual batch allocator forward-gap boundary

This fixture directly calls the production `llama_batch_allocr::init()` and is bounded to six two-row CPU inputs. It is prepared against the root-owned V344 change that appends `allow_forward_gap` as the final `init()` argument. It does not mirror the policy in a helper and does not modify engine sources.

The batch uses `token == nullptr`, one-dimensional `embd` rows (`n_embd == 1`), explicit sequence 0 IDs, explicit positions, and zero output flags. A default-constructed real `llama_vocab` is passed; the allocator does not consult its token count for an embeddings-only batch. The fake memory object exposes sequence 0 at `[0, 63]`; only position-bound queries are permitted. Unexpected memory/context/mutation virtual calls throw and are counted.

| Case | `allow_forward_gap` | Expected |
|---|---:|---|
| Positions 100, 101 after memory max 63 | false | reject |
| Positions 100, 101 after memory max 63 | true | accept |
| Positions 63, 64 | true | reject overlap |
| Positions 60, 61 | true | reject backward start |
| Positions 100, 102 | true | reject an internal hole |
| Positions 64, 65 | false | accept contiguous continuation |

`root-host-link-v345.sh` refuses to compile until `V345_V344_FROZEN=1` and exact frozen runtime metadata are supplied. It links to the explicit versioned DSO paths, runs the CPU fixture with `CUDA_VISIBLE_DEVICES` empty, and never initializes the backend, loads a model, or invokes inference. The PLE/fan/GPU runtime is outside this fixture.
