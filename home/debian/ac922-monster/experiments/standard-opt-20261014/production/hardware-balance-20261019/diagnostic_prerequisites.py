"""Allow performance collection after the specific observed prefix eviction.

The original qualification remains FAIL and this path cannot authorize release.
"""
def check(main):
    if main['status'] == 'MODEL_COMPLETE_PENDING_LOG_AND_TEXT_REVIEW':
        return False
    assert main['status'] == 'FAIL'
    assert main['full_pool_and_single_capacity_semantics'] == 'PASS'
    assert main['four512_all_2048_token_ids_exact'] is True
    assert main['four512_full_texts_exact'] is True
    assert len(main['waves']) == 10
    assert all(not wave.get('errors') for wave in main['waves'])
    fill, hit = main['waves'][-2:]
    assert fill['id'].startswith('hardware-semantic-prefix-fill-')
    assert hit['id'].startswith('hardware-semantic-prefix-hit-')
    assert fill['texts']['long-paris'].strip() == hit['texts']['long-paris'].strip() == 'Paris'
    assert hit['timings']['cached_tokens'] == 0
    assert fill['timings']['post_wave_reclaimed_banks'] >= 1
    return True
