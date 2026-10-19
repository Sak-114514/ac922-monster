"""One long-input semantic request also activates the lazily loaded R5 helper."""
import asyncio
from pathlib import Path

import lastwave_http_acceptance as http

HERE = Path(__file__).resolve().parent


async def main():
    output = HERE / 'post-restore-long-canary.json'
    report = {'status': 'RUNNING', 'phases': {}}
    base, root = http.api_urls('http://127.0.0.1:8007/v1')
    runner = http.Runner(http.StandardHttpClient(), base, root, 'flash-next-iq3-xxs',
                         http.load_keyfile(Path('/home/debian/apigateway/api_key')),
                         output, report)
    corpus = http.load_corpus_text(HERE.parents[1] / 'fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json')
    prompt, _ = http.build_real_corpus_prompt(corpus, 32768)
    try:
        result = await runner.nonstream(prompt, 24, 'post-restore-native-long')
        assert result['text'].strip() == 'EPSILON72941'
        assert result['usage']['prompt_tokens'] >= 8192
        report.update(status='PASS', request=result,
                      prompt_length='approximate real-corpus; API token count is authoritative')
    finally:
        http.write_report(output, report)


if __name__ == '__main__':
    asyncio.run(main())
