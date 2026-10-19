"""Use the build window for one actual long-context production canary."""
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F5 = HERE.parent/'safe-pool-reject-20261019'
sys.path.insert(0,str(F5))
import lastwave_http_acceptance as http

async def main():
    output = HERE/'l5-long200k-c2.json'
    report = {'status':'RUNNING','phases':{},'scope':'one long input semantic/capacity canary, not a baseline or pressure run'}
    base,root = http.api_urls('http://127.0.0.1:8007/v1')
    runner = http.Runner(http.StandardHttpClient(),base,root,'flash-next-iq3-xxs',
                         http.load_keyfile(Path('/home/debian/apigateway/api_key')),output,report)
    corpus = http.load_corpus_text(HERE.parents[1]/'fixtures/long-prefill-combo-v45-20261016/v45-real-corpus-source.json')
    prompt,_ = http.build_real_corpus_prompt(corpus,235000)
    try:
        result = await runner.nonstream(prompt,24,'production-actual-long200k')
        report['request'] = result
        http.write_report(output,report)
        assert 200000 <= result['usage']['prompt_tokens'] <= 262140
        assert result['text'].strip() == 'EPSILON72941'
        report.update(status='PASS',request=result,
                      prompt_length='approximate source target; actual API usage is authoritative')
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        http.write_report(output,report)

if __name__ == '__main__':
    asyncio.run(main())
