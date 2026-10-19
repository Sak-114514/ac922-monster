import asyncio,unittest
from unittest.mock import patch
import httpx
from fastapi import FastAPI
import llm_routes

Factory=httpx.AsyncClient

class Bytes(httpx.AsyncByteStream):
    def __init__(self,data):self.data=data;self.closed=False
    async def __aiter__(self):yield self.data
    async def aclose(self):self.closed=True

class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self,handler,path='/v1/models',body=None):
        app=FastAPI();llm_routes.install_model_routes(app)
        transport=httpx.MockTransport(handler)
        def upstream(*args,**kw):return Factory(*args,**kw,transport=transport)
        async with Factory(transport=httpx.ASGITransport(app=app),base_url='http://test') as frontend:
            with patch.object(llm_routes.httpx,'AsyncClient',upstream):
                if body is None:return await frontend.get(path)
                return await frontend.post(path,content=body,headers={'content-type':'application/json'})
    async def test_models(self):
        def handler(req):
            self.assertEqual(str(req.url),'http://127.0.0.1:8001/v1/models')
            return httpx.Response(200,json={'data':[{'id':'test-model'}]})
        r=await self.invoke(handler);self.assertEqual(r.json()['data'][0]['id'],'test-model')
    async def test_sse_raw_and_close(self):
        stream=Bytes(b'data: {"x":1}\n\n: keep-alive\n\ndata: [DONE]\n\n')
        def handler(req):return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=stream)
        r=await self.invoke(handler,'/v1/chat/completions',b'{"stream":true}')
        self.assertEqual(r.content,stream.data);self.assertTrue(stream.closed)
        self.assertEqual(r.headers['x-accel-buffering'],'no')
    async def test_backend_failure(self):
        def handler(req):raise httpx.ConnectError('test transport error',request=req)
        r=await self.invoke(handler);self.assertEqual(r.status_code,503)
    async def test_body_limit_before_upstream(self):
        def handler(req):raise AssertionError('oversized request forwarded')
        r=await self.invoke(handler,'/v1/completions',b'x'*(llm_routes.BODY_LIMIT+1));self.assertEqual(r.status_code,413)

if __name__=='__main__':unittest.main(verbosity=2)
