"""Model routes reuse the gateway's existing authentication middleware."""
import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

BACKEND='http://127.0.0.1:8001'
BODY_LIMIT=4<<20


def install_model_routes(app):
    async def forward(path,request):
        body=bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body)>BODY_LIMIT:
                raise HTTPException(413,'Model request exceeds 4 MiB.')
        client=httpx.AsyncClient(timeout=httpx.Timeout(1805,connect=5))
        try:
            upstream=await client.send(client.build_request(
                request.method,BACKEND+path,content=bytes(body),
                headers={'content-type':request.headers.get('content-type','application/json')}),stream=True)
        except httpx.RequestError:
            await client.aclose()
            raise HTTPException(503,'Model backend is unavailable.')

        async def close():
            await upstream.aclose()
            await client.aclose()

        content_type=upstream.headers.get('content-type','application/json')
        if upstream.status_code<400 and content_type.startswith('text/event-stream'):
            async def stream():
                try:
                    async for part in upstream.aiter_raw(): yield part
                finally: await close()
            return StreamingResponse(stream(),status_code=upstream.status_code,
                headers={'content-type':content_type,'cache-control':'no-cache','x-accel-buffering':'no'},
                background=BackgroundTask(close))
        try:
            data=await upstream.aread()
            return Response(data,status_code=upstream.status_code,headers={'content-type':content_type})
        finally:
            await close()

    @app.get('/v1/models')
    async def models(request:Request): return await forward('/v1/models',request)

    @app.post('/v1/chat/completions')
    async def chat(request:Request): return await forward('/v1/chat/completions',request)

    @app.post('/v1/completions')
    async def completion(request:Request): return await forward('/v1/completions',request)
