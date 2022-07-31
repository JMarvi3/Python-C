from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse

METHOD_GET = "GET"
METHOD_POST = "POST"
METHOD_HEAD = "HEAD"
HTML_CODE = """\
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Example</title>
    </head>
    <body>
        <h2>Simple HTML Code</h2>
        <h3>Python C API</h3>
        <a href="https://github.com/GiorgioMegrelli/">Github</a>
    </body>
</html>
"""


@dataclass
class Info:
    method: str
    url: str


Get = lambda u: Info(METHOD_GET, u)
Post = lambda u: Info(METHOD_POST, u)
Head = lambda u: Info(METHOD_HEAD, u)


def main(*, host: str, port: int) -> None:
    app = FastAPI()
    html_response = HTMLResponse(content=HTML_CODE, status_code=200)

    @app.get("/")
    async def root_get():
        return Get("/")

    @app.head("/")
    async def root_head():
        return Head("/")

    @app.post("/")
    async def root_post():
        return Post("/")

    @app.get("/html")
    async def html_get():
        return html_response

    @app.head("/html")
    async def html_head():
        return html_response

    @app.post("/text")
    async def text_post():
        return "Just text response on Post Request"

    @app.post("/request-header")
    async def request_header(request: Request):
        test_headers = {
            "Request": str(True).lower(),
            "Hello": "World",
            "Custom-Header-1": "Custom-Header-1-Value",
            "Custom-Header-2": "Custom-Header-2-Value",
        }
        request_headers = dict(request.headers)
        result = {}
        for k, v in test_headers.items():
            kl = k.lower()
            result[kl] = kl in request_headers and v == request_headers[kl]
        return result

    uvicorn.run(app, host=host, port=port, reload=False, debug=True)


if __name__ == "__main__":
    main(host="localhost", port=8080)
