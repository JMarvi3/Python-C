import click as cli
import myhttp

DEFAULT_LOCAL_PORT = 8080
FULL_PRINT: bool = True


def print_result(value) -> None:
    if FULL_PRINT:
        print(value)
    else:
        print(f"String result of {len(value)} bytes")


def external() -> None:
    """Tests external urls"""
    host = "info.cern.ch"
    url = "/hypertext/WWW/TheProject.html"
    print_result(myhttp.get(host, url, None))


def internal(*, port: int) -> None:
    """Tests localhost"""
    host = f"localhost:{port}"
    methods = {
        "get": myhttp.get,
        "head": myhttp.head,
        "post": myhttp.post,
    }
    test_routes = [
        ("get", "/"),
        ("head", "/"),
        ("post", "/"),
        ("get", "/html"),
        ("head", "/html"),
        ("post", "/text"),
    ]

    for method, url in test_routes:
        print(f"{method.upper()} at '{url}':")
        print_result(methods[method](host, url, None))

    headers = {
        "Request": str(True).lower(),
        "Hello": "World",
        "Custom-Header-1": "Custom-Header-1-Value",
        "Custom-Header-2": "Custom-Header-2-Value",
    }
    response = myhttp.post(host, "/request-header", None, headers)
    print_result(response)


@cli.command()
@cli.option("--no-external", is_flag=True, default=False, help="Exclude external links")
@cli.option(
    "--no-internal",
    is_flag=True,
    default=False,
    help="Exclude internal (local) links",
)
@cli.option(
    "--port",
    type=int,
    default=DEFAULT_LOCAL_PORT,
    show_default=True,
    help="Localhost port",
)
@cli.option("--full-print", is_flag=True, default=False, help="Prints full response")
def main(no_external: bool, no_internal: bool, port: int, full_print: bool) -> None:
    global FULL_PRINT
    FULL_PRINT = full_print

    if not no_external:
        external()
    if not no_internal:
        internal(port=port)


if __name__ == "__main__":
    main()
