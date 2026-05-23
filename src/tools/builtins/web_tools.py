from __future__ import annotations

from .common import *

class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_link = False
        self.current_href: str | None = None
        self.current_text: list[str] = []
        self.results: list[JsonObject] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        css = attrs_dict.get("class") or ""
        if href and ("result__a" in css or "/l/?" in href):
            self.in_link = True
            self.current_href = href
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.in_link:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self.in_link:
            return
        title = html.unescape(" ".join("".join(self.current_text).split()))
        url = _clean_duckduckgo_url(self.current_href or "")
        if title and url and not any(item["url"] == url for item in self.results):
            self.results.append({"title": title, "url": url})
        self.in_link = False
        self.current_href = None
        self.current_text = []

def _clean_duckduckgo_url(value: str) -> str:
    value = html.unescape(value)
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return value

class WebSearchTool(BaseTool):
    name = "WebSearch"
    description = "Search the web using DuckDuckGo HTML search and return result titles and URLs."
    read_only = True
    input_schema = _schema(
        {
            "query": {"type": "string"},
            "timeout_ms": {"type": "number", "minimum": 1, "default": 10000},
        },
        required=["query"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        query = str(raw_input["query"])
        timeout = _timeout_seconds(raw_input)
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "SiYi/0.1"})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return _error(f"web search failed: {exc}", context)
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        results = parser.results
        content = "\n".join(f"{index + 1}. {item['title']}\n   {item['url']}" for index, item in enumerate(results))
        return _ok(content or "(no web results)", context, {"query": query, "results": results})

class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)

class WebFetchTool(BaseTool):
    name = "WebFetch"
    description = "Fetch a URL and return text content or raw response text."
    read_only = True
    input_schema = _schema(
        {
            "url": {"type": "string"},
            "raw": {"type": "boolean", "default": False},
            "max_chars": {"type": "integer", "minimum": 1, "default": 20000},
            "timeout_ms": {"type": "number", "minimum": 1, "default": 10000},
        },
        required=["url"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        url = str(raw_input["url"])
        max_chars = int(raw_input.get("max_chars") or context.max_result_chars)
        timeout = _timeout_seconds(raw_input)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "SiYi/0.1"})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return _error(f"web fetch failed: {exc}", context)
        content_type = response.headers.get("content-type", "")
        if raw_input.get("raw") or "html" not in content_type:
            text = response.text
        else:
            parser = _TextExtractor()
            parser.feed(response.text)
            text = "\n".join(parser.parts)
        return _ok(
            truncate_text(text, max_chars),
            context,
            {"url": str(response.url), "status_code": response.status_code, "content_type": content_type},
        )
