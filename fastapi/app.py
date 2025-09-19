"""Simplified ASGI-like application emulating FastAPI for tests."""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - test environment always includes pydantic
    BaseModel = object  # type: ignore

try:  # pragma: no cover - shim always available in tests
    from fastapi.responses import StreamingResponse as ShimStreamingResponse
except ImportError:  # pragma: no cover - fallback when responses module missing
    ShimStreamingResponse = None  # type: ignore


class HTTPException(Exception):
    """Exception carrying an HTTP status code and payload."""

    def __init__(self, status_code: int, detail: str | Mapping[str, Any]) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class _Route:
    method: str
    path: str
    handler: Callable[..., Any]
    response_model: Any = None
    response_class: Any = None

    def match(self, request_path: str) -> Optional[Dict[str, str]]:
        pattern = re.sub(r"{([^}]+)}", r"(?P<\1>[^/]+)", self.path)
        match = re.fullmatch(pattern, request_path)
        if match:
            return match.groupdict()
        return None


class FastAPI:
    """Very small subset of the FastAPI interface used in tests."""

    def __init__(self, title: str = "FastAPI", version: str = "0.0.0") -> None:
        self.title = title
        self.version = version
        self._routes: List[_Route] = []
        self._mounts: List[Tuple[str, object]] = []

    # Route registration -------------------------------------------------
    def get(
        self,
        path: str,
        response_model: Any | None = None,
        response_class: Any | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("GET", path, response_model, response_class)

    def post(
        self,
        path: str,
        response_model: Any | None = None,
        response_class: Any | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("POST", path, response_model, response_class)

    def put(
        self,
        path: str,
        response_model: Any | None = None,
        response_class: Any | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("PUT", path, response_model, response_class)

    def delete(
        self,
        path: str,
        response_model: Any | None = None,
        response_class: Any | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("DELETE", path, response_model, response_class)

    def _add_route(
        self,
        method: str,
        path: str,
        response_model: Any | None,
        response_class: Any | None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append(
                _Route(method=method, path=path, handler=func, response_model=response_model, response_class=response_class)
            )
            return func

        return decorator

    def mount(self, path: str, app: object, name: str | None = None) -> None:  # pragma: no cover - behaviour unused in tests
        self._mounts.append((path, app))

    # Request handling ---------------------------------------------------
    def handle_request(self, method: str, path: str, body: Any | None = None) -> "Response":
        path_only, _, query_string = path.partition("?")
        query_params = {
            key: values[0] if values else None
            for key, values in parse_qs(query_string, keep_blank_values=True).items()
        }
        for route in self._routes:
            if route.method != method:
                continue
            path_params = route.match(path_only)
            if path_params is None:
                continue
            try:
                payload = self._build_payload(route.handler, {**query_params, **path_params}, body)
                result = route.handler(**payload)
                return _normalise_response(result, route.response_class)
            except HTTPException as exc:
                return Response(status_code=exc.status_code, data={"detail": exc.detail})
        return Response(status_code=404, data={"detail": "Not Found"})

    def _build_payload(
        self,
        handler: Callable[..., Any],
        path_params: Mapping[str, str],
        body: Any | None,
    ) -> Dict[str, Any]:
        signature = inspect.signature(handler)
        arguments: Dict[str, Any] = dict(path_params)
        if body is None:
            return arguments
        for name, parameter in signature.parameters.items():
            if name in arguments:
                continue
            annotation = parameter.annotation
            if annotation is inspect.Signature.empty:
                arguments[name] = body
                continue
            if isinstance(annotation, type) and _is_pydantic_model(annotation):
                arguments[name] = annotation(**body)
            elif isinstance(annotation, type) and is_dataclass(annotation):
                arguments[name] = annotation(**body)
            else:
                arguments[name] = body
        return arguments


@dataclass
class Response:
    status_code: int
    data: Any

    def json(self) -> Any:
        return self.data

    @property
    def text(self) -> str:
        if isinstance(self.data, str):
            return self.data
        return str(self.data)


def _normalise_response(result: Any, response_class: Any | None) -> Response:
    if isinstance(result, Response):
        return result
    if ShimStreamingResponse is not None and isinstance(result, ShimStreamingResponse):
        body = "".join(str(part) for part in result)
        return Response(status_code=200, data=body)
    if is_dataclass(result):
        return Response(status_code=200, data=asdict(result))
    if BaseModel is not object and isinstance(result, BaseModel):  # type: ignore[isinstance]
        return Response(status_code=200, data=result.dict())
    if hasattr(result, "to_dict") and callable(result.to_dict):
        return Response(status_code=200, data=result.to_dict())
    if hasattr(result, "dict") and callable(result.dict):
        return Response(status_code=200, data=result.dict())
    if isinstance(result, (dict, list, str)):
        return Response(status_code=200, data=result)
    return Response(status_code=200, data=result)


def _is_pydantic_model(cls: Any) -> bool:
    return BaseModel is not object and isinstance(cls, type) and issubclass(cls, BaseModel)


__all__ = ["FastAPI", "HTTPException", "Response"]
