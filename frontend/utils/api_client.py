import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
import streamlit as st

# Ensure project root is on sys.path for embedded fallback mode
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings


def _normalize_backend_url(raw_url: str) -> str:
    """
    Replace 'http://localhost:' with 'http://127.0.0.1:' so Windows IPv6 [::1]
    TCP SYN timeouts (~2.05s per new connection) are completely avoided.
    """
    cleaned = raw_url.rstrip("/")
    if cleaned.startswith("http://localhost:"):
        cleaned = cleaned.replace("http://localhost:", "http://127.0.0.1:", 1)
    elif cleaned == "http://localhost":
        cleaned = "http://127.0.0.1"
    return cleaned


@st.cache_resource
def _get_shared_http_session() -> requests.Session:
    """Return a persistent thread-safe HTTP keep-alive session."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class BackendAPIClient:
    """
    High-performance HTTP client for communicating with the FastAPI backend.
    Features:
    - Persistent HTTP keep-alive connection pooling via requests.Session()
    - IPv4 127.0.0.1 normalization to avoid Windows localhost IPv6 2s timeouts
    - Streamlit @st.cache_data read caching with automatic cache invalidation on writes
    - Embedded FastAPI TestClient fallback if external server is unreachable
    """

    def __init__(self, base_url: Optional[str] = None):
        raw = base_url or os.getenv("BACKEND_URL") or settings.BACKEND_URL
        self.base_url = _normalize_backend_url(raw)
        self._session = _get_shared_http_session()
        self._embedded_client = None
        self.mode = "http"

    def _get_embedded_client(self):
        if self._embedded_client is None:
            from fastapi.testclient import TestClient
            from backend.app.main import app

            self._embedded_client = TestClient(app)
        return self._embedded_client

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        timeout = kwargs.pop("timeout", 60)
        try:
            resp = self._session.request(method, url, timeout=timeout, **kwargs)
            self.mode = "http"
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            self.mode = "embedded"
            client = self._get_embedded_client()
            resp = client.request(method, path, **kwargs)
            if resp.status_code >= 400:
                raise RuntimeError(resp.json().get("detail", f"HTTP {resp.status_code}"))
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            detail = str(exc)
            try:
                detail = exc.response.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(detail) from exc

    @staticmethod
    def invalidate_caches() -> None:
        """Invalidate all cached read data after any state-mutating operation."""
        _cached_health_stats.clear()
        _cached_list_documents.clear()
        _cached_document_details.clear()
        _cached_list_sessions.clear()
        _cached_chat_history.clear()
        _cached_latest_evaluation.clear()

    def get_health_stats(self, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache:
            return _cached_health_stats(self.base_url)
        return self._request("GET", "/health")

    def list_documents(
        self, status_filter: Optional[str] = None, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        if use_cache:
            return _cached_list_documents(self.base_url, status_filter)
        params = {"status_filter": status_filter} if status_filter else {}
        return self._request("GET", "/documents", params=params)

    def get_document_details(
        self, document_id: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        if use_cache:
            return _cached_document_details(self.base_url, document_id)
        return self._request("GET", f"/documents/{document_id}")

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        res = self._request("DELETE", f"/documents/{document_id}")
        self.invalidate_caches()
        return res

    def upload_documents(
        self,
        files_payload: List[Tuple[str, bytes, str]],
        chunk_size: int = 350,
        chunk_overlap: int = 50,
    ) -> Dict[str, Any]:
        multipart_files = [
            ("files", (fname, fbytes, ftype))
            for fname, fbytes, ftype in files_payload
        ]
        data = {
            "chunk_size": str(chunk_size),
            "chunk_overlap": str(chunk_overlap),
        }
        res = self._request("POST", "/documents/upload", files=multipart_files, data=data)
        self.invalidate_caches()
        return res

    def ask_question(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        res = self._request("POST", "/chat/ask", json=payload)
        self.invalidate_caches()
        return res

    def list_sessions(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        if use_cache:
            return _cached_list_sessions(self.base_url)
        return self._request("GET", "/chat/sessions")

    def create_session(self, title: str = "New Conversation") -> Dict[str, Any]:
        res = self._request("POST", "/chat/sessions", params={"title": title})
        self.invalidate_caches()
        return res

    def get_chat_history(
        self, session_id: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        if use_cache:
            return _cached_chat_history(self.base_url, session_id)
        return self._request("GET", f"/chat/history/{session_id}")

    def delete_chat_history(self, session_id: str) -> Dict[str, Any]:
        res = self._request("DELETE", f"/chat/history/{session_id}")
        self.invalidate_caches()
        return res

    def search_retrieval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/retrieval/search", json=payload)

    def run_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        res = self._request("POST", "/evaluation/run", json=payload, timeout=180)
        self.invalidate_caches()
        return res

    def get_latest_evaluation(self, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        if use_cache:
            return _cached_latest_evaluation(self.base_url)
        return self._request("GET", "/evaluation/latest")


@st.cache_data(ttl=10, show_spinner=False)
def _cached_health_stats(base_url: str) -> Dict[str, Any]:
    client = BackendAPIClient(base_url=base_url)
    return client._request("GET", "/health")


@st.cache_data(ttl=10, show_spinner=False)
def _cached_list_documents(
    base_url: str, status_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    client = BackendAPIClient(base_url=base_url)
    params = {"status_filter": status_filter} if status_filter else {}
    return client._request("GET", "/documents", params=params)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_document_details(base_url: str, document_id: str) -> Dict[str, Any]:
    client = BackendAPIClient(base_url=base_url)
    return client._request("GET", f"/documents/{document_id}")


@st.cache_data(ttl=10, show_spinner=False)
def _cached_list_sessions(base_url: str) -> List[Dict[str, Any]]:
    client = BackendAPIClient(base_url=base_url)
    return client._request("GET", "/chat/sessions")


@st.cache_data(ttl=10, show_spinner=False)
def _cached_chat_history(base_url: str, session_id: str) -> Dict[str, Any]:
    client = BackendAPIClient(base_url=base_url)
    return client._request("GET", f"/chat/history/{session_id}")


@st.cache_data(ttl=30, show_spinner=False)
def _cached_latest_evaluation(base_url: str) -> Optional[Dict[str, Any]]:
    client = BackendAPIClient(base_url=base_url)
    return client._request("GET", "/evaluation/latest")
