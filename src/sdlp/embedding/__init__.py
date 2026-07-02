"""임베딩 패키지 — ST 백엔드 + 임베딩 캐시.

ponytail: 지금은 ST 백엔드 하나뿐이라 factory(build_embedder) 없이 직접 STTextEmbedder 사용.
          P-SP 등 2번째 백엔드가 생기면 그때 factory 추가.
"""
from sdlp.embedding.spec import EmbedSpec
from sdlp.embedding.st import STTextEmbedder, resolve_device

__all__ = ["EmbedSpec", "STTextEmbedder", "resolve_device"]
