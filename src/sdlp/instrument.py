"""실험 계측 유틸 — 논문용 수치(latency / throughput / 벡터 인덱스 크기) 측정.

★ Latency 산정 규칙: 인덱스 생성 시간은 추론 latency 에 포함하지 않는다.
   추론 latency = 쿼리 임베딩 + 인덱스 검색 + 투표(+판별). 인덱스 빌드는 별도 필드로만 기록.
ponytail: 원시 측정 함수만. metrics.json 조립은 파이프라인이 생기는 S8 에서.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


# timer() 결과를 담는 홀더. 블록 종료 후 .sec 에 소요시간(초)이 채워짐.
@dataclass
class _Elapsed:
    sec: float = 0.0


# with 블록의 실행 시간을 초 단위로 재는 컨텍스트 매니저.
# 사용: `with timer() as t: ...` → `t.sec` 로 소요시간 접근.
@contextmanager
def timer():
    e = _Elapsed()
    start = perf_counter()
    try:
        yield e
    finally:
        e.sec = perf_counter() - start


# 초당 처리량(items/sec). 시간이 0 이하면 0 반환.
def throughput(n_items: int, seconds: float) -> float:
    return n_items / seconds if seconds > 0 else 0.0


# 벡터 인덱스의 on-disk 크기(MB). 파일이면 그 크기, 디렉터리면 하위 파일 합.
def index_size_mb(path: str | Path) -> float:
    path = Path(path)
    if path.is_file():
        total = path.stat().st_size
    else:
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024 * 1024)
