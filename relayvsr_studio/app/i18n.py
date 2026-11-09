"""Minimal English / Korean string table."""

from __future__ import annotations

import locale
import os

_LANG = "en"

STRINGS: dict[str, dict[str, str]] = {
    "en": {},
    "ko": {
        "RelayVSR Studio": "RelayVSR Studio",
        "Queue": "작업 목록",
        "Add files…": "파일 추가…",
        "Add folder…": "폴더 추가…",
        "Remove": "제거",
        "Clear finished": "완료 항목 지우기",
        "Drop videos or images here": "여기에 동영상이나 이미지를 끌어다 놓으세요",
        "Model": "모델",
        "4x (RelayVSR)": "4배 (RelayVSR)",
        "2x (RelayVSR_X2)": "2배 (RelayVSR_X2)",
        "Not downloaded": "미다운로드",
        "Installed": "설치됨",
        "Output": "출력",
        "Output size": "출력 크기",
        "Native scale": "모델 기본 배율",
        "Fit 1080p": "1080p에 맞춤",
        "Fit 1440p": "1440p에 맞춤",
        "Fit 4K (2160p)": "4K(2160p)에 맞춤",
        "Custom width": "사용자 지정 너비",
        "Width": "너비",
        "Output folder": "출력 폴더",
        "Same as input": "입력 파일과 같은 폴더",
        "Browse…": "찾아보기…",
        "Encoder": "인코더",
        "Codec": "코덱",
        "Quality (CRF / CQ)": "품질 (CRF / CQ)",
        "Preset": "프리셋",
        "Copy audio": "오디오 복사",
        "Overwrite existing files": "기존 파일 덮어쓰기",
        "Advanced": "고급",
        "Wan weights": "Wan 가중치",
        "Auto": "자동",
        "VRAM strategy": "VRAM 전략",
        "Full frame": "전체 프레임",
        "Tiled keyframes": "키프레임 타일링",
        "Block offload": "블록 오프로드 (CPU)",
        "Attention kernels": "어텐션 커널",
        "Keyframe interval": "키프레임 간격",
        "Seed": "시드",
        "Detail strength": "디테일 강도",
        "Keyframe KV window": "키프레임 KV 윈도우",
        "Frame range": "프레임 범위",
        "Start frame": "시작 프레임",
        "Max frames": "최대 프레임 수",
        "all": "전체",
        "Show previews": "미리보기 표시",
        "Start": "시작",
        "Cancel": "취소",
        "Open output folder": "출력 폴더 열기",
        "Log": "로그",
        "Ready": "준비됨",
        "Environment": "환경",
        "Settings": "설정",
        "Environment setup": "환경 설정",
        "Install / repair engine": "엔진 설치 / 복구",
        "Download": "다운로드",
        "Downloading…": "다운로드 중…",
        "Models": "모델 파일",
        "Models folder": "모델 폴더",
        "Engine folder": "엔진 폴더",
        "Close": "닫기",
        "The inference engine is not installed yet. Install it now? (downloads about 3.5 GB)": "추론 엔진이 아직 설치되지 않았습니다. 지금 설치할까요? (약 3.5 GB 다운로드)",
        "Engine ready": "엔진 준비됨",
        "Engine not installed": "엔진 미설치",
        "Checking environment…": "환경 확인 중…",
        "Waiting": "대기",
        "Running": "진행 중",
        "Done": "완료",
        "Failed": "실패",
        "Cancelled": "취소됨",
        "Loading models…": "모델 로딩 중…",
        "Before": "입력",
        "After": "출력",
        "frames": "프레임",
        "Status": "상태",
        "Input": "입력",
        "Language": "언어",
        "External ffmpeg path (optional)": "외부 ffmpeg 경로 (선택)",
        "Use external ffmpeg for encoding": "인코딩에 외부 ffmpeg 사용",
        "Engine": "엔진",
        "GPU": "GPU",
        "No CUDA GPU detected": "CUDA GPU를 찾을 수 없습니다",
        "Add at least one input file.": "입력 파일을 하나 이상 추가하세요.",
        "Model files for this scale are not downloaded yet. Download them now?": "이 배율의 모델 파일이 아직 없습니다. 지금 다운로드할까요?",
        "Install the engine first (Environment).": "먼저 엔진을 설치하세요 (환경 메뉴).",
        "Free disk space": "여유 디스크 공간",
        "About": "정보",
        "Quality first (BF16 if it fits)": "품질 우선 (가능하면 BF16)",
        "Low VRAM": "저용량 VRAM",
        "Engine process exited unexpectedly": "엔진 프로세스가 예기치 않게 종료되었습니다",
        "Estimated VRAM": "예상 VRAM",
        "Plan": "실행 계획",
        "Keyframes": "키프레임",
        "elapsed": "경과",
        "ETA": "남은 시간",
        "peak": "최대",
        "Output written to": "출력 저장 위치",
        "Setup log": "설치 로그",
        "Install location": "설치 위치",
        "Python environment": "파이썬 환경",
        "Triton kernels": "Triton 커널",
        "available": "사용 가능",
        "unavailable (SDPA fallback)": "사용 불가 (SDPA 대체 경로)",
        "Apply": "적용",
        "Remove environment": "환경 삭제",
        "Pause after current": "현재 작업 후 일시정지",
        "Open": "열기",
        "Preset: RTX 50 series (Blackwell) tuned": "프리셋: RTX 50 시리즈(Blackwell) 최적화",
    },
}


def set_language(lang: str | None) -> None:
    global _LANG
    if lang in (None, "", "auto"):
        code = os.environ.get("RELAYVSR_LANG") or (locale.getlocale()[0] or "")
        try:
            import ctypes

            windll = getattr(ctypes, "windll", None)
            if windll is not None:
                lcid = windll.kernel32.GetUserDefaultUILanguage()
                if lcid & 0xFF == 0x12:
                    code = "ko"
        except Exception:  # noqa: BLE001
            pass
        _LANG = "ko" if str(code).lower().startswith("ko") else "en"
    else:
        _LANG = lang if lang in STRINGS else "en"


def language() -> str:
    return _LANG


def tr(text: str) -> str:
    return STRINGS.get(_LANG, {}).get(text, text)


__all__ = ["language", "set_language", "tr"]
