# ================================
# diar_asr.py — qwen3-pyannote-diar-asr
# - pyannote 話者分離 + Qwen3-ASR セグメント単位文字起こし
# - 辞書なし最小構成。VTT / TXT 出力。Docker ベース運用。
# ================================

import gc
import os
import sys
import time
import tempfile
import shutil
import warnings
import subprocess
import traceback
from pathlib import Path

warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt_vtt(t: float) -> str:
    """秒数から VTT 形式のタイムスタンプを生成する。24時間超でも正しく計算する。"""
    total_ms = int(round(float(t) * 1000))
    total_seconds, ms = divmod(total_ms, 1000)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ================================
# GPU メモリ解放
# ================================
def _log_gpu_memory(device: int, log_fn) -> None:
    if device == 0:
        try:
            import torch
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            log_fn(f"[VRAM] Allocated={alloc:.2f} MB, Reserved={reserved:.2f} MB")
        except Exception:
            pass


def _release_gpu_memory(device: int, log_fn) -> None:
    log_fn("[STEP] GPUメモリを解放...")
    _log_gpu_memory(device, log_fn)
    gc.collect()
    if device == 0:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    _log_gpu_memory(device, log_fn)
    log_fn("[OK] GPU 解放完了")


# ================================
# Torch import + PyTorch 2.6+ workaround (pyannote 用)
# TORCH_LOAD_WEIGHTS_ONLY=0 のとき weights_only=False でパッチし、互換性を確保する。
# ================================
import torch

# pyannote 未ロード時でも TorchVersion を safe globals に登録する（torch.load 互換のため）
try:
    from torch.serialization import add_safe_globals
    add_safe_globals([torch.torch_version.TorchVersion])
except Exception:
    pass

# pyannote ロード時に必要になるクラスを safe globals に登録する
try:
    from torch.serialization import add_safe_globals
    from pyannote.audio.core.task import Specifications, Problem
    add_safe_globals([Specifications, Problem])
except Exception:
    pass

_force_legacy = os.getenv("TORCH_LOAD_WEIGHTS_ONLY", "0").strip().lower() in ("0", "false", "no", "off")
if _force_legacy:
    _orig_torch_load = torch.load
    def _torch_load_legacy(*args, **kwargs):
        kwargs.pop("weights_only", None)
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_legacy
    log("[INFO] Patched torch.load(weights_only=False) for PyTorch 2.6+ compatibility.")
else:
    log("[INFO] torch.load patch disabled (TORCH_LOAD_WEIGHTS_ONLY!=0).")


def main() -> None:
    # ================================
    # Paths (Docker volume mount 前提)
    # ================================
    work_source = Path(os.getenv("WORK_SOURCE", "/work/source"))
    work_output_env = os.getenv("WORK_OUTPUT", "").strip()
    work_output = Path(work_output_env) if work_output_env else Path("/work/output")
    work_hf = Path(os.getenv("WORK_HF_CACHE", "/work/hf_cache"))
    work_tmp = Path(os.getenv("WORK_TMP", "/work/tmp"))

    work_output.mkdir(parents=True, exist_ok=True)
    work_hf.mkdir(parents=True, exist_ok=True)
    work_tmp.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(work_hf))
    os.environ.setdefault("HF_HUB_CACHE", str(work_hf / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(work_hf / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(work_hf))
    os.environ.setdefault("TMPDIR", str(work_tmp))
    os.environ.setdefault("TEMP", str(work_tmp))
    os.environ.setdefault("TMP", str(work_tmp))

    # ================================
    # HF login（未設定時は明示的に終了）
    # ================================
    from huggingface_hub import login, HfApi

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        log("[FATAL] 環境変数 HF_TOKEN が未設定です。初回実行時は必須です。")
        log("  docker run に -e HF_TOKEN=$env:HF_TOKEN を付けるか、Windows のユーザー環境変数に HF_TOKEN を設定してください。")
        sys.exit(1)
    login(token=hf_token)

    try:
        who = HfApi().whoami(token=hf_token)
        who_str = who.get("name") or who.get("email") or who.get("username") or "Unknown"
        log(f"[OK] HF login: {who_str}")
    except Exception:
        log("[WARN] whoami 取得に失敗。すでに認証済みであれば問題ありません。")

    # ================================
    # Settings
    # ================================
    input_filename = os.getenv("INPUT_FILENAME", "input.m4a")
    input_file = work_source / input_filename

    num_speakers_env = os.getenv("NUM_SPEAKERS", "").strip()
    if num_speakers_env in ("", "none", "null", "auto"):
        num_speakers = None
    else:
        try:
            num_speakers = int(num_speakers_env)
        except ValueError:
            raise ValueError(
                f"NUM_SPEAKERS には数値または auto を指定してください。"
                f"現在の値: '{num_speakers_env}'"
            )

    model_diar = os.getenv("MODEL_DIAR", "pyannote/speaker-diarization-3.1")
    model_asr = os.getenv("MODEL_ASR", "Qwen/Qwen3-ASR-1.7B")
    asr_language = os.getenv("ASR_LANGUAGE", "Japanese")
    asr_max_new_tokens = int(os.getenv("ASR_MAX_NEW_TOKENS", "256"))

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    log(f"[INFO] device={'cuda:0' if device == 0 else 'cpu'}, torch_dtype={torch_dtype}")

    log(f"[CFG] WORK_SOURCE={work_source}")
    log(f"[CFG] WORK_OUTPUT={work_output}")
    log(f"[CFG] INPUT_FILENAME='{input_filename}'")
    log(f"[CFG] NUM_SPEAKERS_ENV='{num_speakers_env}' -> NUM_SPEAKERS={num_speakers}")
    log(f"[CFG] MODEL_DIAR={model_diar}")
    log(f"[CFG] MODEL_ASR={model_asr}")
    log(f"[CFG] ASR_MAX_NEW_TOKENS={asr_max_new_tokens}")

    if not input_file.exists():
        raise FileNotFoundError(f"入力が見つかりません: {input_file}")

    # ================================
    # Imports
    # ================================
    from pyannote.audio import Pipeline
    from pydub import AudioSegment
    from tqdm.auto import tqdm

    # ================================
    # Convert to WAV (16k/mono)
    # ================================
    log("[STEP] 入力を WAV(16kHz mono) に変換...")
    tmp_in_dir = tempfile.mkdtemp(prefix="wav_", dir=str(work_tmp))
    wav16 = Path(tmp_in_dir) / "input_16k_mono.wav"

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-y",
        "-i", str(input_file),
        "-ac", "1", "-ar", "16000",
        str(wav16),
    ]
    subprocess.run(cmd, check=True)
    audio16 = AudioSegment.from_file(str(wav16))
    log("[OK] 変換完了")

    # ================================
    # Diarization
    # ================================
    dia = None
    log("[STEP] 話者分離モデルをロード...")
    try:
        # TODO: 将来的に pyannote が token= を標準化したら use_auth_token から移行する
        dia = Pipeline.from_pretrained(model_diar, use_auth_token=hf_token)
        if dia is None:
            raise RuntimeError("Pipelineのロードに失敗しました。モデル名やHF_TOKENの権限を確認してください。")
        if device == 0:
            dia.to(torch.device("cuda"))
        if hasattr(dia, "embedding_batch_size"):
            dia.embedding_batch_size = 1
    except Exception:
        log(f"[FATAL ERROR] モデルロード中に例外発生:\n{traceback.format_exc()}")
        raise

    log("[STEP] 話者分離を実行...")
    t0 = time.time()
    try:
        if device == 0:
            log(f"[VRAM] Allocated before DIAR: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        if num_speakers is None:
            diar = dia(str(wav16))
        else:
            diar = dia(
                str(wav16),
                num_speakers=int(num_speakers),
                min_speakers=int(num_speakers),
                max_speakers=int(num_speakers),
            )
        segments = list(diar.itertracks(yield_label=True))
        log(f"[OK] 話者分離: {len(segments)} セグメント ({time.time()-t0:.2f}s)")
        labels = sorted({lbl for _, _, lbl in segments})
        log(f"[DIAR] labels={labels} N={len(labels)}")
    except Exception:
        log(f"[FATAL ERROR] 話者分離の実行中に例外発生:\n{traceback.format_exc()}")
        if dia is not None:
            dia = None
        _release_gpu_memory(device, log)
        raise

    # 話者分離終了後、ASR 開始前に GPU メモリを解放
    if dia is not None:
        dia = None
    _release_gpu_memory(device, log)

    # ================================
    # Qwen3-ASR ロード（pyannote 解放後に GPU メモリを空けてから読み込む）
    # ================================
    log("[STEP] Qwen3-ASR をロード...")
    try:
        from qwen_asr import Qwen3ASRModel

        asr_model = Qwen3ASRModel.from_pretrained(
            model_asr,
            dtype=torch_dtype,
            device_map="cuda:0" if device == 0 else "cpu",
            max_inference_batch_size=1,
            max_new_tokens=asr_max_new_tokens,
        )
    except Exception:
        log(f"[FATAL ERROR] Qwen3-ASR ロード中に例外発生:\n{traceback.format_exc()}")
        raise

    # ================================
    # セグメント単位 ASR
    # ================================
    log("[STEP] セグメントごとにASR中...")
    tmp_seg_dir = tempfile.mkdtemp(prefix="seg_", dir=str(work_tmp))
    results = []

    try:
        for idx, (turn, _, spk) in enumerate(tqdm(segments, desc="ASR", leave=True), 1):
            st, ed = float(turn.start), float(turn.end)
            if ed <= st:
                continue

            seg = audio16[int(st * 1000) : int(ed * 1000)]
            seg_path = Path(tmp_seg_dir) / f"seg_{idx:04d}.wav"
            seg.export(str(seg_path), format="wav")

            # ASR 失敗時は空文字のまま結果に含める（セグメントはスキップしない。後段の突合・補正で扱いやすい）
            try:
                out_list = asr_model.transcribe(
                    audio=str(seg_path),
                    language=asr_language,
                )
                text = (out_list[0].text or "").strip() if out_list else ""
            except Exception as e_asr:
                log(f"[WARN] セグメント {idx} ASR 失敗: {e_asr}")
                text = ""

            results.append({"speaker": spk, "start": st, "end": ed, "text": text})
    except Exception:
        log(f"[FATAL ERROR] ASR推論中に例外発生:\n{traceback.format_exc()}")
        raise

    # ASR 完了後も GPU メモリを解放しておく（一貫性・将来の追加処理や切り分けのため）
    _release_gpu_memory(device, log)

    # ================================
    # Write outputs
    # ================================
    base = input_file.stem
    vtt_lines = ["WEBVTT\n"]
    for i, r in enumerate(results, 1):
        vtt_lines.append(
            f"{i}\n{fmt_vtt(r['start'])} --> {fmt_vtt(r['end'])}\n<v {r['speaker']}>{r['text']}</v>\n"
        )
    (work_output / f"{base}.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")

    with (work_output / f"{base}.txt").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[{r['speaker']}] {r['text']}\n")

    # ================================
    # Cleanup
    # ================================
    shutil.rmtree(tmp_seg_dir, ignore_errors=True)
    shutil.rmtree(tmp_in_dir, ignore_errors=True)

    log(f"[OK] 出力: {work_output}")
    log("[DONE] 完了")


if __name__ == "__main__":
    main()
