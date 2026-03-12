# qwen3-pyannote-diar-asr

Windows 11 + Docker Desktop（WSL2）+ NVIDIA GPU 環境で、**pyannote による話者分離**と **Qwen3-ASR による日本語文字起こし**を連携して実行する Docker ベースのツールである。

このリポジトリは、次の一連の処理を行う。

1. 入力音声を **16kHz / mono の WAV** に変換する  
2. `pyannote/speaker-diarization-3.1` で **話者分離**を行う  
3. 分割した各区間を **Qwen3-ASR**（例: `Qwen/Qwen3-ASR-1.7B`）で **文字起こし**する  
4. 話者ラベル付きの **TXT** と **VTT** を出力する  

> 注意: 音声ファイル、文字起こし結果、ログ、キャッシュ、トークンは Git にコミットしてはいけない。個人情報・機密情報が含まれやすいためである。本リポジトリは、それらを Git 管理外に置く前提で設計している。

---

## 前提環境

- Windows 11
- Docker Desktop（WSL2 backend）
- NVIDIA GPU + 比較的新しいドライバ
- PowerShell

### 簡易動作確認

```powershell
docker context show
wsl -l -v
nvidia-smi

# Docker 上で GPU が見えているか確認
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

## 推奨ディレクトリ構成

モデルキャッシュや一時 WAV により I/O が増えるため、保存先によって体感速度が変わる。

* **SSD 強推奨**: Hugging Face キャッシュ（`hf_cache`）、一時ファイル（`tmp`）
* **HDD でも可**: 入力音声、出力テキスト

例:

* 入力/出力: `D:\asr\work\source`
* HF キャッシュ: `D:\asr\hf_cache`
* 一時ファイル: `D:\asr\tmp`

```powershell
$base = "D:\asr"
New-Item -ItemType Directory -Force -Path "$base\work\source" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\work\output" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\hf_cache" | Out-Null
New-Item -ItemType Directory -Force -Path "$base\tmp" | Out-Null
```

---

## Hugging Face トークン（HF_TOKEN）

`pyannote/speaker-diarization-3.1` は Hugging Face 上で利用規約への同意が必要なことが多い。事前にブラウザでログインし、モデルページで利用規約に同意しておくこと。

```powershell
$token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
[Environment]::SetEnvironmentVariable("HF_TOKEN", $token, "User")
# 新しい PowerShell を開き直して確認
$env:HF_TOKEN
```

---

## ビルド

```powershell
cd <このリポジトリのフォルダ>
docker build -t qwen3-diar-asr:cu126 .
```

---

## 実行

入力音声（例: `meeting_sample.m4a`）を `D:\asr\work\source\` に置く。

```powershell
docker run --rm --gpus all `
  -e HF_TOKEN="$env:HF_TOKEN" `
  -e INPUT_FILENAME="meeting_sample.m4a" `
  -e NUM_SPEAKERS="2" `
  -v D:\asr\work\source:/work/source `
  -v D:\asr\work\output:/work/output `
  -v D:\asr\hf_cache:/work/hf_cache `
  -v D:\asr\tmp:/work/tmp `
  -e WORK_OUTPUT="/work/output" `
  qwen3-diar-asr:cu126
```

`WORK_OUTPUT` を省略した場合は、既定で `WORK_SOURCE/output` を使う。

出力例:

* `D:\asr\work\output\meeting_sample.txt`
* `D:\asr\work\output\meeting_sample.vtt`

---

## 環境変数

### 基本

* `HF_TOKEN`: Hugging Face トークン
* `INPUT_FILENAME`: 入力音声ファイル名
* `NUM_SPEAKERS`: 話者数（例: `2`、空または `auto` で自動推定）
* `MODEL_DIAR`: 話者分離モデル（既定: `pyannote/speaker-diarization-3.1`）
* `MODEL_ASR`: ASR モデル（既定: `Qwen/Qwen3-ASR-1.7B`）
* `ASR_LANGUAGE`: ASR 言語（既定: `Japanese`）

### パス関連

* `WORK_SOURCE`: 入力ディレクトリ（既定: `/work/source`）
* `WORK_OUTPUT`: 出力ディレクトリ（既定: `WORK_SOURCE/output`）
* `WORK_HF_CACHE`: Hugging Face キャッシュ（既定: `/work/hf_cache`）
* `WORK_TMP`: 一時ファイル保存先（既定: `/work/tmp`）

### デバッグ用

* `TORCH_LOAD_WEIGHTS_ONLY`: `1` で `torch.load(..., weights_only=False)` パッチを無効化

---

## ログ

コンソールには進行状況、VRAM 使用量、話者分離セグメント数、ASR 進捗、出力パス、例外時の traceback を出力する。

---

## 補足

* 話者分離後に GPU メモリ解放を行い、ASR 実行前の VRAM 圧迫を抑えている
* 一部環境でのモデルロード問題に備え、`torch.load(..., weights_only=False)` の互換パッチを含む
* 音声や文字起こしは個人情報・機密情報を含みやすいため、リポジトリ外で管理することを推奨する
* 固有名詞の補正や文脈補正は、後段の Zoom VTT 突合や LLM 修正（例: TranscriptMerger）に委ねる想定である

---

## README (English)

**qwen3-pyannote-diar-asr** is a Docker-based tool for **speaker diarization (pyannote)** and **Japanese ASR (Qwen3-ASR)** on Windows 11 + Docker Desktop (WSL2) + NVIDIA GPU. It converts input audio to 16kHz mono WAV, runs pyannote diarization, transcribes each segment with Qwen3-ASR, and outputs speaker-labeled TXT and VTT. Dictionary and Layer A/B features are not included; the design is minimal for stable runs. See the Japanese section above for build/run and environment variables.
