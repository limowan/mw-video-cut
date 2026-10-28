"""火山引擎逐词转录（豆包语音 · 大模型录音文件识别）

只用 Python 标准库 + 本机 FFmpeg，不需要安装任何包。
输出两份文件：
  逐词 JSON   —— cues[0].words：每个字的起止秒数；停顿用 isGap 空隙词表示（最长 1 秒一段）
  分句预览 TXT —— 停顿 ≥ --gap 秒断一句：`序号 [起-止] 这句话  (停 X 秒)`

凭证（不要写进代码、不要发在对话里），按顺序找：
  1. 环境变量 VOLC_ASR_API_KEY（必填），VOLC_ASR_RESOURCE_ID、VOLC_ASR_MODEL（可选）
  2. Skill 根目录 config.json 里的 volcAsr.apiKey / resourceId / modelName

用法：
  python volc_transcribe.py 素材/原片.mov --out 源文件/转录/火山逐词转录.json --preview 源文件/转录/转录_分句预览.txt
"""
import argparse, base64, json, os, subprocess, sys, tempfile, time, urllib.error, urllib.request, uuid
from pathlib import Path

SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
PUNCT = set("，。！？；：、,.!?;:…—")


def load_credentials():
    cfg = {}
    config_file = Path(__file__).resolve().parent.parent / "config.json"
    if config_file.exists():
        cfg = json.loads(config_file.read_text(encoding="utf-8")).get("volcAsr", {}) or {}
        # 示例里的中文占位文字不算填写
        cfg = {k: v for k, v in cfg.items() if isinstance(v, str) and v.strip() and v.isascii()}
    key = os.environ.get("VOLC_ASR_API_KEY") or cfg.get("apiKey")
    if not key:
        sys.exit("缺少火山引擎 API Key：设置环境变量 VOLC_ASR_API_KEY，或在 config.json 的 volcAsr.apiKey 里填写")
    resource = os.environ.get("VOLC_ASR_RESOURCE_ID") or cfg.get("resourceId") or "volc.seedasr.auc"
    model = os.environ.get("VOLC_ASR_MODEL") or cfg.get("modelName") or "bigmodel"
    return key, resource, model


def extract_audio(video, out_mp3):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", str(out_mp3)]
    subprocess.run(cmd, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out_mp3)],
                           check=True, capture_output=True, text=True)
    return float(probe.stdout.strip())


def call(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.headers.get("X-Api-Status-Code"), json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"火山接口 HTTP {e.code}：{e.read().decode('utf-8', 'ignore')[:300]}")


def timed_words(payload):
    result = payload.get("result", payload) if isinstance(payload, dict) else {}
    words = []
    for utt in result.get("utterances") or []:
        mine = []
        for w in utt.get("words") or []:
            text, s, e = (w.get("text") or "").strip(), w.get("start_time"), w.get("end_time")
            if text and isinstance(s, (int, float)) and isinstance(e, (int, float)) and 0 <= s < e:
                mine.append({"text": text, "start": s / 1000, "end": e / 1000})
        # 把整句文本里的标点挂到前一个字上（字幕断句用）
        text, cur = utt.get("text") or "", 0
        for w in mine:
            if not text.startswith(w["text"], cur):
                break
            cur += len(w["text"])
            marks = ""
            while cur < len(text) and (text[cur] in PUNCT or text[cur].isspace()):
                marks += text[cur].strip()
                cur += 1
            if marks:
                w["punctuation"] = marks
        words += mine
    return words


def transcribe(audio, language, key, resource, model):
    headers = {"Content-Type": "application/json", "X-Api-Key": key, "X-Api-Resource-Id": resource,
               "X-Api-Request-Id": str(uuid.uuid4()), "X-Api-Sequence": "-1"}
    body = {"user": {"uid": "mw-video-cut"},
            "audio": {"data": base64.b64encode(Path(audio).read_bytes()).decode(), "format": "mp3", "codec": "raw",
                      "rate": 16000, "bits": 16, "channel": 1, "language": language},
            "request": {"model_name": model, "enable_itn": True, "enable_punc": True, "show_utterances": True}}
    status, _ = call(SUBMIT_URL, headers, body)
    if status and status != "20000000":
        sys.exit(f"火山拒绝了转录请求，状态码 {status}（检查 Key、资源 ID、额度）")
    for _ in range(1200):
        time.sleep(2)
        status, payload = call(QUERY_URL, headers, {})
        words = timed_words(payload)
        if words:
            return words
        if status not in (None, "20000001", "20000002"):
            sys.exit(f"火山转录失败，状态码 {status}")
    sys.exit("火山转录超时")


def build_document(words, duration, language):
    out, last = [], 0.0
    def gap(a, b):
        if b - a <= 0.1:
            return
        while a < b - 0.0005:
            n = min(a + 1, b)
            out.append({"text": "", "start": round(a, 3), "end": round(n, 3), "isGap": True})
            a = n
    for w in words:
        gap(last, w["start"])
        item = {"text": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
        if w.get("punctuation"):
            item["punctuation"] = w["punctuation"]
        out.append(item)
        last = max(last, w["end"])
    gap(last, duration)
    return {"schemaVersion": 1, "provider": "volcengine", "language": language, "cues": [{"id": "volcengine-source", "words": out}]}


def preview(doc, min_gap):
    lines, cur, pause = [], [], 0.0
    def flush(p):
        if cur:
            lines.append(f"{len(lines) + 1:03d} [{cur[0]['start']:.2f}-{cur[-1]['end']:.2f}] "
                         + "".join(w["text"] + w.get("punctuation", "") for w in cur) + f"  (停 {p:.2f})")
    for w in doc["cues"][0]["words"]:
        if w.get("isGap"):
            pause += w["end"] - w["start"]
            continue
        if cur and pause >= min_gap:
            flush(pause)
            cur = []
        pause = 0.0
        cur.append(w)
    flush(pause)
    return "\n".join(lines) + "\n"


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="火山引擎逐词转录")
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="逐词 JSON 输出路径")
    ap.add_argument("--preview", help="分句预览 TXT 输出路径")
    ap.add_argument("--language", default="zh-CN")
    ap.add_argument("--gap", type=float, default=0.35, help="分句预览的断句停顿（秒）")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} 已存在，不覆盖；换个文件名或先手动处理旧文件")
    key, resource, model = load_credentials()
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = Path(tmp) / "audio.mp3"
        duration = extract_audio(a.video, mp3)
        print(f"音频 {duration:.1f} 秒，已上传火山引擎，等待转录……", flush=True)
        words = transcribe(mp3, a.language, key, resource, model)
    doc = build_document(words, duration, a.language)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n = sum(1 for w in doc["cues"][0]["words"] if not w.get("isGap"))
    print(f"完成：{n} 个字，时长 {duration:.1f} 秒 → {out}")
    if a.preview:
        Path(a.preview).write_text(preview(doc, a.gap), encoding="utf-8")
        print(f"分句预览 → {a.preview}")


if __name__ == "__main__":
    main()
