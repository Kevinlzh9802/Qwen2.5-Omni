#!/usr/bin/env python3
"""Batch inference with Qwen2.5-Omni over a folder of video/audio pairs.

Expected data layout
====================
<data_root>/
    subfolder_A/
        clip_01.mp4
        clip_01.wav
        clip_02.mp4
        clip_02.wav
    subfolder_B/
        ...

The script discovers every {.mp4, .wav} pair that shares the same stem
inside each immediate subfolder of *data_root*, feeds each pair to the
model together with a user-defined prompt template, and writes the
collected responses to a single JSON file.

Output JSON structure
=====================
{
    "subfolder_A": [
        {"file": "clip_01", "response": "..."},
        {"file": "clip_02", "response": "..."}
    ],
    "subfolder_B": [...]
}
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


# ---------------------------------------------------------------------------
# Folder traversal
# ---------------------------------------------------------------------------

def collect_av_pairs(data_root: str) -> Dict[str, List[dict]]:
    """Scan *data_root* for video/audio pairs.

    For every immediate subfolder in *data_root*, find all `.mp4` files and
    check whether a matching `.wav` file with the same stem exists.  Only
    complete pairs are returned.

    Returns
    -------
    dict
        Mapping from subfolder name to a sorted list of dicts, each with
        keys ``"stem"``, ``"video"``, ``"audio"`` (absolute paths).
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: Dict[str, List[dict]] = {}

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue

        # Collect all mp4 stems
        video_files = {f.stem: f for f in sorted(subfolder.glob("*.mp4"))}
        pairs: List[dict] = []

        for stem, video_path in sorted(video_files.items()):
            audio_path = subfolder / f"{stem}.wav"
            if audio_path.exists():
                pairs.append(
                    {
                        "stem": stem,
                        "video": str(video_path),
                        "audio": str(audio_path),
                    }
                )
            else:
                print(f"[WARN] No matching .wav for {video_path}, skipping.")

        if pairs:
            result[subfolder.name] = pairs

    return result


# ---------------------------------------------------------------------------
# Single-pair inference
# ---------------------------------------------------------------------------

def infer_single(
    model,
    processor,
    video_path: str,
    audio_path: str,
    prompt: str,
    system_prompt: str,
    use_audio_in_video: bool,
    max_new_tokens: int,
) -> str:
    """Run a single-turn inference for one video/audio pair and return the
    model's text response."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(
        messages, use_audio_in_video=use_audio_in_video
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=False,
    )

    response = processor.batch_decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return response


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Qwen2.5-Omni inference over a folder of video/audio pairs."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "/scratch/zli33/models/Qwen2.5-Omni-7B"),
        help="Model id or local model path.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Parent folder containing subfolders of mp4/wav pairs.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to the output JSON file (default: results.json).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated thinker tokens.",
    )
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="Also use the audio track embedded in video input.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load prompt template ---
    prompt_path = Path("prompt.txt")
    if not prompt_path.is_file():
        raise FileNotFoundError("Prompt file not found: prompt.txt (expected in working directory)")
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    print(f"[INFO] Prompt template:\n{prompt_template}\n")

    # --- Collect all video/audio pairs ---
    av_pairs = collect_av_pairs(args.data_root)
    total_pairs = sum(len(v) for v in av_pairs.values())
    print(f"[INFO] Found {total_pairs} video/audio pair(s) across {len(av_pairs)} subfolder(s).")
    if total_pairs == 0:
        print("[WARN] Nothing to process. Exiting.")
        return

    # --- Load model ---
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[INFO] Loading model: {args.model}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)

    # --- Run inference ---
    results: Dict[str, list] = {}
    processed = 0

    for subfolder_name, pairs in av_pairs.items():
        subfolder_results: list = []
        for pair in pairs:
            processed += 1
            print(
                f"[{processed}/{total_pairs}] {subfolder_name}/{pair['stem']}  ...",
                flush=True,
            )
            try:
                response = infer_single(
                    model=model,
                    processor=processor,
                    video_path=pair["video"],
                    audio_path=pair["audio"],
                    prompt=prompt_template,
                    system_prompt=args.system_prompt,
                    use_audio_in_video=args.use_audio_in_video,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                response = f"[ERROR] {exc}"
                print(f"  ⚠ Error: {exc}")

            subfolder_results.append(
                {"file": pair["stem"], "response": response}
            )

        results[subfolder_name] = subfolder_results

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[INFO] Results saved to {output_path.resolve()}")

    pkl_path = output_path.with_suffix(".pkl")
    with pkl_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"[INFO] Pickle saved to {pkl_path.resolve()}")


if __name__ == "__main__":
    main()
