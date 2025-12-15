import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =========================================================
# 1. Environment & Model Setup
# =========================================================
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_results_final_final_test_final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("🚀 Traffic Sign Quality Assessment (Final Portfolio Version)")

model_path = "best.pt"
yolo_model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")

# ---- LLaVA ----
model_id = "llava-hf/llava-1.5-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

print("✅ Models loaded successfully")

# =========================================================
# 2. Prompts
# =========================================================
PROMPTS = {
    0: "Is the number on this speed limit 30 sign clearly readable? If there is dirt, fading, or damage, mention it.",
    1: "Is the number on this speed limit 50 sign clearly readable? If there is dirt, fading, or damage, mention it.",
    2: "Is the pedestrian symbol on this crosswalk sign clearly readable? If there is dirt or damage, mention it.",
    3: "Are the figures on this children protection zone sign clearly readable? If there is dirt or damage, mention it.",
    4: "Is the symbol on this no parking sign clearly readable? If there is dirt or damage, mention it.",
    5: "Is the text on this STOP sign clearly readable? If there is dirt, graffiti, or damage, mention it.",
    6: "Is the arrow on this one-way sign clearly readable? If there is dirt or damage, mention it.",
    7: "Is the symbol on this slow sign clearly readable? If there is dirt or damage, mention it.",
}
DEFAULT_PROMPT = (
    "Is the text or symbol on this traffic sign clearly readable? "
    "If there is any damage, dirt, fading, or graffiti, mention it."
)

# =========================================================
# 3. Final Decision Logic (MAINTENANCE FIXED)
# =========================================================
def final_decision(answer: str):
    if not answer:
        return "UNKNOWN", "unknown", "unknown"

    t = answer.lower()

    # --- HARD FAIL (기능 상실) ---
    hard_fail_kw = [
        "cannot read",
        "not readable at all",
        "completely obscured",
        "severely damaged",
        "missing",
        "hole in the middle",
        "text is missing"
    ]
    if any(k in t for k in hard_fail_kw):
        return "FAIL", "illegible", "worn"

    # --- BORDERLINE → MAINTENANCE ---
    borderline_kw = [
        "not clearly readable",
        "partially readable",
        "partially obscured",
        "somewhat readable",
        "difficult to read",
        "hard to read",
        "faded",
        "rust",
        "graffiti",
        "dirty"
    ]
    if any(k in t for k in borderline_kw):
        return "MAINTENANCE", "borderline", "worn"

    # --- CLEAR PASS ---
    pass_kw = [
        "clearly readable",
        "clearly visible",
        "easy to read",
        "no dirt",
        "no damage",
        "no fading",
        "in good condition"
    ]
    if any(k in t for k in pass_kw):
        return "PASS", "legible", "clean"

    # --- FALLBACK ---
    return "UNKNOWN", "unknown", "unknown"


COLOR_MAP = {
    "PASS": "green",
    "MAINTENANCE": "orange",
    "FAIL": "red",
    "UNKNOWN": "gray",
}

# =========================================================
# 4. Main Pipeline
# =========================================================
def process_batch():
    image_files = [
        f for f in glob.glob(os.path.join(INPUT_FOLDER, "*"))
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"▶️ Processing {len(image_files)} images")
    records = []

    for img_path in tqdm(image_files, desc="Auditing"):
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")

        results = yolo_model(img, verbose=False)[0]

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        detected = False

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = yolo_model.names.get(cls_id, "unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < 0.5:
                records.append({
                    "Filename": filename,
                    "Class_Name": class_name,
                    "Confidence": round(conf, 4),
                    "FINAL_STATUS": "UNKNOWN",
                    "Legibility": "unknown",
                    "Condition": "unknown",
                    "VLM_Assessment": "Skipped (Low Confidence)",
                    "Class_ID": cls_id,
                })
                continue

            detected = True
            crop = img.crop((x1, y1, x2, y2))
            prompt = PROMPTS.get(cls_id, DEFAULT_PROMPT)

            inputs = processor(
                images=crop,
                text=f"USER: <image>\n{prompt}\nASSISTANT:",
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False
                )

            decoded = processor.decode(output[0], skip_special_tokens=True)
            answer = decoded.split("ASSISTANT:")[-1].strip()

            status, legibility, condition = final_decision(answer)

            records.append({
                "Filename": filename,
                "Class_Name": class_name,
                "Confidence": round(conf, 4),
                "FINAL_STATUS": status,
                "Legibility": legibility,
                "Condition": condition,
                "VLM_Assessment": answer,
                "Class_ID": cls_id,
            })

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor=COLOR_MAP[status],
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 8,
                f"{class_name} | {status}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor=COLOR_MAP[status], alpha=0.85),
            )

        if not detected:
            records.append({
                "Filename": filename,
                "Class_Name": "Not Detected",
                "Confidence": 0.0,
                "FINAL_STATUS": "UNKNOWN",
                "Legibility": "unknown",
                "Condition": "unknown",
                "VLM_Assessment": "N/A",
                "Class_ID": -1,
            })

        ax.axis("off")
        plt.savefig(
            os.path.join(OUTPUT_FOLDER, f"analyzed_{filename}"),
            bbox_inches="tight"
        )
        plt.close(fig)

    df = pd.DataFrame(records)
    df.to_csv(
        os.path.join(OUTPUT_FOLDER, "final_quality_report.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ Analysis complete. CSV + visualizations generated.")

# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    process_batch()
