# VLM-based Traffic Sign Visual Quality Assessment System

A hybrid AI pipeline combining Object Detection (YOLO) and Vision-Language Models (VLM) to assess traffic sign legibility.  
Designed to bridge the gap between **"machine detection confidence"** and **"human readability"** without additional training.

---

## Features

-   **Hybrid Pipeline**: Combines **YOLOv8** (Localization) + **LLaVA-1.5** (Visual Reasoning).
-   **No Additional Training Required**: The system intentionally avoids retraining YOLO on damaged or worn sign classes, focusing instead on reasoning-based evaluation to ensure generalization.
-   **Zero-shot Quality Assessment**: Evaluates physical conditions (rust, dirt, fading) using natural language prompts without relying on specific damaged-data training.
-   **3-Tier Safety Logic**: Classifies signs into **PASS**, **MAINTENANCE**, or **FAIL** based on legibility and physical condition.
-   **Automated Reporting**: Generates visual results (color-coded boxes) and a structured CSV report.
<img width="1204" height="827" alt="스크린샷 2025-12-15 19 35 26" src="https://github.com/user-attachments/assets/50f4ffe5-4064-4f2d-9f83-7b9655c84c43" />
---

## HOW TO USE 


1.  **Environment Setup**
    Ensure you have Python 3.8+ and a CUDA-capable GPU. Install the required libraries:
    ```bash
    pip install torch ultralytics transformers accelerate bitsandbytes pandas matplotlib tqdm
    ```

2.  **Prepare Input Images**
    Place your road/traffic sign images into the `input_images` folder.

3.  **Run the Analysis**
    Execute the main batch processing script:
    ```bash
    python run_batch_vlm.py
    ```
    *Note: The script automatically loads `best.pt` if available, or falls back to `yolov8n.pt`.*

4.  **Check Results**
    Go to the `output_results` folder to find:
    -   **`analyzed_*.jpg`**: Annotated images with status-coded bounding boxes.
    -   **`final_quality_report.csv`**: Detailed logs including detection confidence and VLM reasoning text.
    -   **`class_statistics.csv`**: Statistical summary of pass/fail rates.

---

## System Logic & Decision Policy

This project separates the role of **Detection** and **Reasoning** to maximize reliability.

1.  **Detection Stage (YOLOv8)**
    -   Locates the sign and filters out low-confidence detections (< 0.5).

2.  **Reasoning Stage (LLaVA-1.5)**
    -   Evaluates the cropped sign image using class-specific prompts (e.g., *"Is the arrow clearly readable? If there is rust, mention it."*).

3.  **Decision Stage (Conservative Rule-based Normalization)**
    -   The decision logic is **intentionally conservative** due to the safety-critical nature of traffic infrastructure.

### Decision Logic Summary

| Condition observed in VLM output | Final Status | Action Required |
| :--- | :--- | :--- |
| **Cannot read / missing symbol** | **🔴 FAIL** | **Immediate Replacement** (Safety Hazard) |
| **Readable but rusted / dirty / faded** | **🟠 MAINTENANCE** | **Preventive Action** (Monitor/Clean) |
| **Clearly readable, no damage** | **🟢 PASS** | **None** (Good Condition) |
| **Low confidence / ambiguous** | **⚪ UNKNOWN** | **Manual Review** |

* **MAINTENANCE Definition**: The sign is legible but shows physical degradation that may compromise future visibility or safety.

---

## 🛠️ Tech Stack

* **Language**: Python 3.8+
* **Object Detection**: YOLOv8 (Ultralytics)
* **Vision-Language Model**: LLaVA-1.5-7B (Hugging Face Transformers)
* **Optimization**: BitsAndBytes (4-bit Quantization), Accelerate
* **Data Analysis**: Pandas, Matplotlib

---

## Contribution Summary

This work demonstrates how Vision-Language Models can **augment conventional object detection systems into maintenance-aware infrastructure monitoring pipelines**, without the need for additional data collection or retraining.

<br>

---
---

<br>

# VLM 기반 교통 표지판 시각적 품질 자동 평가 시스템

객체 탐지(YOLO)와 시각-언어 모델(VLM)을 결합하여 교통 표지판의 '실질적 가독성'을 평가하는 하이브리드 AI 파이프라인입니다.  
**탐지 신뢰도(Confidence)를 넘어, 사람이 실제로 읽을 수 있는가**를 판단하는 데 초점을 맞추었습니다.

---

## 특징

-   **하이브리드 파이프라인**: **YOLOv8** (위치 탐지) + **LLaVA-1.5** (상태 진단) 결합
-   **추가 학습 배제 설계**: 본 시스템은 표지판의 훼손 상태를 객체 탐지 모델에 별도로 학습시키지 않으며, 추론 기반 판단(VLM)을 통해 일반화 성능을 확보합니다.
-   **Zero-shot 품질 평가**: 별도의 손상 데이터셋 학습 없이, VLM의 추론 능력만으로 녹/오염/변색 등을 진단합니다.
-   **3단계 안전 등급**: **PASS** (정상), **MAINTENANCE** (예방 정비), **FAIL** (식별 불가)로 자동 분류합니다.
-   **자동화된 리포팅**: 시각화된 결과 이미지(색상 박스)와 정량적 CSV 데이터를 생성합니다.

---

## 사용 방법

1.  **환경 설정**
    Python 3.8 이상 및 CUDA 지원 GPU가 필요합니다. 필수 라이브러리를 설치하세요:
    ```bash
    pip install torch ultralytics transformers accelerate bitsandbytes pandas matplotlib tqdm
    ```

2.  **이미지 준비**
    분석할 도로 주행 이미지나 표지판 이미지를 `input_images` 폴더에 넣습니다.

3.  **분석 실행**
    메인 스크립트를 실행합니다:
    ```bash
    python run_batch_vlm.py
    ```
    *참고: 폴더 내에 학습된 `best.pt`가 없으면 자동으로 `yolov8n.pt`를 다운로드하여 사용합니다.*

4.  **결과 확인**
    `output_results` 폴더에서 다음을 확인하세요:
    -   **`analyzed_*.jpg`**: 상태별 색상(초록/주황/빨강) 박스가 쳐진 이미지
    -   **`final_quality_report.csv`**: 탐지 신뢰도, VLM 원문 답변, 최종 판정이 포함된 로그
    -   **`class_statistics.csv`**: 표지판 종류별 불량률 통계

---

## 시스템 로직 및 판정 기준

본 프로젝트는 **탐지(Detection)** 와 **판단(Reasoning)** 의 역할을 분리하여 정확도를 높였습니다.

1.  **탐지 단계 (Detection)**: YOLOv8을 통해 표지판의 위치를 찾고 이미지를 추출합니다.
2.  **판단 단계 (Reasoning)**: LLaVA-1.5 VLM이 추출된 이미지를 보고 시인성과 물리적 상태를 진단합니다.
3.  **결정 단계 (Decision)**: 안전 최우선 원칙에 따라 보수적인 룰 기반 정규화를 수행합니다.

### 판정 로직 요약 (Decision Logic Summary)

| VLM 관측 결과 | 최종 판정 | 조치 사항 |
| :--- | :--- | :--- |
| **식별 불가 / 내용 가려짐** | **🔴 FAIL** | **즉시 교체** (안전 위협) |
| **식별 가능하나 녹/오염/변색** | **🟠 MAINTENANCE** | **예방 정비** (세척 및 모니터링) |
| **선명하고 깨끗함** | **🟢 PASS** | **조치 없음** (정상) |
| **탐지 신뢰도 부족 / 판단 불가** | **⚪ UNKNOWN** | **수동 검수** |

* **MAINTENANCE 정의**: 현재는 식별 가능하나, 물리적 열화가 진행되어 향후 시인성이나 안전을 저해할 수 있는 상태.

---

## 🛠️ 사용 기술 (Tech Stack)

* **Language**: Python 3.8+
* **Object Detection**: YOLOv8 (Ultralytics)
* **Vision-Language Model**: LLaVA-1.5-7B (Hugging Face Transformers)
* **Optimization**: BitsAndBytes (4-bit Quantization), Accelerate
* **Data Analysis**: Pandas, Matplotlib

---

## 프로젝트 기여 요약 (Contribution Summary)


본 프로젝트는 기존 객체 탐지 시스템을 수정하거나 재학습하지 않고, **Vision-Language Model을 결합하여 유지보수 판단이 가능한 인프라 관리 시스템으로 확장한 실증 사례**입니다.

<br>

---
---

<br>

## Image Sources & Data Attribution
**Traffic Sign Images**
This project uses a combination of publicly available real-world traffic sign images and synthetically generated images for system demonstration and evaluation purposes.
1. Real-world Images
* Collected from public road environments and open-access datasets
* Used strictly for research and non-commercial purposes
* No personally identifiable information (PII) is included
* Images are used only to evaluate visual legibility and physical condition of traffic signs
Example sources include:
* Public-domain or open-license road scene images
* Self-collected photographs taken in public spaces
2. Synthetic Images
* Artificially generated traffic sign images created for controlled experiments
* Used to simulate various conditions such as:
    * Dirt, rust, fading
    * Partial occlusion
    * Physical degradation
* Synthetic samples are clearly labeled (e.g., *_synth.png) and are not intended to represent real traffic environments

Dataset Usage Notice
* This repository does not redistribute proprietary or restricted datasets
* Images are used solely to demonstrate the proposed VLM-based quality assessment pipeline
* The project focuses on system design and reasoning logic, not dataset construction
If you are the owner of any image used in this project and believe attribution or removal is required, please open an issue or contact the author.

Ethical Considerations
* The system is designed for research and infrastructure maintenance support
* It is not intended for real-time autonomous driving decisions
* All evaluations are performed offline with human-in-the-loop review recommended for deployment

## 이미지 출처 및 데이터 사용 안내
본 프로젝트는 공개 접근이 가능한 실제 교통 표지판 이미지와 실험 목적의 합성(Synthetic) 이미지를 함께 사용합니다.
1. 실제 이미지
* 공개된 도로 환경에서 촬영되었거나 공개 라이선스 기반 데이터셋에서 수집
* 연구 및 비상업적 목적에 한하여 사용
* 개인 식별 정보(PII)는 포함하지 않음
* 교통 표지판의 시인성 및 물리적 상태 평가 목적으로만 활용
2. 합성 이미지
* 실험 통제를 위해 인위적으로 생성된 이미지
* 녹, 오염, 변색, 가림 현상 등을 시뮬레이션하기 위해 사용
* 파일명에 _synth로 명확히 구분됨
* 실제 도로 환경을 그대로 재현하는 목적은 아님

데이터 사용 고지
* 본 저장소는 저작권이 제한된 데이터셋을 재배포하지 않습니다
* 본 프로젝트의 목적은 데이터 공개가 아닌 시스템 설계 및 추론 로직 검증입니다
* 이미지 소유자가 출처 표기 또는 삭제를 요청할 경우 즉시 조치합니다




