# Immune Cell AI-Assisted Classifier 免疫细胞 AI 辅助判读工具

An end-to-end H&E-stained immune-cell classification tool with interpretability and a human-in-the-loop confidence guardrail.
端到端 H&E 染色免疫细胞辅助判读工具:图像上传 → 细胞识别 → Grad-CAM 可解释归因 → 低置信预警转人工。

> **Product positioning / 产品定位**: assist — never replace — the pathologist. Predictions below 50% confidence trigger an explicit "recommend manual review" warning, keeping the final decision with the human expert.
> 定位为"辅助而非替代":低置信(<50%)结果强制提示人工复核,最终决定权留给使用者。

**Project**: In collaboration with the University of Sydney medical faculty · core lead for evaluation-system design and product delivery
**团队**:6 人课程项目,本人主导评测体系设计与结果产品化

---

## Results 模型评测结果

5 architectures benchmarked under a unified protocol (per-class recall & macro-F1 as primary metrics under 5:1 class imbalance; ROC-AUC / Log Loss as secondary; confusion-matrix error attribution). All 10 pairwise ensembles evaluated.

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| ResNet50 | 0.604 | 0.451 | 0.657 |
| DenseNet121 | 0.545 | 0.448 | 0.689 |
| EfficientNet-B0 | 0.606 | 0.453 | 0.707 |
| ViT-B/16 | 0.574 | 0.449 | 0.690 |
| MobileNetV3 | 0.621 | 0.455 | 0.644 |
| **MobileNetV3 + EfficientNet-B0 (ensemble, final)** | **0.612** | **0.488** | — |

Final ensemble choice trades ~0.9pp accuracy for a +7.3% relative macro-F1 gain and complementary errors (error correlation 0.388) — the balanced metric matters more in an assistive setting.

## Dataset 数据

22,005 H&E-stained immune-cell images in 3 classes: 4,987 B cells, 15,394 T cells (CD4+ and CD8+ merged), 1,624 Macrophages (~5:1 imbalance). Split 60/20/20 with a fixed seed; class imbalance handled via WeightedRandomSampler and augmentation.

原始数据集(作者提供):[Dropbox 链接](https://www.dropbox.com/scl/fo/avptnrxjluth1w414ufla/AEt8ARm3mICcu8gFuj_QIK8?rlkey=wuw9isad1dw4b90o1aqdvfee6&e=1&dl=0)

## App features 应用功能

- Upload an H&E cell image (PNG/JPG) and classify: B Cell / Macrophage / T Cell
- Choose any of the 5 individual models or all 10 pairwise ensemble combinations
- Confidence score and per-class probability breakdown
- Low-confidence (<50%) warning recommending manual pathologist review
- Grad-CAM heatmaps per ensemble component, with color scale and reading guide
- Sidebar model-performance comparison table

## Quickstart 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py        # then open http://localhost:8501
```

Runs on CPU; CUDA is used automatically if available.

## Project documents 产品文档(docs/)

| Document | Content |
|---|---|
| `docs/免疫细胞AI辅助判读工具_一页纸PRD.md` | One-page PRD: scene research (cited), user stories, goals/non-goals, P0–P2 requirements, acceptance criteria, ADRs, review & delivery log, quality-ops & A/B plan, limitations & path to production |
| `docs/系统架构图.png` (+ `.drawio` source) | 5-layer system architecture: UI → inference orchestration → model zoo → Grad-CAM layer → assets |
| `docs/判读主链路泳道图.png` (+ `.drawio` source) | Main pipeline swimlane with the low-confidence → manual-review boundary |
| `docs/界面原型线框图.png` (+ `.drawio` source) | Low-fidelity UI wireframe: upload state & low-confidence result state, with interaction-design annotations |
| `docs/app_ui_home.png` | Real delivery screenshot of the shipped app |
| `workflow_diagram.png` | Original project workflow diagram |

## Repository notes 仓库说明

- **Model checkpoints (~480 MB)** are not included — individual files exceed GitHub's 100 MB limit. Retrain via `Model Building Code.ipynb`, or obtain weights from the author.
- **Dataset (1.8 GB, 22,005 images)** is not included for size reasons; class counts and split protocol are documented above.
- `Project_Report.html` reproduces the full evaluation from checkpoints (see `Project_Report.ipynb` / Quarto).

## Run in Google Colab (from the original submission)

```python
# Step 1 — Install dependencies and mount Drive
!pip install streamlit grad-cam pyngrok -q
from google.colab import drive
drive.mount('/content/drive')

# Step 2 — Copy app to local disk
!cp "/content/drive/My Drive/ImmuneCellClassifier/app.py" /content/app.py

# Step 3 — Launch with ngrok tunnel
from pyngrok import ngrok
import threading, subprocess, time

ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # get free token at dashboard.ngrok.com

def run():
    subprocess.run(['streamlit', 'run', '/content/app.py',
                    '--server.port', '8501', '--server.headless', 'true'])

threading.Thread(target=run, daemon=True).start()
time.sleep(4)
print("App running at:", ngrok.connect(8501))
```
