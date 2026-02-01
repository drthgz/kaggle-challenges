# Kaggle Challenges Repository

A structured workspace for participating in Kaggle monthly competitions with organized learning and reusable techniques.

## 📁 Repository Structure

```
kaggle-challenges/
├── README.md                    # This file - overview and navigation
├── LEARNINGS.md                 # Master document of cumulative ML insights
├── WORKFLOW.md                  # Step-by-step guide for running challenges
├── CHALLENGE_TEMPLATE.md        # Template for starting new challenges
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── 2026-01/                     # January 2026 Challenge (Regression)
│   ├── notebook.ipynb          # Main ML pipeline
│   ├── notes.md                # Project-specific insights
│   ├── data/                   # Datasets (in .gitignore)
│   └── submissions/            # Final CSV submissions
│
├── 2026-02/                     # February 2026 Challenge (Classification)
│   ├── notebook.ipynb
│   ├── notes.md
│   ├── data/
│   └── submissions/
│
├── shared/                      # Reusable across all challenges
│   ├── utils.py                # Common utility functions
│   ├── techniques.md           # ML techniques reference library
│   └── templates/              # Starter templates
│       └── preprocessing_template.ipynb
│
└── archive/                     # Completed/historical challenges
```

## 🏃 Challenge Workflow

See [WORKFLOW.md](WORKFLOW.md) for detailed execution steps. Quick overview:

1. Create challenge folder: `2026-XX/` with standard structure
2. Download data into `data/` subfolder
3. Work in `notebook.ipynb` - develop ML pipeline
4. Record learnings in `notes.md` (project-specific only)
5. Generate submission CSV to `submissions/`
6. Update root [LEARNINGS.md](LEARNINGS.md) with generalizable insights

## 📊 Active Challenges

| Challenge | Type | Metric | Status | Result |
|-----------|------|--------|--------|--------|
| 2026-01 | Regression | RMSE | ✅ Complete | 2243 private |
| 2026-02 | Classification | AUC-ROC | 🟢 In Progress | — |

## 🧠 Learning Structure

**Challenge-Specific** (`notes.md`): 
- What worked/didn't work for THIS problem
- Hyperparameters tuned
- Feature engineering ideas tested
- Leaderboard position and score

**General Learnings** (`LEARNINGS.md`):
- Patterns across multiple challenges
- Model comparison results
- Preprocessing best practices
- Feature engineering ROI analysis
- When to use each technique

**Reusable Code** (`shared/`):
- `utils.py` - Functions used across challenges
- `techniques.md` - Reference for techniques
- `templates/` - Starter notebooks

## 🛠 Setup

```bash
cd kaggle-challenges/
pip install -r requirements.txt
cd 2026-XX/  # Replace with challenge number
jupyter notebook notebook.ipynb
```

## ✨ Key Principles

- **Project files in challenge folders**: Keep clutter minimal (notebook, data, submissions, notes)
- **Learning at root level**: LEARNINGS.md and WORKFLOW.md for everyone to reference
- **Reusable code in shared**: Avoid duplication across challenges
- **notes.md is brief**: Focus on THIS challenge's learnings, not general theory

---

Last updated: January 31, 2026
