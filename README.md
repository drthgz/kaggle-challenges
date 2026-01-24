# Kaggle Challenges Repository

A structured workspace for participating in Kaggle monthly competitions, completing archived challenges, and building a personal ML techniques library.

## 📁 Repository Structure

```
kaggle-challenges/
├── README.md                    # This file - overview and navigation
├── requirements.txt             # Python dependencies for all challenges
├── LEARNINGS.md                 # Master document of insights and patterns
├── .gitignore                   # Git ignore rules for data and submissions
│
├── 2026-01/                     # January 2026 Challenge
│   ├── README.md               # Challenge summary, approach, results
│   ├── notebook.ipynb          # Working notebook (local iterations)
│   ├── submissions/            # Final submission versions
│   │   └── notebook_v1.ipynb
│   ├── data/                   # Dataset files (in .gitignore)
│   └── notes.md                # Key learnings and insights
│
├── 2026-02/                     # Future challenges follow same pattern
│   ├── README.md
│   ├── notebook.ipynb
│   ├── submissions/
│   ├── data/
│   └── notes.md
│
├── shared/                      # Reusable code and resources
│   ├── utils.py                # Common utility functions
│   ├── techniques.md           # ML techniques reference
│   └── templates/              # Starter code templates
│       └── preprocessing_template.ipynb
│
└── archive/                     # Completed/archived challenges
```

## 🎯 Workflow

1. **Create Challenge Folder**: New folder for each competition with consistent structure
2. **Develop**: Work in `notebook.ipynb` locally
3. **Document**: Record learnings in `notes.md` and update challenge README
4. **Submit**: Create final version in `submissions/` folder
5. **Learn**: Add patterns to `shared/` and `LEARNINGS.md`

## 📊 Current Challenges

- **2026-01**: Kaggle Playground Series - Predicting Student Test Scores ([Kaggle Link](https://www.kaggle.com/competitions/playground-series-s6e1))

## 🛠 Getting Started

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Learning Resources

- See [LEARNINGS.md](LEARNINGS.md) for cumulative insights
- Check [shared/techniques.md](shared/techniques.md) for technique reference
- Review individual challenge README.md files for approach documentation

## ✨ Key Practices

- **Document as you learn**: Write in `notes.md` while working
- **Build reusable code**: Extract functions to `shared/utils.py`
- **Add context**: Explain your approach in README files
- **Comment code**: Explain *why* you chose specific techniques

---

Last updated: January 24, 2026
