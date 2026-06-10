# 🗂️ Git Setup Explainer — How Your Two Repos Work

## The Big Picture

You have **one folder on your Mac** but it is connected to **two separate GitHub repos**:

```
Your Mac: /Users/jinishrajan/ucd/Medimaging/
│
├── remote: origin  ──────► github.com/Ant-Smalls/MRNet_Assignment   (TEAM repo)
│                            Shared with: Antony, the whole group
│                            They can SEE everything pushed here
│
└── remote: mine   ──────► github.com/Dublindeveloper/mrnet-learning  (YOUR repo)
                             Only YOU can see this (private)
                             This is your personal backup + Colab source
```

---

## The Two Branches

```
main branch                 ← Team's original code (Antony set this up)
local-learning-end-to-end   ← YOUR branch with all the ML code we built
```

**Rule:** You are always working on `local-learning-end-to-end`.

---

## Cheat Sheet — Which Command Does What

### Check where you are right now
```bash
cd /Users/jinishrajan/ucd/Medimaging

# See which branch you are on
git branch

# See what files have changed since last commit
git status

# See both remotes
git remote -v
```

---

### Saving your work locally (commit)
```bash
# Stage all changed files
git add code/src/

# Commit with a message describing what you did
git commit -m "your message here"

# Example:
git commit -m "feat: improve Grad-CAM overlay colours"
```
> ✅ This saves to your Mac. Nothing goes to the internet yet.

---

### Push to YOUR private repo (safe — only you see this)
```bash
git push mine local-learning-end-to-end
```
> ✅ Backs up your code to github.com/Dublindeveloper/mrnet-learning  
> ✅ This is what Google Colab clones from  
> ❌ Team members cannot see this

---

### Push to the TEAM repo (careful — everyone sees this)
```bash
git push origin local-learning-end-to-end
```
> ⚠️  Only do this when the team agrees to merge your branch  
> ⚠️  Antony or the team lead will likely do a "Pull Request" for this  
> ❌ Never push `local_learning_hub/` contents — they are gitignored anyway

---

### Pull latest team changes (keep up with the group)
```bash
git fetch origin
git merge origin/main
```
> ✅ Gets any updates Antony or teammates pushed to the team repo  
> ✅ Merges them into your working branch

---

### Typical daily workflow
```bash
# 1. Start work — check you are on your branch
git branch
# Should show: * local-learning-end-to-end

# 2. Write code / make changes

# 3. Save your progress locally
git add code/src/
git commit -m "describe what you changed"

# 4. Back up to your private repo
git push mine local-learning-end-to-end

# 5. That's it! Team never sees anything.
```

---

## What is gitignored (never goes anywhere)

The `.gitignore` file tells Git to silently ignore these:

| Folder / File | Why ignored |
|---|---|
| `local_learning_hub/` | Your entire private learning space |
| `code/src/data/` | MRNet dataset — 7.2 GB, too large for git |
| `.venv/` | Python virtual environment |
| `__pycache__/` | Python auto-generated files |
| `.DS_Store` | Mac system files |

> Even if you do `git add .` by accident, none of the above will be included.

---

## What IS tracked by git (goes to GitHub when you push)

```
code/
├── src/
│   ├── modules/
│   │   ├── data_preprocessing_transformation.py  ✅
│   │   ├── baseline_models.py                    ✅
│   │   └── comparative_models.py                 ✅
│   ├── train.py                                  ✅
│   ├── evaluation.py                             ✅
│   ├── explainability.py                         ✅
│   ├── data_sanity_check.py                      ✅
│   └── visualise_sample.py                       ✅
requirements.txt                                  ✅
.gitignore                                        ✅
```

---

## If something goes wrong

### Accidentally staged files you did not want to add
```bash
git reset HEAD code/src/some_file.py
```

### See the full commit history
```bash
git log --oneline -10
```

### Undo the last commit (but keep the file changes)
```bash
git reset --soft HEAD~1
```

### Check the difference between what you changed and last commit
```bash
git diff code/src/train.py
```

---

## Summary in one sentence

> Commit often → push to `mine` to back up privately → only push to `origin` when the team agrees to merge.
