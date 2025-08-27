--

# 🤖 GPT-2 Text Generation – ShadowFox AIML Internship (Advanced Task)

## 📌 Project Overview

This project is part of the **Advanced Level Task** for the *ShadowFox AIML Internship*.
The goal is to explore **Language Models (LMs)** and demonstrate **GPT-2** for **prompt-based text generation** using Hugging Face Transformers.

---

## 🎯 Objectives

* Load a pre-trained **GPT-2** model and tokenizer.
* Provide custom **prompts** and generate continuation text.
* Experiment with **temperature**, **top-p**, **max\_length**, and **no\_repeat\_ngram\_size**.
* Evaluate outputs qualitatively and note strengths/limitations.

---

## ⚙️ Tech Stack

* **Language:** Python 3.8+
* **Libraries:** `transformers`, `torch`, `numpy` (optional), `matplotlib` (optional)

---

## 🗂️ Project Files

* `advanced_task_gpt2.ipynb` – Jupyter notebook to run the experiment
* `README.md` – This file
* `requirements.txt` – Dependencies

---

## 🔑 Steps Followed

1. Imported GPT-2 tokenizer and model (`GPT2LMHeadModel`, `GPT2Tokenizer`).
2. Chose a **prompt** to seed generation.
3. Generated text using sampling (temperature, nucleus top-p).
4. Tweaked parameters and compared results.
5. Summarized insights and limitations.

---

## ⌨️ Example Input & 🧾 Example Output

### Input (Prompt)

```
Artificial Intelligence is transforming the world of healthcare by
```

### Output (Sample)

```
Artificial Intelligence is transforming the world of healthcare by enabling faster diagnosis,
supporting clinicians with decision tools, and uncovering patterns in medical data that humans
often miss. From predicting patient risk to personalizing treatments, AI systems are helping
hospitals improve outcomes and reduce costs. However, responsible deployment—privacy, fairness,
and transparency—remains essential to building trust in these technologies.
```

*Note: Output varies each run due to sampling.*

---

## 🚀 How to Run

### 1) Clone the repo

```bash
git clone https://github.com/your-username/ShadowFox-AIML-Internship.git
cd ShadowFox-AIML-Internship/Advanced_GPT2_TextGeneration
```

### 2) Create & activate a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run (Notebook or Script)

**Option A — Jupyter Notebook**

```bash
jupyter notebook advanced_task_gpt2.ipynb
```

**Option B — Python Script (same logic)**
Create `gpt2_textgen.py` with this content:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Artificial Intelligence is transforming the world of healthcare by"
inputs = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(
    inputs,
    max_length=80,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    no_repeat_ngram_size=2,
    num_return_sequences=1
)

print("\nPrompt:", prompt)
print("\nGenerated Text:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Run it:

```bash
python gpt2_textgen.py
```


## 🔍 Strengths & Limitations

**Strengths:** Coherent short-form text, quick prototyping, controllable sampling.
**Limitations:** May produce generic or factually incorrect text; sensitive to prompts; not grounded in external knowledge.

---

## 📌 Future Improvements

* Fine-tune GPT-2 on a domain dataset (e.g., healthcare abstracts).
* Add a **Streamlit** UI for interactive prompting.
* Compare GPT-2 vs distilled/medium variants for speed/quality trade-offs.

---

## 🏆 Skills Gained

* Prompt engineering & text generation
* Transformer pipelines (Hugging Face)
* Practical hyperparameter tuning for sampling

---

## 👤 Author

**Parth Dasharathbhai Prajapati**

* GitHub: [https://github.com/Parth173600](https://github.com/Parth173600)
* LinkedIn: [https://www.linkedin.com/in/dte-gecbh-com-parth-prajapati-55980a282](https://www.linkedin.com/in/dte-gecbh-com-parth-prajapati-55980a282)

---
