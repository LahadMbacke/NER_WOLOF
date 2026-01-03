# NER Wolof - Named Entity Recognition for Wolof Language

A fine-tuned **GLiNER** model for Named Entity Recognition (NER) in Wolof, a language spoken primarily in Senegal, Gambia, and Mauritania. Fine-tuned on the **MasakhaNER** dataset from Hugging Face.

## 🎯 Project Overview

This project provides:
- A fine-tuned GLiNER model for Wolof NER
- Training scripts to reproduce or improve the model


## 🤗 Fine-tuned Model

The fine-tuned model is available on Hugging Face Hub:

👉 **[Lahad/gliner_wolof_NER](https://huggingface.co/Lahad/gliner_wolof_NER)**

### Quick Usage

```python
from gliner import GLiNER

# Load the fine-tuned model
model = GLiNER.from_pretrained("Lahad/gliner_wolof_NER")

# Predict entities
text = "Ousmane Sonko jàngae na ci Daaray Cheikh Anta Diop ci Dakar."
labels = ["PER", "ORG", "LOC", "DATE"]

entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(f"{entity['text']} => {entity['label']} (score: {entity['score']:.2f})")
```

**Output:**
```
Ousmane Sonko => PER (score: 0.95)
Daaray Cheikh Anta Diop => ORG (score: 0.89)
Dakar => LOC (score: 0.97)
```

## 📊 Dataset

This project uses the [MasakhaNER](https://huggingface.co/datasets/masakhaner) dataset, which provides high-quality NER annotations for 10 African languages including Wolof (`wol`).

**Entity Types:**
- **PER** - Person names
- **ORG** - Organizations
- **LOC** - Locations
- **DATE** - Dates

## 📈 Evaluation Results

Evaluation on the test set (539 samples):

| Entity Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **DATE**    | 30.77%    | 22.86% | 26.23%   | 70      |
| **LOC**     | 76.75%    | 84.95% | 80.65%   | 206     |
| **ORG**     | 41.89%    | 56.36% | 48.06%   | 55      |
| **PER**     | 53.02%    | 70.69% | 60.59%   | 174     |
| **GLOBAL**  | **58.87%**| **68.32%** | **63.24%** | 505 |

### ⚠️ Performance Note

The model was fine-tuned on a relatively limited dataset (MasakhaNER Wolof). Current performance reflects this constraint, particularly for **DATE** and **ORG** entity types which have fewer training examples.

**Future Improvements:**
- Collect and annotate more data in Wolof
- Increase source diversity (newspapers, social media, literature)
- Experiment with data augmentation techniques

With more annotated data, we expect to significantly improve the model's performance.


## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train_gliner.py --config config/config.yaml
```



## 🏷️ Entity Types

| Label | Description | Example |
|-------|-------------|---------|
| `PER` | Person names | Ousmane Sonko, Lahad Mbacke |
| `LOC` | Locations | Dakar, Sénégal, Tuuba |
| `B-ORG` / `I-ORG` | Organizations | CEDEAO, ONU |
| `B-DATE` / `I-DATE` | Dates | 01 Oktoobar 2025 |
| `B-MISC` / `I-MISC` | Miscellaneous entities | Korité, Tabaski |
| `O` | Outside any entity | - |

ve

## � Training Tips

1. **Start with a good base model**: Use `Gliner` for best results on African languages.

2. **Data quality matters**: MasakhaNER provides high-quality annotations.

3. **Hyperparameter tuning**:
   - Learning rate: `1e-5` to `5e-5`
   - Batch size: `8` to `32` (depending on GPU memory)
   - Epochs: `5` to `20`

4. **Use early stopping**: Prevents overfitting on small datasets.

5. **Monitor validation metrics**: Focus on F1 score for imbalanced NER data.

## 🔍 Wolof Language Resources

- [Wolof Wikipedia](https://wo.wikipedia.org/)
- [Masakhane NER](https://github.com/masakhane-io/masakhane-ner) - African NER datasets


## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Masakhane](https://www.masakhane.io/) - NLP for African languages
- The Wolof language community