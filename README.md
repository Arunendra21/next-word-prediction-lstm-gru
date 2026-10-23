# Next Word Prediction Using LSTM and GRU

This project builds a **next word prediction** system: given a sequence of words, the model predicts the single most likely next word. Two recurrent neural network models are trained and compared on exactly the same data — an **LSTM** model and a **GRU** model — and both are used for prediction at the end.

**Run it in Google Colab (no setup needed):**
[Open in Google Colab](https://colab.research.google.com/drive/1cg0Rs1vcje7qMKQfhyAW7jn8cPVKLbq3)

The notebook linked above is the full project, executed end to end: every cell was run in order, both models were trained, and all outputs are saved inside the notebook.

---

## Where the Data Comes From

The dataset is the text of **William Shakespeare's play *Hamlet***.

- **Source:** the [NLTK](https://www.nltk.org/) `gutenberg` corpus, which packages texts from [Project Gutenberg](https://www.gutenberg.org/).
- **Exact file used:** `shakespeare-hamlet.txt` from the NLTK Gutenberg corpus.
- **How the code gets it (Step 1 of the notebook):**

```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-hamlet.txt')

with open('hamlet.txt', 'w') as file:
    file.write(data)
```

The raw text downloaded from the corpus is immediately saved as **`hamlet.txt`**, and that saved file is what every later step reads. A copy of `hamlet.txt` is included in this repository, so the project can also be run offline without downloading anything.

**Dataset size (from the actual run):**

| Property | Value |
|---|---|
| Characters in `hamlet.txt` | ~162,900 |
| Lines of text | ~4,900 |
| Unique words in the vocabulary | 4,817 |
| `total_words` used by the models (vocabulary + 1) | 4,818 |
| Longest input sequence (`max_sequence_len`) | 14 words |

---

## Step-by-Step: What the Code Does

The notebook (`experiments.ipynb`) is organized in the order below. Each numbered step is a heading in the notebook, followed by the code for that step.

### Step 1 — Data Collection
Downloads the Gutenberg corpus through NLTK, loads `shakespeare-hamlet.txt`, and saves it locally as `hamlet.txt` (explained in the data section above).

### Step 2 — Data Preprocessing
- Reads `hamlet.txt` back and converts the whole text to **lowercase**, so "Hamlet" and "hamlet" are treated as the same word.
- Creates a Keras `Tokenizer` and fits it on the text. The tokenizer assigns every unique word an integer index, stored in `tokenizer.word_index` (for example `'the': 1, 'and': 2, 'to': 3 ...`).
- `total_words` is set to the vocabulary size + 1 = **4,818**. The extra slot is index 0, which is reserved for padding.

### Step 3 — Create Input Sequences
The model learns from examples of "some words → next word", so the text is converted into n-gram sequences:
- The text is processed **line by line**.
- Each line is converted to a list of word indexes using the tokenizer.
- For a line with indexes `[w1, w2, w3, w4]`, every prefix becomes a training example: `[w1, w2]`, `[w1, w2, w3]`, `[w1, w2, w3, w4]`.
- All of these prefixes together form `input_sequences` (roughly 25,700 sequences).

### Step 4 — Pad Sequences
Sequences have different lengths, so they are made uniform:
- `max_sequence_len` is the length of the longest sequence = **14**.
- Every sequence is **pre-padded** with zeros at the beginning (`padding='pre'`) so all of them are exactly 14 indexes long.

### Step 5 — Create Predictors and Label
Each padded sequence is split into an input and an answer:
- **Predictors `x`** = all indexes except the last one (13 words of context).
- **Label `y`** = the last index (the word the model must predict).

### Step 6 — One-Hot Encode the Labels
The label is converted with `to_categorical` into a one-hot vector of length 4,818 — a 1 at the position of the correct next word and 0 everywhere else. This matches the softmax output of the models, which produce a probability for every word in the vocabulary.

### Step 7 — Split into Training and Testing Sets
`train_test_split` divides the data into **80% training** and **20% testing/validation**. Training runs at 644 steps per epoch.

### Step 8 — Define Early Stopping
Early stopping watches the validation loss while training:
- `monitor='val_loss'`, `patience=3` — training stops if the validation loss does not improve for 3 epochs in a row.
- `restore_best_weights=True` — the weights from the best epoch are kept, not the last one.
- A separate early stopping instance is created for each model (`early_stopping_lstm` and `early_stopping_gru`).
- Both models are configured to train for up to **50 epochs**, but in practice early stopping ends training much sooner (see the results below).

### Step 9 — Build the LSTM Model
The first model, `lstm_model`, is a Keras `Sequential` model:

| Layer | Configuration | Output |
|---|---|---|
| Embedding | 4,818 words → 100-dimensional vectors, input length 13 | (13, 100) |
| LSTM | 150 units, `return_sequences=True` | (13, 150) |
| Dropout | rate 0.2 (helps prevent overfitting) | (13, 150) |
| LSTM | 100 units | (100) |
| Dense | 4,818 units, `softmax` activation | probability per word |

Total parameters: **1,219,418**. It is compiled with the `adam` optimizer, `categorical_crossentropy` loss, and `accuracy` as the metric.

### Step 10 — Train the LSTM Model
`lstm_model.fit(...)` trains on the training set, validating on the test set after every epoch, with the LSTM early stopping callback attached.

### Step 11 — Build the GRU Model
The second model, `gru_model`, is identical to the LSTM model except the two LSTM layers are replaced by **GRU** layers of the same sizes (150 units with `return_sequences=True`, then 100 units), with the same embedding layer, dropout rate, dense softmax output layer, optimizer, loss, and metric. Keeping everything else the same makes the comparison between the two architectures fair.

### Step 12 — Train the GRU Model
`gru_model.fit(...)` trains the GRU model on the same training data, with the same validation data and its own early stopping callback.

### Step 13 — Predict the Next Word
The `predict_next_word(model, tokenizer, text, max_sequence_len)` function works with either model:
1. Converts the input text into word indexes with the same tokenizer.
2. If the input is longer than the context window, keeps only the last `max_sequence_len - 1` indexes.
3. Pre-pads the indexes to length 13.
4. Runs `model.predict(...)` to get a probability for all 4,818 words.
5. Picks the word with the highest probability (`argmax`) and returns it.

### Step 14 — Test Both Models
Both trained models are tested on the same input sentences, so their predictions can be compared side by side (results below).

### Step 15 — Save the Models and the Tokenizer
- `lstm_model` is saved as `next_word_lstm.h5`.
- `gru_model` is saved as `next_word_gru.h5`.
- The fitted `tokenizer` is saved as `tokenizer.pickle` (using `pickle`), because predictions are impossible without the exact same word-to-index mapping used in training.

---

## Results from the Latest Colab Run

Both models were trained in the linked Colab notebook on 23 October 2026:

| | LSTM model | GRU model |
|---|---|---|
| Epochs run (of 50 max) | 4 (early stopping) | 4 (early stopping) |
| Best validation loss | 6.7649 (epoch 1) | 6.8113 (epoch 1) |
| Training time in Colab | ~3.5 minutes | ~2.75 minutes |

Test predictions from that run:

| Input text | LSTM prediction | GRU prediction |
|---|---|---|
| "To be or not to be" | the | the |
| "Barn. Last night of all, When yond same" | the | the |

A note on reading these results: in this run the validation loss was lowest at epoch 1 for both models and never improved afterwards, so early stopping restored the epoch-1 weights. The predictions reflect that — the model falls back to very common words. This is an educational project for learning how sequence models, tokenization, and early stopping work, not a production-quality language model.

> The `.h5` model files committed in this repository are the original trained LSTM models that came with the project. Running the notebook yourself re-trains both models and produces fresh `next_word_lstm.h5` and `next_word_gru.h5` files.

---

## The Streamlit App

`app.py` turns the trained LSTM model into a small web app:

1. Loads `next_word_lstm.h5` and `tokenizer.pickle`.
2. Shows a text box (default input: "To be or not to") and a **Predict Next Word** button.
3. Reads the context length directly from the model (`model.input_shape[1] + 1`), runs the same `predict_next_word` logic as the notebook, and displays the predicted next word.

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Files in This Repository

| File | What it is |
|---|---|
| `experiments.ipynb` | The full notebook: data collection, preprocessing, and training/testing of both the LSTM and GRU models |
| `app.py` | Streamlit web app for next word prediction using the LSTM model |
| `hamlet.txt` | The dataset — Shakespeare's *Hamlet* text saved from the NLTK Gutenberg corpus |
| `next_word_lstm.h5` | Original pre-trained LSTM model |
| `next_word_lstm_model_with_early_stopping.h5` | Original pre-trained LSTM model trained with early stopping |
| `tokenizer.pickle` | The fitted tokenizer (word ↔ index mapping) needed to use the models |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Requirements

From `requirements.txt`: TensorFlow 2.15.0, pandas, NumPy, scikit-learn, TensorBoard, Matplotlib, Streamlit, and SciKeras. The notebook additionally uses NLTK (for the Gutenberg corpus), which is pre-installed in Google Colab.

## How to Run the Notebook Yourself

1. Open the Colab link at the top of this README, **or** upload `experiments.ipynb` to [Google Colab](https://colab.research.google.com/) yourself.
2. Use **Runtime → Run all**. The cells run in order: the data is downloaded and saved, both models are trained (expect roughly 6–7 minutes total), tested, and saved.
3. The trained `next_word_lstm.h5`, `next_word_gru.h5`, and `tokenizer.pickle` files appear in the Colab file panel and can be downloaded from there.
