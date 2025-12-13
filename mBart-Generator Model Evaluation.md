##  Model Evaluation & Error Analysis

**Performance Summary:**
The generator model (Facebook mBART) currently achieves a **BLEU score of 4.56**, which is below the target threshold for usable translations. The **ROUGE-L score of 1.58** indicates a significant lack of structural coherence in the generated responses.


MODEL PERFORMANCE REPORT
🔹 BLEU Score:   4.56  (Higher is better, >15 is decent for Urdu)
🔹 chrF Score:   23.66  (Best metric for Urdu, aim for >40)
🔹 ROUGE-L:      1.58  (Sentence structure match)
🔹 METEOR:       20.04  (Synonym/Meaning match)


---  Qualitative Analysis (First 3 Samples) ---

Question: کووڈ-19 کی عام علامات کیا ہیں؟
Gold Ans: عام علامات میں بخار، کھانسی اور سانس لینے میں دشواری شامل ہیں۔
Model Ans: کووڈ-19 کے عام علامات میں rhinitis، وائرس اور وائرس کی علامات شامل ہیں، جن کے لیے rhinitis وائرس کی تشخیص ضروری ہے۔
--------------------------------------------------
Question: کووڈ-19 کی تشخیص کے لیے کون سا ٹیسٹ عام طور پر استعمال ہوتا ہے؟
Gold Ans: تشخیص کے لیے عام طور پر rRT-PCR سویب ٹیسٹ استعمال ہوتے ہیں۔
Model Ans: کووڈ-19 کے تشخیص کے لیے عام طور پر ٹیسٹ عام طور پر استعمال ہوتے ہیں مگر ہر مریض کے لیے خاص ٹیسٹ ضروری ہیں۔
--------------------------------------------------
Question: ہاتھوں کی صفائی وبا کے دوران کیوں ضروری ہے؟
Gold Ans: صابن اور پانی سے کم از کم 20 سیکنڈ تک ہاتھ دھونے سے وائرس کے پھیلاؤ کا خطرہ کم ہوتا ہے۔
Model Ans: وبا کے دوران ہاتھوں کی صفائی اور صفائی ضروری ہے تاکہ جلدی پہنچ سکے۔
--------------------------------------------------

**Primary Issues Identified:**

**Training Loss:** Dropped rapidly from `3.64` to `0.07` (near-perfect memorization).
***Validation Loss:** Diverged significantly, increasing from `3.28` (Epoch 1) to `3.75` (Epoch 4).
***Conclusion:** The model stopped learning useful patterns after **Epoch 1**. The subsequent epochs (2-4) only degraded the model's generalization capabilities.

1.  **Severe Overfitting:**
    * **Observation:** The training loss decreased rapidly to **0.073** (Epoch 4), while the validation loss *increased* from **3.28** (Epoch 1) to **3.74** (Epoch 4).
    * **Analysis:** The model has "memorized" the training data but fails to generalize to unseen data. This is common when fine-tuning large models like mBART on smaller datasets without sufficient regularization.

2.  **Degenerate Repetition (Hallucination):**
    * **Observation:** Qualitative analysis shows the model entering repetition loops (e.g., repeating *"rhinitis virus rhinitis"*).
    * **Analysis:** This "repetition penalty" issue occurs when the decoder gets stuck in a high-probability loop. It suggests the need for `repetition_penalty` in the generation parameters or Early Stopping to prevent the model from degrading into these loops.


Attempted Fixes:

training_args = Seq2SeqTrainingArguments(
    # ... your other args ...
    num_train_epochs=5,           # Keep this, but let EarlyStopping cut it short
    load_best_model_at_end=True,  # CRITICAL: Revert to the best epoch (Epoch 1)
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,            # Helps prevent overfitting
    metric_for_best_model="eval_loss",
    
    # GENERATION ARGUMENTS (Fixes the "rhinitis" loop)
    generation_config=GenerationConfig(
        max_new_tokens=128,
        repetition_penalty=1.2,   # Penalizes repeating words
        no_repeat_ngram_size=3,   # Prevents 3-word phrase repeats
    )
)

Model still provides poor accuracy.

**Conclusion:**
Either switch to a different API mBart is good for translation but it isnt recognizing the propper context.
The model requires regularization (higher weight decay, dropout) and "Early Stopping" to prevent overfitting. We must also adjust the decoding strategy (beam search parameters) to fix the repetition loops.
