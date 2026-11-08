# toxic-comment_detection
A multi-label classification model for detecting toxic comments (NSFW, hate speech, and bullying) in Telegram chats using DistilBERT.

# Description
Toxic comments (pornographic, offensive, racist, harassing) are a widespread issue on social media. We trained a DistilBERT model, a variant of the BERT architecture, on our prepared dataset. The model is fine-tuned to detect toxic comments that fall under three categories: NSFW, bullying and hate speech. Essentially, our model is a multi-label fine-tuned model. We used DistilBERT in order to lower computation power, and to see the results of a small language model.

# Installation
In order to run the project one needs to use the following Python libraries using package manager pip:
'''pip install Transformers torch pandas lightning numpy pickle'''

# Usage
The model is availabe in my Hugging Face profile: https://huggingface.co/Yoav-Yosef/toxic-comment-detection-model, and is available to download.

# Dataset
The dataset used for training and evaluating the DistilBERT model is the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle. This dataset contains a collection of comments labeled across six categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate.

**Preprocessing Steps:**

1. **Data Cleaning**: Extensive text cleaning was performed to remove noise and irrelevant information from the comments. This included:
   - Removing HTML/XML tags, special characters, URLs, and links
   - Converting text to lowercase
   - Removing non-English text using the `langdetect` library
   - Tokenization and stopword removal using the `spacy` library
   - Removing numerical values, curly braces, and colons

2. **Handling Empty Rows**: Rows with empty comments were removed from the dataset using the `remove_empty_rows` function.

3. **Column Preprocessing**:
   - A new column 'bullying' was created by taking the maximum value of the 'threat' and 'insult' columns.
   - The 'obscene' column was renamed to 'nsfw' (Not Safe For Work).
   - The 'identity_hate' column was renamed to 'hate_speech'.
   - The 'threat' and 'insult' columns were dropped.

After preprocessing, the dataset was split into training and testing sets, with the cleaned data saved as CSV files for further use.

**Final Dataset:**
The final dataset used for training and evaluation consists of the following columns:

- `comment_text`: The text of the comment.
- `nsfw`: A binary label indicating whether the comment contains Not Safe For Work (NSFW) content.
- `hate_speech`: A binary label indicating whether the comment contains hate speech.
- `bullying`: A binary label indicating whether the comment contains bullying content.

The value counts for each label in the training set are visualized using the `visualize_columns` function, providing insights into the class distribution and potential imbalances.

By performing these preprocessing steps, the dataset was cleaned and prepared for training the DistilBERT model, ensuring that the model receives high-quality input data and learns to classify toxic comments accurately.

# Model Architecture - DistilBERT
DistilBERT was chosen for this project due to its reduced computational requirements and faster inference speed compared to the larger BERT model, while still maintaining high performance. The specific configuration used for the DistilBERT model in this project is as follows:

- Pretrained Model: `distilbert-base-uncased`
- Maximum Sequence Length: 512
- Learning Rate: 2e-5
- Batch Size: 16
- Number of Epochs: 5

The rationale behind choosing DistilBERT over other models like BERT or RoBERTa is twofold:

1. **Reduced Computation Requirements**: DistilBERT is a distilled version of the larger BERT model, making it significantly smaller in size (66M parameters compared to BERT's 110M parameters). This reduced model size translates to lower computational requirements, making it more feasible to train and deploy the model on resource-constrained environments.

2. **Faster Inference**: Due to its smaller size, DistilBERT can perform inference much faster than its larger counterparts. This is particularly beneficial for real-time applications or scenarios where low latency is crucial.

Despite its smaller size, DistilBERT has been shown to achieve performance close to that of BERT on various natural language processing tasks, making it an attractive choice for this multi-label classification problem.

# Evaluation
The performance of the DistilBERT model was evaluated on a held-out test set. The following evaluation metrics were chosen:

| Metric    | Value  |
|-----------|--------|
| AUROC     | 0.9304 |
| F1-Score  | 0.7930 |
| Loss      | 0.6311 |

The Area Under the Receiver Operating Characteristic (AUROC) is a commonly used metric for evaluating binary and multi-label classification models. It measures the model's ability to distinguish between positive and negative instances across different classification thresholds. An AUROC score of 0.9304 indicates that the model performs well in separating the toxic comments from non-toxic ones.
The F1-Score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance. It is particularly useful in multi-label classification tasks where both precision (avoiding false positives) and recall (capturing true positives) are important. The F1-Score of 0.7930 suggests that the model achieves a good balance between these two metrics.
The Loss value represents the overall error or cost function minimized during the training process. A lower loss value generally indicates better model performance, but it should be interpreted in conjunction with other metrics like AUROC and F1-Score.
These evaluation metrics were chosen to provide a comprehensive assessment of the model's performance in detecting toxic comments accurately. The AUROC and F1-Score are particularly informative for this multi-label classification task, as they capture the model's ability to identify toxic comments while minimizing false positives and false negatives.

# Contact
email: yoav.yosef1@gmail.com
