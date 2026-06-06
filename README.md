# A-Hybrid-Transformer-TextGCN-Framework-for-Depression-Detection-and-Severity-Classification-
A HYBRID TRANSFORMER–TEXTGCN FRAMEWORK FOR DEPRESSION DETECTION AND SEVERITY CLASSIFICATION FROM SOCIAL MEDIA TEXT USING EMOJI-BASED EMOTIONAL REPRESENTATION


## Overview

Depression is one of the most common mental health disorders worldwide, affecting millions of individuals and often remaining undiagnosed due to the dependence on traditional clinical assessments. This project presents a Hybrid Transformer–TextGCN framework for automated depression detection and severity classification using social media text and emoji-based emotional representations.

The proposed framework combines transformer-based contextual learning with graph-based semantic modelling to capture linguistic, emotional, and structural relationships present in online textual communication. The system is designed to perform both binary depression detection and multi-class severity classification within a unified architecture.

---

## Problem Statement

Traditional depression diagnosis relies on interviews, questionnaires, and clinical observation, which may not always be accessible or scalable. Social media platforms provide a large volume of user-generated textual content that can reveal behavioural and emotional patterns associated with depression.

The objective of this project is to develop an AI-driven framework capable of:

* Detecting depression from social media text.
* Classifying depression severity levels.
* Improving cross-platform generalization.
* Supporting early mental health screening through automated analysis.

---

## Datasets

The framework was evaluated using multiple publicly available datasets collected from:

* Twitter
* Reddit
* Online Web Forums
* Cross-platform Social Media Sources 

The datasets contain both binary labels (Depressed / Non-Depressed) and severity labels (Minimal, Mild, Moderate, Severe). Cross-dataset evaluation was performed to improve robustness and real-world applicability.

---

## Methodology

### Data Preprocessing

* Text Cleaning
* Normalization
* Tokenization
* Emoji Extraction
* Dataset Labeling

### Transformer-Based Contextual Learning

The project utilizes multiple transformer models:

* BERT
* RoBERTa
* MentalBERT
* MentalRoBERTa
* DistilRoBERTa

These models generate contextual embeddings that capture semantic relationships within textual sequences.

### Emoji-Based Emotional Feature Extraction

Emoji representations are extracted and incorporated as emotional indicators to improve affective understanding and sentiment interpretation from social media content.

### TextGCN Semantic Modelling

A Text Graph Convolutional Network (TextGCN) is used to model document-word relationships through graph construction using:

* PMI (Pointwise Mutual Information)
* TF-IDF (Term Frequency–Inverse Document Frequency)

This enables the framework to learn both local and global semantic dependencies.

### Feature Fusion and Classification

Transformer embeddings, emoji-based features, and graph representations are fused into a joint representation for:

* Binary Depression Detection
* Depression Severity Classification

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Streamlit
* BERT
* RoBERTa
* MentalBERT
* MentalRoBERTa
* DistilRoBERTa
* TextGCN
* NLP
* Deep Learning
* Graph Neural Networks
* Sentiment Analysis

---

## Key Features

* Hybrid Transformer–TextGCN Architecture
* Emoji-Based Emotional Representation
* Binary Depression Detection
* Severity Classification
* Cross-Dataset Learning
* Semantic Graph Modelling
* Real-Time Streamlit Deployment
* Mental Health Recommendation Support

---

## Experimental Results

### Cross-Dataset Evaluation Performance

* Accuracy: 98.7%
* Precision: 99.0%
* Recall: 98.7%
* F1-Score: 98.8%

### Key Findings

* Emoji-based emotional features improved affective understanding.
* TextGCN enhanced document-word semantic learning.
* Cross-dataset training improved generalization across platforms.
* The hybrid architecture outperformed standalone transformer approaches on challenging datasets.
* The framework demonstrated strong robustness for depression detection and severity assessment.

---

## Real-Time Prototype

A Streamlit-based web application was developed to demonstrate practical deployment of the framework. The system allows users to enter textual inputs and receive:

* Depression Status Prediction
* Severity Classification
* Confidence Scores
* Personalized Recommendations

---

## Applications

* Early Depression Screening
* Mental Health Monitoring
* Healthcare Decision Support
* Social Media Behaviour Analysis
* AI-Assisted Mental Health Assessment

---

## Future Work

Future improvements include:

* Multimodal Learning (Text, Speech, Facial Expressions, Physiological Signals)
* Explainable AI (XAI)
* Multilingual Depression Detection
* Cross-Lingual Learning
* Optimized Real-Time Deployment
* Clinical Decision Support Integration

---

## Disclaimer

This project is intended to support mental health analysis and early screening. It should not be used as a replacement for professional medical diagnosis, psychiatric evaluation, or clinical judgment.

