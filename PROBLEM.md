# Assignment for Machine Learning Candidate: AI Safety Models Proof of Concept (POC)

## Overview

As a Machine Learning candidate, you are tasked with designing and implementing a Proof of Concept (POC) for a suite of **AI Safety Models** aimed at enhancing user safety in a conversational AI platform (e.g., similar to a chat application or social media messaging system). The POC should demonstrate your ability to lead the development of scalable, ethical ML solutions while considering real-world constraints such as real-time performance, data privacy, and model interpretability.

### Key Requirements

- **Abuse Language Detection:** Real-time identification of harmful, threatening, or inappropriate content in user-generated text.
- **Escalation Pattern Recognition:** Detection of when conversations are becoming emotionally dangerous, such as through repeated aggressive language or intensifying negativity.
- **Crisis Intervention:** AI recognition of severe emotional distress or self-harm indicators, with potential triggers for human intervention.
- **Content Filtering:** Age-appropriate content filtering for guardian-supervised accounts, ensuring content is suitable based on user age profiles.

Your POC should integrate these models into a cohesive system, such as a simple web-based chat simulator or a command-line tool, where inputs (e.g., simulated conversations) are processed in real-time or near-real-time. Focus on demonstrating end-to-end functionality, from data preprocessing to model inference and output handling.

---

## Deliverables

### 1. Code Repository

A complete, self-contained GitHub repository (or equivalent) with your POC implementation. Include:

- Source code for model training, inference, and integration.
- Any necessary datasets (use publicly available, anonymized datasets; do **not** include sensitive data).
- Documentation (e.g., `README.md`) explaining setup, dependencies, and how to run the POC.
- Scripts for model evaluation (e.g., metrics like precision, recall, F1-score for classification tasks).

### 2. 10-Minute Walkthrough Video

Record a video (e.g., via Loom, YouTube, or screen recording tool) no longer than 10 minutes explaining:

- Your overall code logic and architecture.
- Why you chose your specific approach (e.g., model selection, frameworks, handling of edge cases).
- Pros and cons of your approach (e.g., trade-offs in accuracy vs. speed, ethical considerations).
- Demonstration of the POC in action with sample inputs and outputs.
- Any assumptions made and potential improvements for production scaling.

**Provide a link to the video in your submission.**

### 3. Technical Report

A 2-4 page PDF document summarizing:

- High-level design decisions.
- Data sources and preprocessing steps.
- Model architectures and training details.
- Evaluation results with metrics.
- Discussion on leadership aspects, such as how you would guide a team to iterate on this POC.

---

## Technical Guidelines

- **Languages and Frameworks:** Use Python as the primary language. Leverage libraries such as scikit-learn, TensorFlow, PyTorch, Hugging Face Transformers, or spaCy for ML/NLP tasks. Ensure the POC runs on standard hardware (e.g., CPU-based for accessibility).
- **Real-Time Aspects:** Simulate real-time processing (e.g., via streaming inputs or low-latency inference).
- **Ethical Considerations:** Incorporate bias mitigation techniques and explain how your models handle fairness across demographics.
- **Scalability:** Design with modularity in mind, allowing for easy extension (e.g., adding more models or integrating with APIs).

---

## Resources and Tips

- Start with researching state-of-the-art approaches for each model (e.g., BERT-based classifiers for abuse detection).
- Prioritize integration: Show how the models work together (e.g., escalation detection triggering crisis intervention).
- Test thoroughly: Include edge cases like multilingual text, slang, or ambiguous language.
- If you encounter blockers, note them in your report with proposed solutions.

