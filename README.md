# MediVision-DL 🩺

This is a project I started to explore how **Transfer Learning** can be applied to medical diagnostics. Specifically, I wanted to see if a lightweight model like MobileNetV3 could accurately categorize skin lesions.

## Why this project?
I’m interested in the intersection of AI and healthcare (specifically for **Track 2** applications). Most medical AI models are too heavy for real-world clinical use on simple hardware. I built this to test efficient architectures.

## What's inside?
- `model.py`: The architecture logic (Transfer learning from ImageNet).
- `app.py`: A Gradio-based interface so I could test it with real images immediately.
- `requirements.txt`: All the stuff you need to install.

## My Learnings & Challenges
1. **Data Imbalance:** During research, I found that datasets like HAM10000 are very imbalanced. While this prototype uses pre-trained weights, future versions will need specific loss-weighting.
2. **Preprocessing:** I realized that medical images vary wildly in lighting, so standard normalization is just the starting point.

## How to run it
1. Install: `pip install -r requirements.txt`
2. Launch: `python app.py`

*Note: This is a research prototype and not for actual medical diagnosis.*