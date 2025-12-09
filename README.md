Run server:
uvicorn main:app --reload --port 8001
🏆 TennisGPT
A Local AI Engine for Tennis Commentary, Match Summaries & Momentum Modeling

TennisGPT is a lightweight, fully local AI system that generates ESPN-style live commentary, match recaps, and dual match summaries for tennis.
It powers tennis scoring applications by providing real-time insights without relying on external APIs.
Built using FastAPI, scikit-learn, TF-IDF retrieval, and a custom momentum classifier, TennisGPT runs entirely offline.
This project is part of my long-term effort to combine tennis, AI, and software engineering into real, production-style tools.

📌 Features
🎤 Live Commentary (RAG + ML)
Generates short, dynamic ESPN-style updates every time the score changes
Uses a RandomForest momentum model trained on handcrafted tennis scenarios
Incorporates player names, game/set score, serving info, and momentum shifts
Retrieval-enhanced: commentary tone comes from a curated dataset of real examples

📄 Singles/Doubles Match Summary
Produces clean, match-recap paragraphs
Adapts tone based on scoreline & key moments
Automatically formats singles or doubles team names

🏅 Dual Match Summary (4–3, 5–2, etc.)
Full team-vs-team recap with finished courts, turning points, and final score
Perfect for college tennis duals or team events

⚙️ Fully Local & Offline
No OpenAI / no API keys required
All models and indexes are trained and stored locally
Designed to run inside a larger tennis match-tracking system

📂 Project Structure
TennisGPT/
│
├── main.py                     # FastAPI entry point
├── tennis_gpt_logic.py         # AI logic for live/commentary/summaries
│
├── data/
│   └── commentary_dataset.csv  # ESPN-style curated commentary dataset
│
└── ml/
    ├── data/
    │   └── momentum_samples.csv       # Training samples for momentum classifier
    │
    ├── train_momentum_model.py        # Trains RandomForest model
    ├── build_commentary_index.py      # Builds TF-IDF retrieval index
    ├── momentum_model.joblib          # Trained model (ignored by git)
    └── commentary_index.joblib        # Retrieval index (ignored by git)

🚀 Running the Project
1️⃣ Create & Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train AI Models (first time)
python ml/train_momentum_model.py
python ml/build_commentary_index.py

4️⃣ Start the API Server
uvicorn main:app --reload --port 8001


API base path:
http://localhost:8001/ai/commentary

🧪 API Examples
🎤 Live Update Example
{
  "mode": "live_update",
  "live_update": {
    "team_home": "Saint Leo Lions",
    "team_away": "Barry Buccaneers",
    "set_score": "3-1",
    "game_score": "40-15",
    "server": "home",
    "home_players": ["Nico Cohen"],
    "away_players": ["Player X"]
  }
}

🎾 Match Summary (Singles)
{
  "mode": "line_final",
  "line_final": {
    "team_home": "Saint Leo Lions",
    "team_away": "Barry Buccaneers",
    "match_type": "singles",
    "home_players": ["Nico Cohen"],
    "away_players": ["Player X"],
    "final_score": "6-3 6-4",
    "winner_side": "home",
    "key_moments": [
      "Cohen broke early for 3–1",
      "Saved two break points at 4–3"
    ]
  }
}

🏅 Dual Match Summary
{
  "mode": "dual_final",
  "dual_final": {
    "team_home": "Saint Leo Lions",
    "team_away": "Barry Buccaneers",
    "overall_score": "4-3",
    "winner_side": "home",
    "finished_lines": [
      "Cohen def Player X 6-3 6-4 at Line 1",
      "Saint Leo clinched the doubles point"
    ],
    "big_moments": [
      "Cohen clinched at 3-all"
    ]
  }
}

🧠 How TennisGPT Works
1) Momentum Classifier (ML)
A RandomForest model predicts states like:
home_dominating
balanced
away_dominating
home_comeback
away_comeback
Features include:
Set score
Game score
Serve indicator
Break advantage patterns
This provides the “tone” of commentary.

2) Commentary Retrieval (RAG)
TennisGPT uses a TF-IDF vectorizer over a curated dataset of tennis commentary lines.
For each live update:
The model selects the correct momentum state
A text search retrieves the closest example
Names, score, and context are substituted dynamically
This keeps commentary natural and ESPN-like without using LLMs.

3) FastAPI Service Layer
The API wraps the AI logic into one endpoint with three modes:
live_update → real-time commentary
line_final → match summary
dual_final → dual match recap
Designed to integrate directly with a tennis scoring website or app.

🎯 Why I Built This Project
As a competitive tennis player and software engineer, I wanted to create:
A production-style AI system
That works in real time,
Running fully offline,
Using ML + retrieval together (RAG),
And could plug into my Saint Leo Tennis Match Tracker.
This project demonstrates skills across:
Machine learning (modeling, features, training)
Retrieval-based AI pipelines
FastAPI backend design
Sports analytics
Real-time systems engineering
Clean modular architecture
It is a strong showcase of end-to-end applied AI engineering.

👤 Author
Eduardo Cohen
AI Engineer • Software Developer • NCAA Athlete
Passionate about tennis, machine learning, and building tools that combine both.