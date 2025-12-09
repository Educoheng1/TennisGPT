from enum import Enum
from typing import List, Optional, Literal
import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import os
from openai import OpenAI

# Load momentum model (if available)
MOMENTUM_MODEL = None
try:
    model_path = Path(__file__).parent / "ml" / "momentum_model.joblib"
    if model_path.exists():
        MOMENTUM_MODEL = joblib.load(model_path)
        print("Loaded momentum model from:", model_path)
    else:
        print("Momentum model file not found at:", model_path)
except Exception as e:
    print("Failed to load momentum model:", e)
    MOMENTUM_MODEL = None

# Load commentary retrieval index (if available)
COMMENTARY_INDEX = None
try:
    commentary_path = Path(__file__).parent / "ml" / "commentary_index.joblib"
    if commentary_path.exists():
        COMMENTARY_INDEX = joblib.load(commentary_path)
        print("Loaded commentary index from:", commentary_path)
    else:
        print("Commentary index not found at:", commentary_path)
except Exception as e:
    print("Failed to load commentary index:", e)
    COMMENTARY_INDEX = None


# This is what main.py is trying to import:
router = APIRouter(prefix="/ai", tags=["ai"])


# --------- ENUMS ---------

class CommentaryMode(str, Enum):
    LIVE_UPDATE = "live_update"
    LINE_FINAL = "line_final"
    DUAL_FINAL = "dual_final"


# --------- SHARED CONTEXT MODELS ---------

class BaseMatchContext(BaseModel):
    match_id: Optional[str] = None

    # teams
    team_home: str
    team_away: str

    # event info
    event_type: Optional[str] = "dual"   # "dual" or "tournament"
    round_info: Optional[str] = None     # e.g. "Round of 16", "Quarterfinal"
    location: Optional[str] = None
    date: Optional[str] = None           # you can switch to datetime later


class LiveUpdatePayload(BaseMatchContext):
    match_type: str                      # "singles" or "doubles"
    home_players: List[str]
    away_players: List[str]

    current_set: str                     # "first", "second", "third"
    set_score_summary: Optional[str] = None   # e.g. "home leads 1-0 in sets"
    game_score: str                      # e.g. "3-1"

    event_tag: Optional[str] = None      # e.g. "home_broke_serve", "away_held_serve"


class LineFinalPayload(BaseMatchContext):
    match_type: Literal["singles", "doubles"]
    home_players: List[str]
    away_players: List[str]
    final_score: str               # e.g. "6-3 6-4" or "6-3 3-6 7-5"
    winner_side: Literal["home", "away"]
    key_moments: Optional[List[str]] = None   # short phrases like "broke at 3–3", etc.


class DualFinalPayload(BaseMatchContext):
    overall_score: str             # e.g. "4-3", "5-2"
    winner_side: Literal["home", "away"]
    finished_lines: Optional[List[str]] = None
    # e.g. [
    #   "Cohen def. Player X 6-3 6-4 at line 1 singles",
    #   "Saint Leo took the doubles point",
    # ]
    big_moments: Optional[List[str]] = None
    # e.g. ["Cohen clinched at 3-all", "Team came back from losing doubles point"]

# --------- REQUEST / RESPONSE WRAPPERS ---------

class CommentaryRequest(BaseModel):
    mode: CommentaryMode

    live_update: Optional[LiveUpdatePayload] = None
    line_final: Optional[LineFinalPayload] = None
    dual_final: Optional[DualFinalPayload] = None


class CommentaryResponse(BaseModel):
    text: str


# --------- "FAKE BRAIN" HELPERS (no GPT yet) ---------
def generate_line_final_text(data: LineFinalPayload) -> str:
    # Figure out winner/loser names and team
    home_name = " & ".join(data.home_players) if data.match_type == "doubles" else data.home_players[0]
    away_name = " & ".join(data.away_players) if data.match_type == "doubles" else data.away_players[0]

    if data.winner_side == "home":
        winner_name = home_name
        loser_name = away_name
        winner_team = data.team_home
        loser_team = data.team_away
    else:
        winner_name = away_name
        loser_name = home_name
        winner_team = data.team_away
        loser_team = data.team_home

    score_str = data.final_score.strip()
    sets = score_str.split()
    straight_sets = len(sets) == 2

    pieces: List[str] = []

    # Opening sentence
    if data.match_type == "singles":
        pieces.append(
            f"{winner_name} closes out {loser_name} {score_str}, "
            f"picking up a big point for {winner_team}."
        )
    else:  # doubles
        pieces.append(
            f"{winner_name} take the doubles win over {loser_name} {score_str}, "
            f"putting {winner_team} on the board."
        )

    # Add a note about how the match felt
    if straight_sets:
        pieces.append("It was a solid performance in straight sets, with the winner controlling the key moments.")
    else:
        pieces.append("It was a tight three–set battle, with momentum swinging on a few big points.")

    # Use any key_moments you pass from your main app
    if data.key_moments:
        # Take up to 2 moments so it doesn't get too long
        km = " ".join(data.key_moments[:2])
        pieces.append(f"Key stretch: {km}")

    return " ".join(pieces)


def generate_dual_final_text(data: DualFinalPayload) -> str:
    # Figure out which team won
    if data.winner_side == "home":
        winner_team = data.team_home
        loser_team = data.team_away
    else:
        winner_team = data.team_away
        loser_team = data.team_home

    pieces: List[str] = []

    # Opening
    pieces.append(
        f"{winner_team} takes the dual {data.overall_score} over {loser_team}, "
        f"wrapping up the match in {data.location or 'today'}."
    )

    # Mention finished lines if you provide them
    if data.finished_lines:
        # e.g. "Cohen def. Player X 6-3 6-4 at line 1 singles"
        summary_lines = data.finished_lines[:3]
        joined = " ".join(summary_lines)
        pieces.append(f"Singles and doubles results: {joined}")

    # Big moments
    if data.big_moments:
        highlight = " ".join(data.big_moments[:3])
        pieces.append(f"Key moments in the dual: {highlight}")

    # Closing remark
    pieces.append("Overall, it was a strong team effort with energy from every court.")

    return " ".join(pieces)

def predict_momentum_for_live_update(data: LiveUpdatePayload) -> str | None:
    """
    Use the trained momentum model to predict the momentum_label
    for the current live update. Returns a string label or None.
    """
    if MOMENTUM_MODEL is None:
        return None

    set_map = {
        "first": 1,
        "second": 2,
        "third": 3,
    }
    set_number = set_map.get((data.current_set or "").lower(), 1)

    home_games = 0
    away_games = 0
    try:
        parts = (data.game_score or "").replace(" ", "").split("-")
        if len(parts) == 2:
            home_games = int(parts[0])
            away_games = int(parts[1])
    except Exception:
        pass

    set_score_summary = data.set_score_summary or "sets_tied"

    row = {
        "set_number": set_number,
        "home_games": home_games,
        "away_games": away_games,
        "event_tag": data.event_tag or "balanced_point",
        "set_score_summary": set_score_summary,
    }

    df = pd.DataFrame([row])

    try:
        pred = MOMENTUM_MODEL.predict(df)[0]
        return str(pred)
    except Exception as e:
        print("Momentum prediction error:", e)
        return None
def retrieve_commentary_line(
    momentum_label: str | None,
    data: LiveUpdatePayload,
) -> str | None:
    """
    Use the TF-IDF commentary index to find the best ESPN-style line
    for the current situation, with a bias toward the correct side
    (home vs away) based on event_tag or momentum.
    """
    if COMMENTARY_INDEX is None:
        return None

    vectorizer = COMMENTARY_INDEX["vectorizer"]
    matrix = COMMENTARY_INDEX["matrix"]
    df = COMMENTARY_INDEX["df"]

    # Decide preferred situation_tag based on event_tag / momentum
    preferred_tag = None

    if data.event_tag == "home_broke_serve":
        preferred_tag = "home_broke_serve"
    elif data.event_tag == "away_broke_serve":
        preferred_tag = "away_broke_serve"
    elif momentum_label in {
        "home_dominating",
        "away_dominating",
        "home_comeback",
        "away_comeback",
        "tight_set",
        "balanced",
        "momentum_shift",
        "pressure_point",
        "closing_in",
    }:
        preferred_tag = momentum_label

    # Build query text describing the moment
    parts = []
    if momentum_label:
        parts.append(momentum_label.replace("_", " "))
    if data.event_tag:
        parts.append(data.event_tag.replace("_", " "))
    if data.current_set:
        parts.append(f"{data.current_set} set")
    if data.game_score:
        parts.append(f"score {data.game_score}")

    query_text = " | ".join(parts) if parts else "balanced"

    try:
        q_vec = vectorizer.transform([query_text])

        # If we have a preferred_tag, restrict to those rows first
        candidate_indices = None
        if preferred_tag is not None and "situation_tag" in df.columns:
            mask = df["situation_tag"] == preferred_tag
            if mask.any():
                # Only use matching rows
                sub_matrix = matrix[mask]
                sims = cosine_similarity(q_vec, sub_matrix)[0]
                # Map back to original indices
                matching_idx = df[mask].index.to_list()
                best_local = sims.argmax()
                best_idx = matching_idx[best_local]
            else:
                # No matching tag → fall back to all rows
                sims = cosine_similarity(q_vec, matrix)[0]
                best_idx = sims.argmax()
        else:
            # No preferred tag → use all rows
            sims = cosine_similarity(q_vec, matrix)[0]
            best_idx = sims.argmax()

        row = df.iloc[best_idx]
        base_text = str(row["commentary_text"])

        # Swap generic names for actual players if available
        home_name = data.home_players[0] if data.home_players else "the home player"
        away_name = data.away_players[0] if data.away_players else "the away player"

        text = base_text
        text = text.replace("Nico Cohen", home_name)
        text = text.replace("Cohen", home_name.split()[-1] if " " in home_name else home_name)
        text = text.replace("Player X", away_name)

        return text
    except Exception as e:
        print("Commentary retrieval error:", e)
        return None


def predict_momentum_for_live_update(data: LiveUpdatePayload) -> str | None:
    """
    Use the trained momentum model to predict the momentum_label
    for the current live update. Returns a string label or None.
    """

    if MOMENTUM_MODEL is None:
        return None

    # Map set string → number
    set_map = {
        "first": 1,
        "second": 2,
        "third": 3,
    }
    set_number = set_map.get(data.current_set.lower(), 1)

    # Parse game_score like "3-1" → home_games=3, away_games=1
    home_games = 0
    away_games = 0
    try:
        parts = data.game_score.replace(" ", "").split("-")
        if len(parts) == 2:
            home_games = int(parts[0])
            away_games = int(parts[1])
    except Exception:
        pass

    # Default set_score_summary if not provided
    set_score_summary = data.set_score_summary or "sets_tied"

    row = {
        "set_number": set_number,
        "home_games": home_games,
        "away_games": away_games,
        "event_tag": data.event_tag or "balanced_point",
        "set_score_summary": set_score_summary,
    }

    df = pd.DataFrame([row])

    try:
        pred = MOMENTUM_MODEL.predict(df)[0]
        return str(pred)
    except Exception as e:
        print("Momentum prediction error:", e)
        return None
    
def generate_live_update_text(data: LiveUpdatePayload) -> str:
    """
    Generate live update commentary using:
    1) Momentum classifier
    2) Retrieval-based ESPN-style lines
    3) Simple rule-based fallback if needed
    """
    # 1) Predict momentum
    momentum_label = predict_momentum_for_live_update(data)
    print("Momentum prediction:", momentum_label)

    # 2) Try retrieval-based commentary
    retrieved = retrieve_commentary_line(momentum_label, data)
    if retrieved:
        return retrieved

    # 3) Fallback logic if index is missing or retrieval fails
    player = data.home_players[0] if data.home_players else "The home player"
    base = f"{player} is up {data.game_score} in the {data.current_set} set"

    if momentum_label == "home_dominating":
        return base + ", looking in control."
    elif momentum_label == "away_dominating":
        return base + ", but under real pressure as the opponent pushes back."
    elif momentum_label == "home_comeback":
        return base + ", fighting back strongly after trailing earlier."
    elif momentum_label == "away_comeback":
        return base + ", with the opponent clawing their way back into the set."
    elif momentum_label == "tight_set":
        return base + ", in what’s turning into a tight set."
    else:
        if data.event_tag == "home_broke_serve":
            return f"{player} breaks and leads {data.game_score} in the {data.current_set} set, looking strong."
        elif data.event_tag == "away_broke_serve":
            return f"{player} gets broken and now trails {data.game_score} in the {data.current_set} set."
        else:
            return base + "."

from typing import List

def generate_line_final_text(data: LineFinalPayload) -> str:
    # Names for singles/doubles
    home_name = " & ".join(data.home_players) if data.match_type == "doubles" else data.home_players[0]
    away_name = " & ".join(data.away_players) if data.match_type == "doubles" else data.away_players[0]

    if data.winner_side == "home":
        winner_name = home_name
        loser_name = away_name
        winner_team = data.team_home
        loser_team = data.team_away
    else:
        winner_name = away_name
        loser_name = home_name
        winner_team = data.team_away
        loser_team = data.team_home

    score_str = (data.final_score or "").strip()
    sets = score_str.split()
    straight_sets = len(sets) == 2

    pieces: List[str] = []

    # Opening sentence
    if data.match_type == "singles":
        pieces.append(
            f"{winner_name} closes out {loser_name} {score_str}, "
            f"picking up a big point for {winner_team}."
        )
    else:  # doubles
        pieces.append(
            f"{winner_name} take the doubles win over {loser_name} {score_str}, "
            f"putting {winner_team} on the board."
        )

    # Simple feel of the match
    if straight_sets:
        pieces.append("It was a solid performance in straight sets, with the winner controlling the key moments.")
    else:
        pieces.append("It was a tight three–set battle, with momentum swinging on a few big points.")

    # Optional key moments if you send them
    if data.key_moments:
        km = " ".join(data.key_moments[:2])
        pieces.append(f"Key stretch: {km}")

    return " ".join(pieces)


def generate_dual_final_text(data: DualFinalPayload) -> str:
    if data.winner_side == "home":
        winner_team = data.team_home
        loser_team = data.team_away
    else:
        winner_team = data.team_away
        loser_team = data.team_home

    pieces: List[str] = []

    # Opening line
    pieces.append(
        f"{winner_team} takes the dual {data.overall_score} over {loser_team}, "
        f"wrapping things up in {data.location or 'today'}."
    )

    # Mention some individual results if provided
    if data.finished_lines:
        summary_lines = data.finished_lines[:3]
        joined = " ".join(summary_lines)
        pieces.append(f"Singles and doubles results: {joined}")

    # Big moments across the dual
    if data.big_moments:
        highlight = " ".join(data.big_moments[:3])
        pieces.append(f"Key moments in the dual: {highlight}")

    # Closing remark
    pieces.append("Overall, it was a strong team effort with energy from every court.")

    return " ".join(pieces)


# --------- MAIN ENDPOINT ---------

@router.post("/commentary", response_model=CommentaryResponse)
async def generate_commentary(payload: CommentaryRequest) -> CommentaryResponse:
    if payload.mode == CommentaryMode.LIVE_UPDATE:
        if payload.live_update is None:
            raise HTTPException(status_code=400, detail="live_update data required for LIVE_UPDATE mode")
        text = generate_live_update_text(payload.live_update)

    elif payload.mode == CommentaryMode.LINE_FINAL:
        if payload.line_final is None:
            raise HTTPException(status_code=400, detail="line_final data required for LINE_FINAL mode")
        text = generate_line_final_text(payload.line_final)

    elif payload.mode == CommentaryMode.DUAL_FINAL:
        if payload.dual_final is None:
            raise HTTPException(status_code=400, detail="dual_final data required for DUAL_FINAL mode")
        text = generate_dual_final_text(payload.dual_final)

    else:
        raise HTTPException(status_code=400, detail="Unknown mode")

    return CommentaryResponse(text=text)

def call_tennis_gpt(mode: CommentaryMode, payload: BaseMatchContext, extra: dict) -> str:
    """
    Generic helper that calls GPT-4.1-mini to generate tennis commentary.
    `mode` is one of LIVE_UPDATE / LINE_FINAL / DUAL_FINAL.
    `payload` is the base match context (teams, event info, etc.).
    `extra` contains mode-specific fields (scores, players, etc.).
    """

    system_prompt = """
You are TennisGPT, an assistant that writes short, clear tennis commentary 
for a college tennis match tracker.

Goals:
- Use simple, clean English.
- Sound slightly hype and engaging, but not cringe.
- Never invent scores or events that are not in the input.
- Keep names and scores exactly as they appear in the data.
- You have three modes:
  1) "live_update" → 1 sentence describing the current situation.
  2) "line_final" → 2-4 sentences summarizing a single completed match.
  3) "dual_final" → 3-6 sentences summarizing the completed dual match.
If the data is incomplete or unclear, keep the commentary generic instead of guessing.
""".strip()

    user_content = {
        "mode": mode.value,
        "context": {
            "match_id": payload.match_id,
            "team_home": payload.team_home,
            "team_away": payload.team_away,
            "event_type": payload.event_type,
            "round_info": payload.round_info,
            "location": payload.location,
            "date": payload.date,
        },
        "data": extra,
        "instructions": "Generate only the commentary text. Do not explain what you are doing."
    }

    # Use chat.completions API (messages-based)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or "gpt-4o-mini" if you prefer
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ],
        max_tokens=120,
        temperature=0.7,
    )

    text = response.choices[0].message.content
    return text.strip()
