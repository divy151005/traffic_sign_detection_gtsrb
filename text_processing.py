"""Text cleaning, correction, and structuring for OCR-primary guide sign recognition."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List, Optional, Sequence, Tuple


MIN_TOKEN_LENGTH = 3
MIN_DISTANCE = 5
MAX_DISTANCE = 500
FUZZY_CUTOFF = 0.78
MIN_FUSED_CONFIDENCE = 0.45
HIGH_CONFIDENCE_TOKEN = 0.83
NOISE_TOKENS = {
    "THE",
    "AND",
    "LANE",
    "ROAD",
    "AHEAD",
    "EXIT",
    "TOLL",
    "SIGN",
    "BOARD",
    "ONLY",
    "SLOW",
    "SPEED",
}
DIR_TOKENS = {
    "LEFT",
    "RIGHT",
    "STRAIGHT",
    "AHEAD",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "UP",
    "DOWN",
}
DEFAULT_VOCABULARY = {
    "AGRA",
    "ALIGARH",
    "AMBALA",
    "AMRITSAR",
    "BAGHPAT",
    "BATALA",
    "CHANDIGARH",
    "DELHI",
    "FARIDABAD",
    "FIROZPUR",
    "GURUGRAM",
    "HISAR",
    "JALANDHAR",
    "KARNAL",
    "KURUKSHETRA",
    "LUDHIANA",
    "MATHURA",
    "MODINAGAR",
    "MURADNAGAR",
    "PANIPAT",
    "PATIALA",
    "ROHTAK",
    "SONEPAT",
    "SURANA",
    "SUHANA",
    "YAMUNANAGAR",
}


@dataclass
class StructuredTextResult:
    raw_ocr: List[Dict[str, object]]
    cleaned_text: List[str]
    structured_output: str
    structured_entries: List[Dict[str, object]]
    rejected_ocr_noise: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw_ocr": self.raw_ocr,
            "cleaned_text": self.cleaned_text,
            "structured_output": self.structured_output,
            "structured_entries": self.structured_entries,
            "rejected_ocr_noise": self.rejected_ocr_noise,
        }


def summarize_partial_lines(lines: Sequence[str], limit: int = 3) -> str:
    seen: List[str] = []
    for line in lines:
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        seen.append(normalized)
        if len(seen) >= limit:
            break
    return " | ".join(seen)


def normalize_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\u0900-\u097F\u0A00-\u0A7F\s]", " ", text.upper())
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[A-Z]+|\d+|[\u0900-\u097F]+|[\u0A00-\u0A7F]+", normalize_text(text))


def clean_tokens(tokens: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        if token.isdigit():
            cleaned.append(token)
            continue
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if token in NOISE_TOKENS:
            continue
        cleaned.append(token)
    return cleaned


def normalize_direction(tokens: Sequence[str]) -> Optional[str]:
    if not tokens:
        return None
    joined = " ".join(tokens)
    if "STRAIGHT" in tokens and "AHEAD" in tokens:
        return "STRAIGHT AHEAD"
    if "AHEAD" in tokens:
        return "AHEAD"
    if joined:
        return joined
    return None


def build_vocabulary(raw_items: Sequence[Dict[str, object]], extra_terms: Optional[Sequence[str]] = None) -> List[str]:
    vocabulary = set(DEFAULT_VOCABULARY)
    if extra_terms:
        vocabulary.update(term.upper() for term in extra_terms)
    return sorted(vocabulary)


def is_known_place(place: str, vocabulary: Sequence[str]) -> bool:
    if not place:
        return False
    place_tokens = [token for token in tokenize_text(place) if not token.isdigit() and token not in DIR_TOKENS]
    if not place_tokens:
        return False
    for token in place_tokens:
        if token in vocabulary:
            continue
        if get_close_matches(token, vocabulary, n=1, cutoff=FUZZY_CUTOFF):
            continue
        return False
    return True


def correct_token(token: str, vocabulary: Sequence[str]) -> str:
    if token.isdigit() or re.search(r"[\u0900-\u097F\u0A00-\u0A7F]", token):
        return token
    if token in DIR_TOKENS:
        return token
    if token in vocabulary:
        return token
    matches = get_close_matches(token, vocabulary, n=1, cutoff=FUZZY_CUTOFF)
    if matches:
        return matches[0]

    best_match = token
    best_score = 0.0
    for candidate in vocabulary:
        score = SequenceMatcher(None, token, candidate).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match if best_score >= FUZZY_CUTOFF else token


def extract_numbers(tokens: Sequence[str]) -> List[int]:
    numbers: List[int] = []
    for token in tokens:
        if token.isdigit():
            value = int(token)
            if MIN_DISTANCE <= value <= MAX_DISTANCE:
                numbers.append(value)
    return numbers


def split_place_and_distance(tokens: Sequence[str]) -> Tuple[str, Optional[int]]:
    place_tokens = [token for token in tokens if not token.isdigit() and token not in DIR_TOKENS]
    numbers = extract_numbers(tokens)
    place = " ".join(place_tokens).strip()
    distance = numbers[0] if numbers else None
    return place, distance


def is_partial_place(place: str) -> bool:
    if not place:
        return False
    tokens = [token for token in tokenize_text(place) if not token.isdigit() and token not in DIR_TOKENS]
    return bool(tokens)


def extract_direction(tokens: Sequence[str]) -> Optional[str]:
    direction_tokens = [token for token in tokens if token in DIR_TOKENS]
    return normalize_direction(direction_tokens)


def filter_noise_items(raw_items: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    filtered: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    for item in raw_items:
        confidence = float(item.get("confidence", 0.0))
        fused_confidence = float(item.get("fused_confidence", confidence))
        source_count = int(item.get("ocr_source_count", 1))
        text = str(item.get("text", "")).strip()
        tokens = clean_tokens(tokenize_text(text))
        if not text:
            rejected.append({**dict(item), "rejection_reason": "empty_text"})
            continue
        if confidence < 0.50 and fused_confidence < MIN_FUSED_CONFIDENCE:
            rejected.append({**dict(item), "rejection_reason": "low_confidence"})
            continue
        if not tokens:
            rejected.append({**dict(item), "rejection_reason": "no_meaningful_tokens"})
            continue
        if source_count == 1 and confidence < HIGH_CONFIDENCE_TOKEN and not any(token.isdigit() for token in tokens):
            rejected.append({**dict(item), "rejection_reason": "single_source_low_confidence_text"})
            continue
        filtered.append(dict(item))
    return filtered, rejected


def sort_raw_items(raw_items: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        raw_items,
        key=lambda item: (
            float(item.get("y_center", 0.0)) // 30,
            float(item.get("x_center", 0.0)),
        ),
    )


def group_items_by_row(raw_items: Sequence[Dict[str, object]], row_threshold: float = 28.0) -> List[List[Dict[str, object]]]:
    rows: List[List[Dict[str, object]]] = []
    for item in sort_raw_items(raw_items):
        y_center = float(item.get("y_center", 0.0))
        if not rows:
            rows.append([item])
            continue
        last_row = rows[-1]
        row_y = sum(float(entry.get("y_center", 0.0)) for entry in last_row) / max(len(last_row), 1)
        if abs(y_center - row_y) <= row_threshold:
            last_row.append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda item: float(item.get("x_center", 0.0)))
    return rows


def dedupe_lines(lines: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for line in lines:
        normalized = normalize_text(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(line)
    return deduped


def dedupe_entries(entries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    best_by_key: Dict[Tuple[object, object, object], Dict[str, object]] = {}
    for entry in entries:
        key = (
            normalize_text(str(entry.get("place") or "")),
            entry.get("distance"),
            normalize_text(str(entry.get("direction") or "")),
        )
        current = best_by_key.get(key)
        current_score = float(current.get("fused_confidence") or 0.0) if current else -1.0
        candidate_score = float(entry.get("fused_confidence") or 0.0)
        if current is None or candidate_score >= current_score:
            best_by_key[key] = dict(entry)
    return list(best_by_key.values())


def structure_guide_sign_text(
    raw_items: Sequence[Dict[str, object]],
    extra_terms: Optional[Sequence[str]] = None,
) -> StructuredTextResult:
    filtered_items, rejected_noise = filter_noise_items(raw_items)
    sorted_items = sort_raw_items(filtered_items)
    vocabulary = build_vocabulary(sorted_items, extra_terms=extra_terms)

    cleaned_lines: List[str] = []
    structured_entries: List[Dict[str, object]] = []

    for row in group_items_by_row(sorted_items):
        pending_places: List[Tuple[str, float, Optional[str], float, str]] = []
        for item in row:
            tokens = tokenize_text(str(item.get("text", "")))
            tokens = clean_tokens(tokens)
            corrected_tokens = [correct_token(token, vocabulary) for token in tokens]
            if not corrected_tokens:
                continue

            cleaned_line = " ".join(corrected_tokens)
            cleaned_lines.append(cleaned_line)

            place, distance = split_place_and_distance(corrected_tokens)
            direction = extract_direction(corrected_tokens)
            x_center = float(item.get("x_center", 0.0))
            fused_confidence = float(item.get("fused_confidence", item.get("confidence", 0.0)))
            known_place = is_known_place(place, vocabulary) if place else False

            if place and distance is not None:
                entry = {
                    "place": place,
                    "distance": distance,
                    "direction": direction,
                    "fused_confidence": fused_confidence,
                    "place_confidence": "known" if known_place else "partial",
                }
                structured_entries.append(entry)
                continue

            if place:
                place_confidence = "known" if known_place else ("partial" if is_partial_place(place) else "missing")
                if place_confidence != "missing":
                    pending_places.append((place, x_center, direction, fused_confidence, place_confidence))
                continue

            if distance is not None and pending_places:
                best_index = min(range(len(pending_places)), key=lambda idx: abs(pending_places[idx][1] - x_center))
                matched_place, _, pending_direction, pending_confidence, place_confidence = pending_places.pop(best_index)
                structured_entries.append(
                    {
                        "place": matched_place,
                        "distance": distance,
                        "direction": direction or pending_direction,
                        "fused_confidence": max(fused_confidence, pending_confidence),
                        "place_confidence": place_confidence,
                    }
                )
            elif distance is not None:
                structured_entries.append(
                    {
                        "place": None,
                        "distance": distance,
                        "direction": direction,
                        "fused_confidence": fused_confidence,
                        "place_confidence": "missing",
                    }
                )

        for place, _, direction, fused_confidence, place_confidence in pending_places:
            structured_entries.append(
                {
                    "place": place,
                    "distance": None,
                    "direction": direction,
                    "fused_confidence": fused_confidence,
                    "place_confidence": place_confidence,
                }
            )

    cleaned_lines = dedupe_lines(cleaned_lines)
    structured_entries = dedupe_entries(structured_entries)
    paired_output: List[str] = []
    for entry in structured_entries:
        place = entry.get("place")
        distance = entry.get("distance")
        direction = entry.get("direction")
        parts = [str(part) for part in [place, distance, direction] if part not in (None, "", "None")]
        if parts:
            paired_output.append(" ".join(parts))

    structured_output = " | ".join(paired_output)
    if not structured_output and cleaned_lines:
        structured_output = summarize_partial_lines(cleaned_lines)
    return StructuredTextResult(
        raw_ocr=[dict(item) for item in sorted_items],
        cleaned_text=cleaned_lines,
        structured_output=structured_output,
        structured_entries=structured_entries,
        rejected_ocr_noise=rejected_noise,
    )
