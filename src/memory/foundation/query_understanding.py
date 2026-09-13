"""Structured query understanding for ALLI v0.6.

Converts natural-language questions into a structured query representation
that later retrieval sessions can consume.

Different wording → Same structured query meaning.

Examples:
  "What phone do I have?"       → StructuredQuery(subject=user, relation=possess, object=phone)
  "Which mobile am I using?"    → StructuredQuery(subject=user, relation=use, object=mobile)
  "bro what phone I got?"       → StructuredQuery(subject=user, relation=possess, object=phone)
  "wat phone i hav?"            → StructuredQuery(subject=user, relation=possess, object=phone)

Design:
  - General mechanisms, not sentence-by-sentence rules
  - Paraphrase handling via lexical normalization
  - Entity, temporal, cardinality, constraint extraction
  - Negation/uncertainty representation
  - No retrieval or answer generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Structured Query Representation
# ---------------------------------------------------------------------------

@dataclass
class StructuredQuery:
    """Machine-readable representation of a user's query meaning.

    Fields:
        subject: Who/what the query is about (entity name or "user")
        relation: The type of relationship sought (possess, prefer, learn, use, etc.)
        object: The specific thing being asked about (phone, color, language, etc.)
        state: Current state qualifier (current, historical, etc.)
        time: Temporal qualifier (now, before, 2025, etc.)
        cardinality: Expected answer count (one, many, existence, count)
        entity: Resolved entity reference (name, type, relation_to_user)
        domain: Knowledge domain (possessions, interests, learning, identity, etc.)
        constraints: Additional filters (current, historical, specific entity, etc.)
        negated: Whether the query contains negation
        uncertain: Whether the query expresses uncertainty
        is_personal: Whether this is a personal memory query
        is_general: Whether this is a general knowledge query
        raw_query: Original user input
        normalized_query: Cleaned/normalized form
    """

    subject: str = "user"
    relation: str = "unknown"
    object: str = "unknown"
    state: str = "current"
    time: Optional[str] = None
    cardinality: str = "one"
    entity: Optional[str] = None
    domain: str = "unknown"
    constraints: list[str] = field(default_factory=list)
    negated: bool = False
    uncertain: bool = False
    is_personal: bool = True
    is_general: bool = False
    raw_query: str = ""
    normalized_query: str = ""

    def __post_init__(self) -> None:
        if not self.constraints:
            self.constraints = []

    @property
    def is_historical(self) -> bool:
        return self.state == "historical"

    @property
    def is_current(self) -> bool:
        return self.state == "current"

    @property
    def expects_many(self) -> bool:
        return self.cardinality in ("many", "count")

    @property
    def is_boolean(self) -> bool:
        return self.cardinality == "existence"

    def to_dict(self) -> dict:
        """Serialize to a dictionary for downstream consumption."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "state": self.state,
            "time": self.time,
            "cardinality": self.cardinality,
            "entity": self.entity,
            "domain": self.domain,
            "constraints": list(self.constraints),
            "negated": self.negated,
            "uncertain": self.uncertain,
            "is_personal": self.is_personal,
            "is_general": self.is_general,
        }


# ---------------------------------------------------------------------------
# Lexical Normalization — handles paraphrases, typos, slang, contractions
# ---------------------------------------------------------------------------

# Contraction expansion
_CONTRACTIONS = {
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bain't\b": "is not",
    r"\bwhat's\b": "what is",
    r"\bwat's\b": "what is",
    r"\bwhere's\b": "where is",
    r"\bwho's\b": "who is",
    r"\bhow's\b": "how is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bthey're\b": "they are",
    r"\byou're\b": "you are",
    r"\bwe're\b": "we are",
    r"\bhe's\b": "he is",
    r"\bshe's\b": "she is",
    r"\bit's\b": "it is",
    r"\blet's\b": "let us",
    r"\bgonna\b": "going to",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "got to",
    r"\bkinda\b": "kind of",
    r"\bsorta\b": "sort of",
    r"\bdunno\b": "do not know",
    r"\bimo\b": "in my opinion",
    r"\bimho\b": "in my humble opinion",
    r"\brn\b": "right now",
    r"\btbh\b": "to be honest",
    r"\bafk\b": "away from keyboard",
    r"\bbtw\b": "by the way",
    r"\bomg\b": "oh my god",
    r"\bngl\b": "not gonna lie",
    r"\bidk\b": "i do not know",
    r"\bty\b": "thank you",
    r"\bthx\b": "thanks",
    r"\bnp\b": "no problem",
}

# Slang / informal → formal mapping
_SLANG_MAP = {
    "bro": "",
    "bruh": "",
    "dude": "",
    "fam": "",
    "lit": "good",
    "vibe": "",
    "sus": "suspicious",
    "cap": "lie",
    "no cap": "truth",
    "slay": "do well",
    "periodt": "",
    "lowkey": "somewhat",
    "highkey": "very",
    "fr": "for real",
    "istg": "i swear to god",
    "ngl": "not gonna lie",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "rn": "right now",
    "rn": "right now",
    "af": "very",
    "/goat": "greatest",
    "wyd": "what are you doing",
    "wbu": "what about you",
    "hmu": "hit me up",
    "ikr": "i know right",
    "tbt": "throwback thursday",
    "fomo": "fear of missing out",
    "yolo": "you only live once",
    "afaik": "as far as i know",
    "iirc": "if i recall correctly",
    "afaict": "as far as i can tell",
    "tl;dr": "too long did not read",
}

# Typo corrections — common misspellings
_TYPO_CORRECTIONS = {
    "phne": "phone",
    "fone": "phone",
    "fhone": "phone",
    "moble": "mobile",
    "mobil": "mobile",
    "lapp": "laptop",
    "lpatop": "laptop",
    "compter": "computer",
    "computr": "computer",
    "colr": "color",
    # Common slang/"g" dropped verb forms → "-ing"
    "learnin": "learning",
    "learnn": "learning",
    "studyin": "studying",
    "doin": "doing",
    "usin": "using",
    "goin": "going",
    "talkin": "talking",
    "readin": "reading",
    "buildin": "building",
    "workin": "working",
    "colour": "color",
    "clr": "color",
    "favrite": "favorite",
    "favourite": "favorite",
    "favirote": "favorite",
    "favrotie": "favorite",
    "javscript": "javascript",
    "jvscript": "javascript",
    "pyhton": "python",
    "pythn": "python",
    "langauge": "language",
    "languge": "language",
    "progamming": "programming",
    "programing": "programming",
    "learing": "learning",
    "lerning": "learning",
    "studying": "learning",
    "prefernce": "preference",
    "prefrence": "preference",
    "prefernce": "preference",
    "intrst": "interest",
    "interst": "interest",
    "hoby": "hobby",
    "hobies": "hobbies",
    "skll": "skill",
    "skil": "skill",
    "goalz": "goals",
    "gols": "goals",
    "projct": "project",
    "porject": "project",
    "priorty": "priority",
    "priortiy": "priority",
    "habti": "habit",
    "habts": "habits",
    "previus": "previous",
    "previos": "previous",
    "previoud": "previous",
    "curent": "current",
    "currnet": "current",
    "currrent": "current",
    "beofre": "before",
    "befor": "before",
    "befroe": "before",
    "previoulsy": "previously",
    "previosly": "previously",
    "addres": "address",
    "adres": "address",
    "numbr": "number",
    "nmber": "number",
    "nubmer": "number",
    "wat": "what",
    "wht": "what",
    "wut": "what",
    "hav": "have",
    "hv": "have",
    "hve": "have",
    "hae": "have",
    "stil": "still",
    "lapto": "laptop",
    "liek": "like",
    "lik": "like",
    "lyk": "like",
    "lnow": "know",
    "kno": "know",
    "usng": "using",
    "usin": "using",
    "amke": "make",
    "mkae": "make",
    "doig": "doing",
    "doin": "doing",
    "hwo": "how",
    "whne": "when",
    "whe": "when",
    "wheer": "where",
    "whre": "where",
    "wnat": "want",
    "waht": "what",
    "favrit": "favorite",
    "fave": "favorite",
    "favrite": "favorite",
    "samung": "samsung",
    "samsng": "samsung",
    "rockin": "using",
    "slaps": "good",
    "mid": "average",
    "fr": "for real",
    "ngl": "not gonna lie",
    "bruh": "",
    "yo": "",
    "u": "you",
}

# Possessive word normalization
_POSSESSIVE_PATTERNS = [
    (r"\bmy\b", "my"),
    (r"\bmine\b", "my"),
    (r"\bi(?:'ve|\s+have)\s+got\b", "my"),
    (r"\bi\s+have\b", "my"),
    (r"\bi\s+own\b", "my"),
    (r"\bi\s+use\b", "my"),
    (r"\bi\s+am\s+using\b", "my"),
]

# Relation signal words
_RELATION_SIGNALS = {
    "possess": {
        "have", "own", "got", "possess", "possesses",
        "use", "uses", "using", "using currently",
        "carry", "carries", "carrying",
        "phone", "mobile", "laptop", "computer", "tablet", "device",
        "car", "bike", "watch", "camera", "earbuds", "headphones",
    },
    "prefer": {
        "like", "likes", "enjoy", "enjoys", "love", "loves",
        "dislike", "dislikes", "hate", "hates",
        "prefer", "prefers", "fancy", "fancies",
        "favorite", "favourite", "fav", "favrite",
        "color", "colour", "food", "music", "movie",
    },
    "learn": {
        "learn", "learning", "learns",
        "study", "studying", "studies",
        "know", "knows", "knowing",
        "teach", "teaching", "teaches",
        "course", "class", "tutorial",
    },
    "identity": {
        "name", "called", "nickname", "alias",
        "who am i", "who is",
    },
    "goal": {
        "goal", "goals", "aim", "aims",
        "want to become", "plan to", "aspire",
        "ambition", "ambitions", "dream", "dreams",
    },
    "skill": {
        "skill", "skills", "ability", "abilities",
        "proficient", "expert", "good at", "know how to",
    },
}

# Object extraction — common query objects
_OBJECT_PATTERNS = [
    # Devices
    (r"\b(phone|mobile|cell\s*phone|smartphone|handset)\b", "phone"),
    (r"\b(laptop|notebook)\b", "laptop"),
    (r"\b(computer|pc|desktop)\b", "computer"),
    (r"\b(tablet|ipad)\b", "tablet"),
    (r"\b(device|gadget)\b", "device"),
    (r"\b(car|vehicle|automobile)\b", "car"),
    (r"\b(bike|bicycle|motorcycle)\b", "bike"),
    (r"\b(watch|smartwatch)\b", "watch"),
    (r"\b(camera|webcam)\b", "camera"),
    (r"\b(earbuds|earphones|headphones|headset)\b", "audio"),
    (r"\b(keyboard|mouse|monitor|display)\b", "peripheral"),
    # Programming
    (r"\b(python|java|javascript|c\+\+|ruby|go|rust|swift|kotlin|php|sql|html|css|react|angular|vue|node|django|flask|spring|rails|laravel|docker|kubernetes|linux|windows|macos|android|ios|git|github|aws|azure|gcp)\b", "technology"),
    (r"\b(technologies|technology|tech)\b", "technology"),
    (r"\b(language|languages|programming\s+language|programming\s+languages)\b", "language"),
    (r"\b(frameworks?|libraries?|tools?|platforms?|stack)\b", "tool"),
    # Personal
    (r"\b(color|colour)\b", "color"),
    (r"\b(food|cuisine|meal)\b", "food"),
    (r"\b(movie|film|show|series)\b", "entertainment"),
    (r"\b(music|song|artist|band)\b", "music"),
    (r"\b(book|novel|author)\b", "reading"),
    (r"\b(hobby|hobbies|interest|interests)\b", "hobby"),
    # Education
    (r"\b(course|class|subject|topic|field)\b", "subject"),
    (r"\b(degree|diploma|certificate|program)\b", "credential"),
    (r"\b(school|college|university|institute)\b", "institution"),
    # Generic
    (r"\b(thing|stuff|item)\b", "item"),
]


def normalize_query(text: str) -> str:
    """Normalize a natural-language query for structured understanding.

    Handles:
      - Contraction expansion
      - Slang removal/expansion
      - Typo correction
      - Whitespace normalization
      - Punctuation removal
      - Case normalization
    """
    if not text:
        return ""

    result = text.strip().lower()

    # Remove punctuation except apostrophes (needed for contractions)
    result = re.sub(r"[^\w\s']", " ", result)

    # Expand contractions
    for pattern, replacement in _CONTRACTIONS.items():
        result = re.sub(pattern, replacement, result)

    # Expand slang
    words = result.split()
    expanded = []
    for word in words:
        if word in _SLANG_MAP:
            replacement = _SLANG_MAP[word]
            if replacement:
                expanded.append(replacement)
        else:
            expanded.append(word)
    result = " ".join(expanded)

    # Correct typos (word-by-word)
    words = result.split()
    corrected = []
    for word in words:
        if word in _TYPO_CORRECTIONS:
            corrected.append(_TYPO_CORRECTIONS[word])
        else:
            corrected.append(word)
    result = " ".join(corrected)

    # Normalize whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


# ---------------------------------------------------------------------------
# Entity Extraction from Query
# ---------------------------------------------------------------------------

_RELATIONSHIP_WORDS = {
    "mom", "mother", "dad", "father", "brother", "sister",
    "friend", "teacher", "professor", "doctor", "boss", "manager",
    "colleague", "classmate", "roommate", "neighbor",
    "cousin", "uncle", "aunt", "grandma", "grandfather",
    "grandpa", "grandmother", "wife", "husband",
    "girlfriend", "boyfriend", "partner", "sister", "brother",
}


def _extract_entity(normalized: str) -> tuple[str, str]:
    """Extract entity name and type from normalized query.

    Returns (entity_name, entity_type).
    """
    lower = normalized.lower()

    # Handle "who" queries - these are reverse entity lookups
    # "who likes pink" -> subject should be unknown/none for reverse lookup
    # EXCEPT self-identity forms ("who am I", "who are you"), which ask about
    # the user themselves.
    if not re.search(r"\bwho\s+am\s+i\b|\bwho\s+are\s+you\b", lower):
        if re.search(r"\bwho\s+\w", lower):
            return ("unknown", "unknown")

    # Third-party possessive: "mom's favorite color"
    # Skip contractions like "what's" = "what is"
    match = re.search(r"\b(\w+)'s\b", lower)
    if match:
        name = match.group(1)
        # Skip contractions and question words
        _CONTRACTION_WORDS = {"what", "where", "when", "who", "how", "that", "there", "here", "it", "he", "she", "we", "they", "let"}
        if name in _CONTRACTION_WORDS:
            pass  # skip — it's a contraction, not a possessive
        elif name not in {"my", "i", "me", "your", "you", "our", "we", "who", "what", "where", "when", "why", "how", "it", "that", "this"}:
            return (name, "person")

    # "my <person>" pattern
    # Non-person words that should not be treated as entity names
    _NON_PERSON_WORDS = {
        "own", "ownself", "self", "favorite", "favourite", "fav", "favrite",
        "best", "goal", "goals", "name", "nickname", "skill", "skills",
        "learning", "education", "preference", "interest", "interests",
        "project", "task", "priority", "ambition", "dream", "habit", "habits",
        "phone", "laptop", "device", "color", "colour", "food", "music",
        "movie", "language", "languages", "course", "courses", "thing",
        "things", "stuff", "item", "class", "computer", "car", "bike",
        "watch", "camera", "tablet", "earbuds", "headphones", "technology",
        "tool", "framework", "library", "program", "hobby", "hobbies",
        "tool", "platform", "subject", "field", "credential", "institution",
        "reading", "entertainment",
    }
    match = re.search(r"\bmy\s+(\w+)(?:'s)?\b", lower)
    if match:
        name = match.group(1)
        if name in _NON_PERSON_WORDS:
            pass  # skip — not a person entity
        elif name in _RELATIONSHIP_WORDS:
            return (name, "person")
        # "what does my mom like" — but only for relationship words
        elif name in _RELATIONSHIP_WORDS:
            return (name, "person")

    # Bare entity names (e.g., "mom", "dad", "sister") in questions like "does mom like"
    # Pattern: "does mom like", "what color mom like", "what mom like"
    match = re.search(r"\b(?:does|what)\s+(mom|mother|dad|father|brother|sister|friend|teacher|professor|doctor|boss|manager|colleague|classmate|roommate|neighbor|cousin|uncle|aunt|grandma|grandfather|grandpa|grandmother|wife|husband|girlfriend|boyfriend|partner)\b", lower)
    if match:
        return (match.group(1), "person")

    # Self-references
    if re.search(r"\b(?:i|me|my|mine|myself)\b", lower):
        return ("user", "self")

    # "the user" references
    if re.search(r"\bthe\s+user\b", lower):
        return ("user", "self")

    # Default: self if looks personal
    if re.search(r"\b(?:favou?rite|what\s+(?:do|does)\s+(?:i|the\s+user))\b", lower):
        return ("user", "self")

    return ("user", "self")


# ---------------------------------------------------------------------------
# Relation Extraction
# ---------------------------------------------------------------------------

def _extract_relation(normalized: str, object_hint: str = "unknown") -> str:
    """Extract the relation type from normalized query.

    Returns a canonical relation string.
    """
    lower = normalized.lower()

    # Negation check — don't extract relation from negated queries
    is_negated = bool(re.search(
        r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|did\s+not|didn'?t|never|no\s+longer|"
        r"do\s+(?:i|he|she|it|we|they|you)\s+not|does\s+(?:i|he|she|it|we|they|you)\s+not|"
        r"did\s+(?:i|he|she|it|we|they|you)\s+not)\b",
        lower,
    ))

    # Centralized definition-intent guard (semantic, generic)
    # Centralized definition-intent guard (semantic, generic)
    is_definitional = bool(re.search(r"what\s+is\s+\w+", lower) or re.search(r"explain\s+\w+", lower) or re.search(r"tell\s+me\s+about\s+\w+", lower))
    has_personal_ref = bool(re.search(r"(?:my|i\s+am|mine|about\s+me)", lower))
    if is_definitional and not has_personal_ref:
        return "unknown"

        # Centralized project/preference relations (before learning)
    if re.search(r"developing|building|creating", lower) and not is_negated:
        return "project"

        # Check for preference words FIRST (before possess) to avoid "favorite" → possess
    if re.search(r"\b(?:like|likes|enjoy|enjoys|love|loves|fancy|fancies|prefer|prefers|favorite|favourite|fav)\b", lower):
        if re.search(r"\b(?:dislike|dislikes|hate|hates|don'?t\s+like|do\s+not\s+like|don'?t\s+enjoy)\b", lower):
            return "disprefer"
        if is_negated:
            return "disprefer"
        return "prefer"

    # "learning interests"/"study hobbies"/"learning subjects" → learning domain,
    # not a plain preference (evaluated before the open-ended interest regex).
    if re.search(
        r"\b(?:learning|studying)\b.*\b(?:interests?|hobbies?|subjects?|topics?)\b",
        lower,
    ):
        if is_negated:
            return "not_learn"
        return "learn"

    # Open-ended interest/preference queries
    # "what am I interested in", "what am I into these days"
    if re.search(r"\b(?:interested\s+in|into\s+these\s+days|interests?)\b", lower):
        if is_negated:
            return "disprefer"
        return "prefer"

    # Project / working-on — centralized (before learning when project context present)
    if re.search(r"(?:developing|building|creating|working\s+on)", lower) and not is_negated:
        return "project"

    # Project / working-on — centralized (before general learning)
    if re.search(r"(?:developing|building|creating|working\s+on)", lower) and not is_negated:
        return "project"

    # Learning relations (check before possess for "working on")
    if re.search(r"\b(?:learn|learning|learns|study|studying|studies|know|knows|taking|taken|take|working\s+on)\b", lower):
        if is_negated:
            return "not_learn"
        return "learn"

    # Goal relations - "what do I want to build", "what am I working on" (when not tech)
    if re.search(r"\b(?:goal|goals|aim|aims|want\s+to\s+(?:become|build|make|create|do)|plan\s+to|aspire|ambition|ambitions|dream|dreams)\b", lower):
        return "goal"

    # "what am I working on" - if not tech context, could be goal
    # But "working on" is already handled in learning above for tech

    # Possession relations (after preference and learn checks)
    # Skip "use" when referring to technology/tools (should be "learn")
    _TECH_WORDS = {"technology", "technologies", "language", "languages", "framework", "frameworks", "library", "libraries", "tool", "tools", "platform", "platforms"}
    has_tech_object = any(word in lower for word in _TECH_WORDS)
    if re.search(r"\b(?:have|own|got|possess|carry|mine\b|need)\b", lower):
        if is_negated:
            return "not_possess"
        return "possess"
    if re.search(r"\b(?:use|uses|using)\b", lower):
        # Knowledge stacks ("frameworks", "libraries", "tools", "platforms") are
        # things one learns/uses as skills → "learn"; hardware (phone, laptop,
        # device, car) stays "possess" (handled before this branch).
        if has_tech_object and object_hint == "tool":
            return "learn"
        # Explicit learning language combined with tech → "learn"
        if has_tech_object and re.search(r"\b(?:learn|learning|study|studying)\b", lower):
            return "learn"
        if is_negated:
            return "not_possess"
        return "possess"

    # Identity relations
    if re.search(r"\b(?:name|called|nickname|alias|who\s+am\s+i|who\s+is)\b", lower):
        return "identity"

    # Goal relations
    if re.search(r"\b(?:goal|goals|aim|aims|want\s+to\s+become|plan\s+to|aspire|ambition|ambitions|dream|dreams)\b", lower):
        return "goal"

    # Skill relations
    if re.search(r"\b(?:skill|skills|ability|abilities|proficient|expert|good\s+at|know\s+how\s+to)\b", lower):
        return "skill"

    # General "what" queries — infer from object
    if re.search(r"\bwhat\b", lower):
        # Check for negation/dislike first
        if re.search(r"\b(?:dislike|dislikes|hate|hates|don'?t\s+like|do\s+not\s+like|don'?t\s+enjoy)\b", lower):
            return "disprefer"
        if object_hint in ("phone", "laptop", "computer", "tablet", "device", "car", "bike", "watch", "camera", "audio", "peripheral"):
            return "possess"
        if object_hint in ("technology", "language", "tool"):
            # Only map to learn if the question is about learning, not using
            if re.search(r"\b(?:learn|learning|study|studying)\b", lower):
                return "learn"
            return "possess"
        if object_hint in ("color", "food", "entertainment", "music", "reading", "hobby"):
            return "prefer"

    return "unknown"


# ---------------------------------------------------------------------------
# Object Extraction
# ---------------------------------------------------------------------------

def _extract_object(normalized: str) -> str:
    """Extract the object being asked about from normalized query.

    Returns a canonical object string.
    """
    lower = normalized.lower()

    for pattern, obj in _OBJECT_PATTERNS:
        if re.search(pattern, lower):
            return obj

    # Try to extract a noun after question words
    match = re.search(r"\bwhat\s+(?:is|are|do|does|did|was|were|am|have|has|had)\s+(?:my|i(?:'m|\s+am|\s+was|\s+were|\s+did|\s+do|\s+have|\s+had|\s+am)?|the\s+user(?:'s)?)?\s*(\w+)", lower)
    if match:
        obj = match.group(1)
        # Skip pronouns and common verbs as objects
        _SKIP_OBJECTS = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs", "using", "have", "has", "had", "do", "does", "did", "was", "were", "is", "are", "am", "being", "been", "get", "got", "getting", "go", "going", "went", "make", "made", "making", "take", "took", "taking", "give", "gave", "giving", "say", "said", "saying", "know", "knew", "knowing", "think", "thought", "thinking", "want", "wanted", "wanting", "like", "liked", "liking", "use", "used", "need", "needed", "trying", "try", "interested", "into", "working", "stop", "stopped", "stopping"}
        if obj not in _SKIP_OBJECTS:
            return obj

    # Special handling for "what did I stop learning" pattern
    # Extract the object after "stop" or "stopped"
    match = re.search(r"\bstop(?:ped|ping)?\s+(?:learning\s+)?(\w+)", lower)
    if match:
        obj = match.group(1)
        _STOP_SKIP = {"learning", "using", "doing", "liking"}
        if obj not in _STOP_SKIP:
            return obj
        # If it's "learning", try to get the next word
        if obj == "learning":
            match2 = re.search(r"\bstop(?:ped|ping)?\s+learning\s+(\w+)", lower)
            if match2:
                return match2.group(1)

    # "which X do I have"
    match = re.search(r"\bwhich\s+(\w+)", lower)
    if match:
        return match.group(1)

    return "unknown"


# ---------------------------------------------------------------------------
# Temporal / State Extraction
# ---------------------------------------------------------------------------

def _extract_temporal(normalized: str) -> tuple[str, Optional[str]]:
    """Extract temporal state and time qualifier.

    Returns (state, time) where state is "current", "historical", "future", etc.
    """
    lower = normalized.lower()

    # Historical signals
    if re.search(r"\b(?:previously|previous|before|in\s+the\s+past|once|old|former|formerly|past|earlier|used\s+to|use\s+to|was\s+using|were\s+using|did\s+(?:have|use|own)|stopped|stop|ceased)\b", lower):
        return ("historical", None)

    # Future signals
    if re.search(r"\b(?:in\s+the\s+future|next\s+year|someday|eventually|will\s+(?:have|use|own))\b", lower):
        return ("future", None)

    # Current signals
    if re.search(r"\b(?:currently|right\s+now|these\s+days|at\s+present|now|am\s+using|using\s+now)\b", lower):
        return ("current", "now")

    # Specific time references
    time_match = re.search(r"\b(?:in|during|on)\s+(\d{4})\b", lower)
    if time_match:
        return ("historical", time_match.group(1))

    # "before that" / "after that" — relative temporal
    if re.search(r"\bbefore\s+that\b", lower):
        return ("historical", None)
    if re.search(r"\bafter\s+that\b", lower):
        return ("current", None)

    # Default: current (no explicit temporal scope)
    return ("current", None)


# ---------------------------------------------------------------------------
# Cardinality Extraction
# ---------------------------------------------------------------------------

def _extract_cardinality(normalized: str) -> str:
    """Extract expected answer cardinality.

    Returns "one", "many", "existence", or "count".
    """
    lower = normalized.lower()

    # Count queries
    if re.search(r"\bhow\s+many\b", lower):
        return "count"

    # Plural queries (check before existence)
    if re.search(r"\bwhat\s+are\b", lower):
        return "many"
    if re.search(r"\b(?:all|every|each|list|tell\s+me\s+(?:all|every))\b", lower):
        return "many"
    if re.search(r"\bwhat\s+(?:things?|stuff|items?)\b", lower):
        return "many"
    if re.search(r"\b(?:languages?|skills?|goals?|projects?|habits?|tasks?|priorities?|devices?|phones?|cars?|bikes?|watches?)\b", lower):
        if re.search(r"\bwhat\s+(?:are|were)\s+(?:all\s+)?(?:the\s+)?", lower):
            return "many"
        if re.search(r"\bwhich\s+(?:are|were)\b", lower):
            return "many"

    # Existence / boolean queries — only when no specific object is asked
    # "Do I have a laptop?" → existence (no "what" prefix)
    # "What phone do I have?" → one (has "what" prefix)
    if re.search(r"\b(?:do|does|did)\s+(?:i|my|the\s+user)\b", lower):
        # Skip if there's a "what" prefix — that's asking for a specific thing
        if not re.search(r"\bwhat\b", lower):
            # "Do I still have X?" → existence
            if re.search(r"\b(?:still|actually|really|truly)\b", lower):
                return "existence"
            # "Do I like X?" → existence
            if re.search(r"\b(?:like|enjoy|love|hate|dislike)\b", lower):
                return "existence"
            # "Do I have X?" / "Does my mom have X?" → existence (boolean question)
            if re.search(r"\b(?:have|own|possess|use|know|learn|study)\b", lower):
                return "existence"

    # "what X do I have" — singular (asking for a specific thing)
    if re.search(r"\bwhat\s+\w+\s+do\s+i\s+have\b", lower):
        return "one"

    # Default: singular
    return "one"


# ---------------------------------------------------------------------------
# Domain Extraction
# ---------------------------------------------------------------------------

def _extract_domain(normalized: str, relation: str, object_hint: str) -> str:
    """Extract the knowledge domain from the query."""
    lower = normalized.lower()

    if relation in ("possess", "not_possess"):
        return "possessions"
    if relation in ("prefer", "disprefer"):
        return "interests"
    if relation == "learn":
        return "learning"
    if relation == "identity":
        return "identity"
    if relation == "goal":
        return "goals"
    if relation == "skill":
        return "skills"

    # Infer from object
    if object_hint in ("phone", "laptop", "computer", "tablet", "device", "car", "bike", "watch", "camera", "audio", "peripheral"):
        return "possessions"
    if object_hint in ("technology", "language", "tool"):
        return "learning"
    if object_hint in ("color", "food", "entertainment", "music", "reading", "hobby"):
        return "interests"

    # Infer from keywords
    if re.search(r"\b(?:learning|studying|course|class|subject)\b", lower):
        return "learning"
    if re.search(r"\b(?:like|dislike|favorite|favourite|prefer|hate|enjoy)\b", lower):
        return "interests"
    if re.search(r"\b(?:have|own|use|using|possession|device|phone|laptop)\b", lower):
        return "possessions"
    if re.search(r"\b(?:name|nickname|called|who\s+am\s+i)\b", lower):
        return "identity"
    if re.search(r"\b(?:goal|aims?|ambition|dream|want\s+to\s+become)\b", lower):
        return "goals"
    if re.search(r"\b(?:skill|abilities|proficient|good\s+at)\b", lower):
        return "skills"

    return "unknown"


# ---------------------------------------------------------------------------
# Negation / Uncertainty in Queries
# ---------------------------------------------------------------------------

def _detect_query_negation(normalized: str) -> bool:
    """Detect if a query contains negation."""
    return bool(
        re.search(
            r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|didn'?t|did\s+not|"
            r"can'?t|cannot|never|no\s+longer|not)\b",
            normalized,
        )
    )


def _detect_query_uncertainty(normalized: str) -> bool:
    """Detect if a query expresses uncertainty."""
    return bool(
        re.search(
            r"\b(?:i\s+think|i\s+believe|i\s+feel\s+like|maybe|perhaps|possibly|"
            r"might|could|not\s+sure|unsure|apparently|apparently|"
            r"i\s+guess|i\s+suppose|isn't\s+it|right\?)\b",
            normalized,
        )
    )


# ---------------------------------------------------------------------------
# Constraint Extraction
# ---------------------------------------------------------------------------

def _extract_constraints(
    normalized: str,
    state: str,
    time: Optional[str],
    entity: str,
    domain: str,
) -> list[str]:
    """Extract additional constraints from the query."""
    constraints = []
    lower = normalized.lower()

    if state == "historical":
        constraints.append("historical")
    elif state == "current":
        constraints.append("current")
    elif state == "future":
        constraints.append("future")

    if time:
        constraints.append(f"time:{time}")

    if entity and entity != "user":
        constraints.append(f"entity:{entity}")

    if domain and domain != "unknown":
        constraints.append(f"domain:{domain}")

    # "still" implies continuity
    if re.search(r"\bstill\b", lower):
        constraints.append("still")

    # "currently" implies present state
    if re.search(r"\bcurrently\b", lower):
        constraints.append("currently")

    # "before that" / "after that" implies relative temporal
    if re.search(r"\bbefore\s+that\b", lower):
        constraints.append("before_previous")
    if re.search(r"\bafter\s+that\b", lower):
        constraints.append("after_previous")

    return constraints


# ---------------------------------------------------------------------------
# Main Query Understanding Function
# ---------------------------------------------------------------------------

def understand_query(query: str) -> StructuredQuery:
    """Convert a natural-language question into a structured query representation.

    This is the core Session 5 function. It handles:
      - Paraphrases → same structured meaning
      - Grammar variations → normalized form
      - Slang/typos → corrected form
      - Entity extraction → subject
      - Relation extraction → relation
      - Object extraction → object
      - Temporal extraction → state/time
      - Cardinality extraction → one/many/existence/count
      - Negation/uncertainty detection
      - Personal vs general distinction

    Example:
        >>> sq = understand_query("What phone do I have?")
        >>> sq.subject, sq.relation, sq.object
        ('user', 'possess', 'phone')
    """
    raw = query.strip()
    normalized = normalize_query(raw)

    if not normalized:
        return StructuredQuery(
            raw_query=raw,
            normalized_query=normalized,
            is_personal=False,
            is_general=True,
        )

    # Extract components
    entity_name, entity_type = _extract_entity(normalized)
    object_hint = _extract_object(normalized)
    relation = _extract_relation(normalized, object_hint)
    state, time = _extract_temporal(normalized)
    cardinality = _extract_cardinality(normalized)
    domain = _extract_domain(normalized, relation, object_hint)
    negated = _detect_query_negation(normalized)
    uncertain = _detect_query_uncertainty(normalized)

    # Determine personal vs general
    is_personal = bool(
        re.search(
            r"\b(?:my\b|i\s|me\s|mine\b|myself\b|about\s+me\b|the\s+user\b|who\s+am\s+i\b)",
            normalized,
        )
    )
    is_general = not is_personal and bool(
        re.search(
            r"\bwhat\s+is\s+(?:a|an|the)\s+\w+|"
            r"\bhow\s+(?:do|does|did|can|could|would|should)\s+(?:you|we|one|people)\b|"
            r"\bwho\s+(?:invented|discovered|created|wrote)\b|"
            r"\bexplain\s+(?:what|how|why)\b",
            normalized,
        )
    )

    # Extract constraints
    constraints = _extract_constraints(normalized, state, time, entity_name, domain)

    return StructuredQuery(
        subject=entity_name,
        relation=relation,
        object=object_hint,
        state=state,
        time=time,
        cardinality=cardinality,
        entity=entity_name,
        domain=domain,
        constraints=constraints,
        negated=negated,
        uncertain=uncertain,
        is_personal=is_personal,
        is_general=is_general,
        raw_query=raw,
        normalized_query=normalized,
    )
