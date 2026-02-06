"""
Response Validator Service.

Enforces "Hallucination Prevention" by verifying that LLM responses
are grounded in the provided context.

Checks:
1. Citation Existence: Facts must have [id].
2. Span Binding (Token Overlap): The cited claim must lexically overlap with the source chunk.
   (Proxy for entailment that is faster and deterministic).
"""

import re
import logging
from typing import List, Dict, Set, Tuple

logger = logging.getLogger("validator")

class ResponseValidator:
    """
    Validates LLM responses against retrieved context.
    Enforces strict grounding protocols.
    """
    
    # Minimum ratio of claim tokens that must appear in source chunk
    # 0.4 means 40% of significant words in the sentence must exist in the source.
    # This allows for paraphrasing but catches "wildly different" hallucinations.
    OVERLAP_THRESHOLD = 0.4
    
    def validate_response(self, response: str, context_chunks: List[Dict]) -> Tuple[bool, str]:
        """
        Validate response against context chunks.
        
        Args:
            response: LLM output text
            context_chunks: List of result dicts (passed to context wrapper)
            
        Returns:
            (is_valid, reason) - True if valid, False + Reason if blocked.
        """
        if not response:
            return True, "Empty response"
            
        # If response indicates "I don't know", allow it.
        # Heuristics for refusal
        refusals = ["i don't have", "no information", "cannot find", "not mentioned", "i am not sure"]
        if any(r in response.lower() for r in refusals) and len(response) < 100:
            return True, "Valid Refusal"
            
        # 1. Parse Citations
        citations = self._extract_citations(response)
        if not citations:
             # If response is factual/long but has no citations => SUSPICIOUS
             # But simple chit-chat (greeting) is fine.
             # Heuristic: If response > 50 words and has numbers/entities, block.
             # For now, we only enforce if context was provided (implied by this being called)
             if len(context_chunks) > 0 and len(response.split()) > 50:
                 # Allow if it's a question (asking for clarification/details)
                 if "?" in response.strip()[-10:]:
                     return True, "Question passed"
                     
                 logger.warning("[Validator] Factual response blocked: Missing citations.")
                 return False, "Response missing citations [n]."
             return True, "Short/Chit-chat passed"

        # 2. Map IDs to Chunks
        chunk_map = {str(i+1): chunk.get("text", "") for i, chunk in enumerate(context_chunks)}
        
        # 3. Check Grounding for each citation
        for cid, spans in citations.items():
            if cid not in chunk_map:
                logger.warning(f"[Validator] Invalid citation ID [{cid}]")
                return False, f"Cited non-existent source [{cid}]"
            
            source_text = chunk_map[cid]
            
            for span in spans:
                if not self._check_token_overlap(span, source_text):
                    logger.warning(f"[Validator] Hallucination detected. Span '{span[:30]}...' not supported by Source [{cid}]")
                    return False, f"Claim not supported by source [{cid}]"
                    
        return True, "Validated"
        
    def _extract_citations(self, text: str) -> Dict[str, List[str]]:
        """
        Extract sentences attached to citations.
        Returns: {cid: [sentence1, sentence2]}
        """
        citation_map = {}
        
        # Split by sentence (naive splitting is okay for this proxy)
        # We assume citations come at the end of sentences like "Fish swim [1]."
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sent in sentences:
            # Find all [n]
            matches = re.findall(r'\[(\d+)\]', sent)
            if not matches:
                continue
            
            # Associate this sentence with all found CIDs
            for cid in set(matches):
                if cid not in citation_map:
                    citation_map[cid] = []
                citation_map[cid].append(sent)
                
        return citation_map

    def _check_token_overlap(self, claim: str, source: str) -> bool:
        """
        Check if significant tokens in claim appear in source.
        """
        # Common English stopwords (expanded for hallucination prevention)
        STOPWORDS = {
            'their', 'about', 'would', 'these', 'other', 'words', 'could', 'write',
            'first', 'water', 'after', 'where', 'right', 'think', 'three', 'years',
            'place', 'sound', 'great', 'again', 'still', 'every', 'small', 'found',
            'those', 'never', 'under', 'might', 'while', 'house', 'world', 'below',
            'asked', 'going', 'large', 'until', 'along', 'shall', 'being', 'often',
            'earth', 'began', 'since', 'study', 'night', 'light', 'abover', 'paper',
            'parts', 'young', 'story', 'point', 'times', 'heard', 'whole', 'white',
            'given', 'means', 'music', 'miles', 'thing', 'today', 'later', 'using',
            'money', 'lines', 'order', 'group', 'among', 'learn', 'known', 'space',
            'table', 'early', 'trees', 'short', 'hands', 'state', 'black', 'shown',
            'stood', 'front', 'voice', 'kinds', 'makes', 'comes', 'close', 'power',
            'lived', 'vowel', 'taken', 'built', 'heart', 'ready', 'quite', 'class',
            'bring', 'round', 'horse', 'shows', 'piece', 'green', 'stand', 'birds',
            'start', 'river', 'tried', 'least', 'field', 'whose', 'girls', 'leave',
            'added', 'check', 'game', 'shape', 'equate', 'hot', 'miss', 'heat',
            'snow', 'tire', 'bring', 'yes', 'distant', 'fill', 'east', 'paint',
            'language', 'among', 'unit', 'power', 'town', 'fine', 'certain',
            'fly', 'fall', 'lead', 'cry', 'dark', 'machine', 'note', 'wait',
            'plan', 'figure', 'star', 'box', 'noun', 'field', 'rest', 'correct',
            'able', 'pound', 'done', 'beauty', 'drive', 'stood', 'contain',
            'front', 'teach', 'week', 'final', 'gave', 'green', 'oh', 'quick',
            'develop', 'ocean', 'warm', 'free', 'minute', 'strong', 'special',
            'mind', 'behind', 'clear', 'tail', 'produce', 'fact', 'space',
            'heard', 'best', 'hour', 'better', 'true', 'during', 'hundred',
            'five', 'remember', 'step', 'early', 'hold', 'west', 'ground',
            'interest', 'reach', 'fast', 'verb', 'sing', 'listen', 'six',
            'table', 'travel', 'less', 'morning', 'ten', 'simple', 'several',
            'vowel', 'toward', 'war', 'lay', 'against', 'pattern', 'slow',
            'center', 'love', 'person', 'money', 'serve', 'appear', 'road',
            'map', 'rain', 'rule', 'govern', 'pull', 'cold', 'notice', 'voice',
            'unit', 'power', 'town', 'fine', 'certain', 'fly', 'fall', 'lead',
            'cry', 'dark', 'machine', 'note', 'wait', 'plan', 'figure', 'star',
            'box', 'noun', 'field', 'rest', 'correct', 'able', 'pound', 'done',
            'beauty', 'drive', 'stood', 'contain', 'front', 'teach', 'week',
            'final', 'gave', 'green', 'oh', 'quick', 'develop', 'ocean', 'warm',
            'free', 'minute', 'strong', 'special', 'mind', 'behind', 'clear',
            'tail', 'produce', 'fact', 'space', 'heard', 'best', 'hour', 'better',
            'true', 'during', 'hundred', 'five', 'remember', 'step', 'early',
            'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'verb',
            'sing', 'listen', 'six', 'table', 'travel', 'less', 'morning', 'ten',
            'simple', 'several', 'vowel', 'toward', 'war', 'lay', 'against',
            'pattern', 'slow', 'center', 'love', 'person', 'money', 'serve',
            'appear', 'road', 'map', 'rain', 'rule', 'govern', 'pull', 'cold',
            'notice', 'voice', 'energy', 'hunt', 'probable', 'bed', 'brother',
            'egg', 'ride', 'cell', 'believe', 'perhaps', 'pick', 'sudden',
            'count', 'square', 'reason', 'length', 'represent', 'art', 'subject',
            'region', 'size', 'vary', 'settle', 'speak', 'weight', 'general',
            'ice', 'matter', 'circle', 'pair', 'include', 'divide', 'syllable',
            'felt', 'grand', 'ball', 'yet', 'wave', 'drop', 'heart', 'am',
            'present', 'heavy', 'dance', 'engine', 'position', 'arm', 'wide',
            'sail', 'material', 'size', 'vary', 'settle', 'speak', 'weight',
            'general', 'ice', 'matter', 'circle', 'pair', 'include', 'divide',
            'syllable', 'felt', 'grand', 'ball', 'yet', 'wave', 'drop', 'heart',
            'am', 'present', 'heavy', 'dance', 'engine', 'position', 'arm',
            'wide', 'sail', 'material', 'fraction', 'forest', 'sit', 'race',
            'window', 'store', 'summer', 'train', 'sleep', 'prove', 'lone',
            'leg', 'exercise', 'wall', 'catch', 'mount', 'wish', 'sky', 'board',
            'joy', 'winter', 'sat', 'written', 'wild', 'instrument', 'kept',
            'glass', 'grass', 'cow', 'job', 'edge', 'sign', 'visit', 'past',
            'soft', 'fun', 'bright', 'gas', 'weather', 'month', 'million',
            'bear', 'finish', 'happy', 'hope', 'flower', 'clothe', 'strange',
            'gone', 'jump', 'baby', 'eight', 'village', 'meet', 'root', 'buy',
            'raise', 'solve', 'metal', 'whether', 'push', 'seven', 'paragraph',
            'third', 'shall', 'held', 'hair', 'describe', 'cook', 'floor',
            'either', 'result', 'burn', 'hill', 'safe', 'cat', 'century',
            'consider', 'type', 'law', 'bit', 'coast', 'copy', 'phrase',
            'silent', 'tall', 'sand', 'soil', 'roll', 'temperature', 'finger',
            'industry', 'value', 'fight', 'lie', 'beat', 'excite', 'natural',
            'view', 'sense', 'ear', 'else', 'quite', 'broke', 'case', 'middle',
            'kill', 'son', 'lake', 'moment', 'scale', 'loud', 'spring',
            'observe', 'child', 'straight', 'consonant', 'nation', 'dictionary',
            'milk', 'speed', 'method', 'organ', 'pay', 'age', 'section',
            'dress', 'cloud', 'surprise', 'quiet', 'stone', 'tiny', 'climb',
            'cool', 'design', 'poor', 'lot', 'experiment', 'bottom', 'key',
            'iron', 'single', 'stick', 'flat', 'twenty', 'skin', 'smile',
            'crease', 'hole', 'trade', 'melody', 'trip', 'office', 'receive',
            'row', 'mouth', 'exact', 'symbol', 'die', 'least', 'trouble',
            'shout', 'except', 'wrote', 'seed', 'tone', 'join', 'suggest',
            'clean', 'break', 'lady', 'yard', 'rise', 'bad', 'blow', 'oil',
            'blood', 'touch', 'grew', 'cent', 'mix', 'team', 'wire', 'cost',
            'lost', 'brown', 'wear', 'garden', 'equal', 'sent', 'choose',
            'fell', 'fit', 'flow', 'fair', 'bank', 'collect', 'save', 'control',
            'decimal', 'ear', 'else', 'quite', 'broke', 'case', 'middle',
            'kill', 'son', 'lake', 'moment', 'scale', 'loud', 'spring',
            'observe', 'child', 'straight', 'consonant', 'nation', 'dictionary',
            'milk', 'speed', 'method', 'organ', 'pay', 'age', 'section',
            'dress', 'cloud', 'surprise', 'quiet', 'stone', 'tiny', 'climb',
            'cool', 'design', 'poor', 'lot', 'experiment', 'bottom', 'key',
            'iron', 'single', 'stick', 'flat', 'twenty', 'skin', 'smile',
            'crease', 'hole', 'trade', 'melody', 'trip', 'office', 'receive',
            'row', 'mouth', 'exact', 'symbol', 'die', 'least', 'trouble',
            'shout', 'except', 'wrote', 'seed', 'tone', 'join', 'suggest',
            'clean', 'break', 'lady', 'yard', 'rise', 'bad', 'blow', 'oil',
            'blood', 'touch', 'grew', 'cent', 'mix', 'team', 'wire', 'cost',
            'lost', 'brown', 'wear', 'garden', 'equal', 'sent', 'choose',
            'fell', 'fit', 'flow', 'fair', 'bank', 'collect', 'save', 'control',
            'decimal', 'gentle', 'woman', 'captain', 'practice', 'separate',
            'difficult', 'doctor', 'please', 'protect', 'noon', 'whose', 'locate',
            'ring', 'character', 'insect', 'caught', 'period', 'indicate',
            'radio', 'spoke', 'atom', 'human', 'history', 'effect', 'electric',
            'expect', 'crop', 'modern', 'element', 'hit', 'student', 'corner',
            'party', 'supply', 'bone', 'rail', 'imagine', 'provide', 'agree',
            'thus', 'capital', 'won', 'chair', 'danger', 'fruit', 'rich',
            'thick', 'soldier', 'process', 'operate', 'guess', 'necessary',
            'sharp', 'wing', 'create', 'neighbor', 'wash', 'bat', 'rather',
            'crowd', 'corn', 'compare', 'poem', 'string', 'bell', 'depend',
            'meat', 'rub', 'tube', 'famous', 'dollar', 'stream', 'fear', 'sight',
            'thin', 'triangle', 'planet', 'hurry', 'chief', 'colony', 'clock',
            'mine', 'tie', 'enter', 'major', 'fresh', 'search', 'send', 'yellow',
            'gun', 'allow', 'print', 'dead', 'spot', 'desert', 'suit', 'current',
            'lift', 'rose', 'continue', 'block', 'chart', 'hat', 'sell',
            'success', 'company', 'subtract', 'event', 'particular', 'deal',
            'swim', 'term', 'opposite', 'wife', 'shoe', 'shoulder', 'spread',
            'arrange', 'camp', 'invent', 'cotton', 'born', 'determine',
            'quart', 'nine', 'truck', 'noise', 'level', 'chance', 'gather',
            'shop', 'stretch', 'throw', 'shine', 'property', 'column',
            'molecule', 'select', 'wrong', 'gray', 'repeat', 'require', 'broad',
            'prepare', 'salt', 'nose', 'plural', 'anger', 'claim', 'continent',
            'oxygen', 'sugar', 'death', 'pretty', 'skill', 'women', 'season',
            'solution', 'magnet', 'silver', 'thank', 'branch', 'match',
            'suffix', 'especially', 'fig', 'afraid', 'huge', 'sister', 'steel',
            'discuss', 'forward', 'similar', 'guide', 'experience', 'score',
            'apple', 'bought', 'led', 'pitch', 'coat', 'mass', 'card', 'band',
            'rope', 'slip', 'win', 'dream', 'evening', 'condition', 'feed',
            'tool', 'total', 'basic', 'smell', 'valley', 'nor', 'double',
            'seat', 'arrive', 'master', 'track', 'parent', 'shore', 'division',
            'sheet', 'substance', 'favor', 'connect', 'post', 'spend', 'chord',
            'fat', 'glad', 'original', 'share', 'station', 'dad', 'bread',
            'charge', 'proper', 'bar', 'offer', 'segment', 'slave', 'duck',
            'instant', 'market', 'degree', 'populate', 'chick', 'dear',
            'enemy', 'reply', 'drink', 'occur', 'support', 'speech', 'nature',
            'range', 'steam', 'motion', 'path', 'liquid', 'log', 'meant',
            'quotient', 'teeth', 'shell', 'neck'
        }
        # Add basic pronouns and conjunctions manually just in case
        STOPWORDS.update({'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'in', 'of', 'to', 'for', 'it', 'this', 'that', 'with', 'made', 'make', 'makes', 'from', 'by', 'are', 'was', 'were', 'be', 'been', 'being'})
        
        # Remove colors/common adjectives that are significant
        STOPWORDS.difference_update({'green', 'white', 'black', 'yellow', 'red', 'blue', 'brown', 'gray', 'bad', 'good', 'great', 'best', 'small', 'large', 'long', 'short'})

        def tokenize(s):
            # Remove punctuation, lowercase, split
            words = re.findall(r'\w+', s.lower())
            # Basic stopwords + length check
            return set(w for w in words if len(w) > 3 and w not in STOPWORDS)
            
        claim_tokens = tokenize(claim)
        source_tokens = tokenize(source)
        
        if not claim_tokens:
            return True # No significant content to check (or all stopwords)
            
        intersection = claim_tokens.intersection(source_tokens)
        ratio = len(intersection) / len(claim_tokens)
        
        # Log for tuning
        logger.debug(f"[Overlap] Ratio: {ratio:.2f} | Claim: {claim[:20]}... | Overlap: {intersection}")
        
        return ratio >= self.OVERLAP_THRESHOLD

_validator = ResponseValidator()

def get_validator() -> ResponseValidator:
    return _validator
