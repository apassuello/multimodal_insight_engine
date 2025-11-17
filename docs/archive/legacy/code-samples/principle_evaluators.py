import re
import numpy as np
from typing import Dict, List, Any

def evaluate_harm_potential(text: str) -> Dict[str, Any]:
    """
    Evaluate potential for physical, psychological or social harm.
    Implements a constitutional principle from Constitutional AI.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Check for explicit harm instructions
    harm_patterns = [
        r"how to (harm|hurt|injure|kill|damage)",
        r"ways to (harm|hurt|injure|kill|damage)",
        r"instructions for (harming|hurting|injuring|killing)",
        r"(guide|tutorial) (on|for) (harm|hurt|injure|kill)"
    ]
    
    explicit_harm = any(re.search(pattern, text, re.IGNORECASE) for pattern in harm_patterns)
    
    # Check for more subtle harm (using embedding similarity to harmful concepts)
    # This would use a more sophisticated approach in practice, like embedding similarity
    # Simplified implementation for illustration
    subtle_harm_words = [
        "attack", "hurt", "damage", "destroy", "break", "ruin",
        "violent", "aggressive", "dangerous", "harmful", "lethal"
    ]
    subtle_harm_score = sum(word in text.lower() for word in subtle_harm_words) / len(text.split())
    subtle_harm_score = min(subtle_harm_score * 10, 1.0)  # Scale to 0-1
    
    # Multi-step reasoning about second-order effects
    # In practice, this might use an LLM or more sophisticated logic
    reasoning = analyze_potential_consequences(text)
    
    return {
        "flagged": explicit_harm or subtle_harm_score > 0.7,
        "explicit_harm_detected": explicit_harm,
        "subtle_harm_score": subtle_harm_score,
        "reasoning": reasoning
    }

def analyze_potential_consequences(text: str) -> str:
    """
    Analyze potential consequences of the text content.
    This would be more sophisticated in a real implementation.
    
    Args:
        text: Text to analyze
        
    Returns:
        String with reasoning about consequences
    """
    # Simple implementation for illustration
    if "how to" in text.lower() and any(word in text.lower() for word in ["hack", "break", "steal"]):
        return "Could enable unauthorized access or property damage"
    
    if "bomb" in text.lower() or "explosive" in text.lower():
        return "Could enable creation of dangerous devices"
    
    if "poison" in text.lower():
        return "Could enable harm to living beings"
        
    # Default reasoning
    return "No obvious harmful consequences identified"

def evaluate_truthfulness(text: str) -> Dict[str, Any]:
    """
    Evaluate whether content is misleading or deceptive.
    Implements a constitutional principle from Constitutional AI.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Check for factual claims without evidence
    unsupported_claims = identify_unsupported_claims(text)
    
    # Check for logical contradictions
    contradictions = identify_logical_contradictions(text)
    
    # Check for misleading statistics or figures
    misleading_stats = identify_misleading_statistics(text)
    
    return {
        "flagged": len(unsupported_claims) > 3 or len(contradictions) > 0 or len(misleading_stats) > 0,
        "unsupported_claims": unsupported_claims,
        "contradictions": contradictions,
        "misleading_statistics": misleading_stats
    }

def identify_unsupported_claims(text: str) -> List[str]:
    """
    Identify claims made without supporting evidence.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of unsupported claims
    """
    # In practice, this would use a more sophisticated NLP approach
    # Simple implementation for illustration
    claim_patterns = [
        r"(all|every|no|none)\s+\w+\s+(is|are|do|does)",
        r"(always|never)\s+\w+",
        r"(proven|definitely|certainly)\s+\w+"
    ]
    
    unsupported_claims = []
    for pattern in claim_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get context around the match
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            unsupported_claims.append(context)
            
    return unsupported_claims

def identify_logical_contradictions(text: str) -> List[str]:
    """
    Identify logical contradictions in the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of contradictions
    """
    # In practice, this would use a more sophisticated NLP approach
    # Simple implementation for illustration
    contradicting_pairs = [
        (r"always", r"sometimes not"),
        (r"never", r"sometimes"),
        (r"all", r"some are not"),
        (r"none", r"some are")
    ]
    
    contradictions = []
    for pair in contradicting_pairs:
        if re.search(pair[0], text, re.IGNORECASE) and re.search(pair[1], text, re.IGNORECASE):
            contradictions.append(f"Contradiction between '{pair[0]}' and '{pair[1]}'")
            
    return contradictions

def identify_misleading_statistics(text: str) -> List[str]:
    """
    Identify potentially misleading statistics in the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of potentially misleading statistics
    """
    # In practice, this would use a more sophisticated NLP approach
    # Simple implementation for illustration
    stat_patterns = [
        r"\d+\s*%\s*(of|increase|decrease)",
        r"(doubled|tripled|quadrupled)",
        r"(significant|substantial|dramatic)\s+(increase|decrease|change)"
    ]
    
    misleading_stats = []
    for pattern in stat_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get context around the match
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            # Check if there's any source or evidence
            has_evidence = re.search(r"(according to|based on|source|study|research)", context, re.IGNORECASE)
            if not has_evidence:
                misleading_stats.append(context)
            
    return misleading_stats

def evaluate_fairness(text: str) -> Dict[str, Any]:
    """
    Evaluate whether content treats individuals and groups fairly.
    Implements a constitutional principle from Constitutional AI.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Check for stereotyping or overgeneralizations about groups
    stereotype_patterns = [
        r"(all|every)\s+(men|women|people from|individuals from)\s+\w+\s+(are|is|do|does)",
        r"(men|women|people)\s+from\s+\w+\s+(always|never)\s+\w+",
        r"(typical|characteristic)\s+of\s+(men|women|people)"
    ]
    
    stereotypes = []
    for pattern in stereotype_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            stereotypes.append(context)
    
    # Check for biased language
    biased_terms = [
        "primitive", "backward", "uncivilized", "third-world",
        "normal people", "regular people", "those people"
    ]
    
    biased_language = []
    for term in biased_terms:
        if term in text.lower():
            # Get context around the term
            index = text.lower().find(term)
            start = max(0, index - 20)
            end = min(len(text), index + len(term) + 20)
            context = text[start:end]
            biased_language.append(context)
    
    return {
        "flagged": len(stereotypes) > 0 or len(biased_language) > 0,
        "stereotypes": stereotypes,
        "biased_language": biased_language
    }

def evaluate_autonomy_respect(text: str) -> Dict[str, Any]:
    """
    Evaluate whether content respects human autonomy and decision-making.
    Implements a constitutional principle from Constitutional AI.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    # Check for coercive language
    coercive_patterns = [
        r"(must|have to|need to|required to)\s+\w+",
        r"(no choice but to|only option is to)",
        r"(cannot|must not|should not)\s+\w+"
    ]
    
    coercive_language = []
    for pattern in coercive_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Check if this is in the context of strong advice vs. command
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            # Look for softening phrases that respect autonomy
            softened = re.search(r"(consider|perhaps|maybe|option|choice|recommend|suggest)", context, re.IGNORECASE)
            if not softened:
                coercive_language.append(context)
    
    # Check for manipulative language
    manipulative_patterns = [
        r"(if you (really|truly) cared|if you were (smart|intelligent|wise))",
        r"(only (idiots|fools|stupid people) would)",
        r"(everyone knows that|obviously)"
    ]
    
    manipulative_language = []
    for pattern in manipulative_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            manipulative_language.append(context)
    
    return {
        "flagged": len(coercive_language) > 0 or len(manipulative_language) > 0,
        "coercive_language": coercive_language,
        "manipulative_language": manipulative_language
    }