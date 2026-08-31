"""
adversarial/ — obfuscation normalization and adversarial example generation.

Modules
-------
normalizer   : canonicalize obfuscated text before moderation (defensive)
generator    : produce obfuscated variants of clean text for robustness testing (offensive)
multilingual : language-hint detection and transliteration tables

Typical usage
-------------
    from adversarial.normalizer import normalize
    from adversarial.generator import generate_variants
    from adversarial.multilingual import detect_language_hint, MULTILINGUAL_TOXIC_SEEDS

    clean = normalize("h@te y0u st00pid id10t")
    # → "hate you stupid idiot"

    variants = generate_variants("hate", techniques=["symbol_substitution", "spacing"])
    # → [{"technique": "symbol_substitution", "text": "h@te"}, ...]
"""
