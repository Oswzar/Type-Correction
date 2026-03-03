import re
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
from wordfreq import zipf_frequency


class DistilBertTypoCorrector:
    """A lightweight typo corrector powered by DistilBERT fill-mask inference."""

    def __init__(self, logger, model_name: str = "distilbert-base-uncased", max_candidates: int = 50):
        # Store logger so every step is visible in terminal output.
        self.logger = logger
        # Keep model name configurable for future upgrades.
        self.model_name = model_name
        # Limit top-k candidates to reduce memory and CPU load on 8GB RAM machines.
        self.max_candidates = max_candidates
        # Basic typo fallback table guarantees correction for very common fast-typing mistakes.
        self.common_typos = {
            "helo": "hello",
            "worl": "world",
            "teh": "the",
            "recieve": "receive",
            "adress": "address",
            "viewpoitt": "viewpoint",
        }
        # Only skip extremely frequent words; lower-frequency words still get contextual check.
        self.frequency_skip_threshold = 6.0

        # Flag indicates whether DistilBERT pipeline is available.
        self.model_available = False
        self.load_attempted = False
        self.tokenizer = None
        self.model = None
        self.fill_mask = None
        self.logger.info("Model will be lazily loaded on first contextual correction request")

    def correct_text(self, text: str) -> str:
        """Correct typos in input text and return corrected text."""
        # If input is empty or whitespace only, return as-is to avoid unnecessary work.
        if not text or not text.strip():
            self.logger.warning("No text provided for correction")
            return text

        self.logger.info("Original text: %r", text)

        # Split into words/punctuation/spaces while preserving exact separators.
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text) # The data type of tokens is 

        # Process each token one by one (batch size=1) to keep memory usage low.
        for idx, token in enumerate(tokens):
            if not token.isalpha():
                continue

            corrected = self._correct_single_word(token, tokens, idx)
            tokens[idx] = corrected

        corrected_text = "".join(tokens)
        self.logger.info("Corrected text: %r", corrected_text)
        return corrected_text

    def _correct_single_word(self, word: str, all_tokens: List[str], index: int) -> str:
        """Correct a single word by typo table + DistilBERT contextual prediction."""
        original_word = word
        lower_word = word.lower()
        original_frequency = zipf_frequency(lower_word, "en")

        # Skip very short tokens to avoid unstable substitutions like single-letter words.
        if len(lower_word) <= 2:
            return original_word

        # Quick deterministic correction for common typos (including short ones like 'th').
        if lower_word in self.common_typos:
            mapped = self._match_case(self.common_typos[lower_word], original_word)
            self.logger.info("Typo map hit: %s -> %s", original_word, mapped)
            return mapped

        # Skip very short tokens to avoid unstable substitutions like single-letter words.
        if len(lower_word) <= 2:
            return original_word

        # If word is extremely common, keep it to reduce over-correction risk.
        if original_frequency >= self.frequency_skip_threshold:
            return original_word

        if not self.model_available:
            self._ensure_model_loaded()

        if not self.model_available or not self.tokenizer or not self.fill_mask:
            if original_frequency < 2.0:
                fallback_candidate = self._fallback_edit_candidate(lower_word)
                if fallback_candidate:
                    corrected = self._match_case(fallback_candidate, original_word)
                    self.logger.info("Fallback correction: %s -> %s", original_word, corrected)
                    return corrected
            return original_word

        left_context = self._collect_left_context(all_tokens, index)
        right_context = self._collect_right_context(all_tokens, index)
        mask_token = self.tokenizer.mask_token
        masked_text = f"{left_context} {mask_token} {right_context}".strip()

        self.logger.info("Masking word %r with context: %r", original_word, masked_text)

        try:
            # no_grad avoids gradient allocation to reduce memory pressure.
            with torch.no_grad():
                predictions = self.fill_mask(masked_text)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                self.logger.error("OOM detected while correcting %r; keeping original", original_word)
                return original_word
            self.logger.error("Runtime error during model inference: %s", error)
            return original_word
        except Exception as error:
            self.logger.error("Failed to infer correction for %r: %s", original_word, error)
            return original_word

        candidate = self._select_best_candidate(predictions, lower_word)
        if not candidate:
            if original_frequency < 2.0:
                fallback_candidate = self._fallback_edit_candidate(lower_word)
                if fallback_candidate:
                    corrected = self._match_case(fallback_candidate, original_word)
                    self.logger.info("Fallback-after-model correction: %s -> %s", original_word, corrected)
                    return corrected
            return original_word

        corrected = self._match_case(candidate, original_word)
        if corrected != original_word:
            self.logger.info("Model correction: %s -> %s", original_word, corrected)
        return corrected

    def _ensure_model_loaded(self) -> None:
        """Load model/tokenizer once; try local cache first, then online auto-download."""
        if self.load_attempted:
            return
        self.load_attempted = True

        try:
            # Prefer Apple Silicon GPU if available; otherwise stay on CPU for compatibility.
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                # float16 on MPS helps reduce memory usage on low-memory devices.
                self.dtype = torch.float16
                self.logger.info("Using MPS backend with float16 for memory optimization")
            else:
                self.device = torch.device("cpu")
                # CPU uses float32 for stability.
                self.dtype = torch.float32
                self.logger.info("Using CPU backend with float32")

            self.logger.info("Trying local cached tokenizer/model first: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                local_files_only=True,
            )
        except Exception:
            try:
                self.logger.info("Local cache miss; attempting online model auto-download")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
            except Exception as error:
                self.logger.error("Model download/load failed: %s", error)
                self.logger.warning("Falling back to lightweight typo-map mode (no model inference)")
                self.model_available = False
                return

        try:
            self.model.to(self.device)
            self.model.eval()
            pipeline_device = self.device if self.device.type == "mps" else -1
            # Use a fill-mask pipeline for simple and robust contextual correction.
            self.fill_mask = pipeline(
                "fill-mask",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device,
                top_k=self.max_candidates,
            )
            self.model_available = True
            self.logger.info("Model initialized successfully")
        except Exception as error:
            self.logger.error("Model post-load initialization failed: %s", error)
            self.logger.warning("Falling back to lightweight typo-map mode (no model inference)")
            self.model_available = False

    def _fallback_edit_candidate(self, original_lower: str) -> Optional[str]:
        """Find a probable correction using typo map and frequency-only heuristics."""
        if original_lower in self.common_typos:
            return self.common_typos[original_lower]

        candidates = self._generate_edit_distance_one(original_lower)
        best_word = None
        best_freq = 0.0

        for candidate in candidates:
            frequency = zipf_frequency(candidate, "en")
            if frequency > best_freq and frequency >= 3.0:
                best_freq = frequency
                best_word = candidate

        return best_word

    def _generate_edit_distance_one(self, word: str) -> List[str]:
        """Generate edit-distance-1 variants with bounded complexity for low-memory fallback."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [left + right[1:] for left, right in splits if right]
        transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
        replaces = [left + char + right[1:] for left, right in splits if right for char in letters]
        inserts = [left + char + right for left, right in splits for char in letters]
        return list(set(deletes + transposes + replaces + inserts))

    def _select_best_candidate(self, predictions, original_lower: str) -> str:
        """Select the most plausible candidate close to the original typo."""
        best_candidate = ""
        best_score = -1.0

        if isinstance(predictions, dict):
            predictions = [predictions]

        for item in predictions:
            token = item.get("token_str", "").strip().lower()
            model_score = float(item.get("score", 0.0))

            # Ignore empty or non-alphabetic candidates.
            if not token.isalpha():
                continue

            # Require edit distance <= 1 to focus on typo-like spelling corrections.
            distance = self._levenshtein_distance(original_lower, token)
            if distance > 2:
                continue

            # Blend model confidence with word frequency confidence.
            freq_score = max(zipf_frequency(token, "en"), 0.0) / 10.0
            final_score = 0.8 * model_score + 0.2 * freq_score

            if final_score > best_score:
                best_score = final_score
                best_candidate = token

        return best_candidate

    def _collect_left_context(self, tokens: List[str], index: int, max_words: int = 8) -> str:
        """Collect up to max_words of left context."""
        words = []
        for token in reversed(tokens[:index]):
            if token.isalpha():
                words.append(token)
                if len(words) >= max_words:
                    break
        return " ".join(reversed(words))

    def _collect_right_context(self, tokens: List[str], index: int, max_words: int = 8) -> str:
        """Collect up to max_words of right context."""
        words = []
        for token in tokens[index + 1 :]:
            if token.isalpha():
                words.append(token)
                if len(words) >= max_words:
                    break
        return " ".join(words)

    def _match_case(self, new_word: str, original_word: str) -> str:
        """Preserve input capitalization style when replacing words."""
        if original_word.isupper():
            return new_word.upper()
        if original_word.istitle():
            return new_word.title()
        return new_word

    def _levenshtein_distance(self, a: str, b: str) -> int:
        """Compute Levenshtein edit distance with a small dynamic programming table."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev_row = list(range(len(b) + 1))
        for i, char_a in enumerate(a, start=1):
            current_row = [i]
            for j, char_b in enumerate(b, start=1):
                insert_cost = current_row[j - 1] + 1
                delete_cost = prev_row[j] + 1
                replace_cost = prev_row[j - 1] + (char_a != char_b)
                current_row.append(min(insert_cost, delete_cost, replace_cost))
            prev_row = current_row
        return prev_row[-1]


def get_default_model_info() -> Tuple[str, str]:
    """Return default model name and public Hugging Face link."""
    model_name = "distilbert-base-uncased"
    model_link = "https://huggingface.co/distilbert-base-uncased"
    return model_name, model_link
