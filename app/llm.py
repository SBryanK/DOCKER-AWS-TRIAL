from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

from .config import Settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful policy analyst. Answer the question using ONLY the numbered "
    "context passages. Cite every claim with the passage number in square brackets, "
    "e.g. [1]. If the context does not contain the answer, reply exactly: "
    '"The ingested documents do not cover this." '
    "Keep the answer under 120 words. Never use outside knowledge."
)


class GenerationError(RuntimeError):
    """Text-generation backend failed; the message contains an actionable hint."""


@dataclass(slots=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    mode: str


class ChatModel:
    name: str
    model_id: str

    def generate(self, question: str, context_blocks: list[str]) -> Generation:
        raise NotImplementedError


class BedrockChat(ChatModel):
    name = "bedrock"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.llm_model_id
        session_kwargs: dict[str, object] = {"region_name": settings.aws_region}
        if settings.aws_profile:
            session_kwargs["profile_name"] = settings.aws_profile
        session = boto3.Session(**session_kwargs)
        self._client = session.client(
            "bedrock-runtime",
            config=Config(
                connect_timeout=10,
                read_timeout=settings.bedrock_timeout_s,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def generate(self, question: str, context_blocks: list[str]) -> Generation:
        context = "\n\n".join(context_blocks)
        prompt = (
            f"Question: {question}\n\n"
            f"Context passages:\n{context}\n\n"
            "Answer using only the passages above and cite them as [n]."
        )
        try:
            response = self._client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": self.settings.bedrock_temperature,
                    "maxTokens": self.settings.bedrock_max_tokens,
                },
            )
        except (
            ClientError,
            NoCredentialsError,
            PartialCredentialsError,
            EndpointConnectionError,
        ) as exc:
            raise GenerationError(self._hint(exc)) from exc

        text = "".join(
            block.get("text", "") for block in response["output"]["message"]["content"]
        ).strip()
        if not text:
            raise GenerationError("Bedrock returned an empty response.")

        usage = response.get("usage", {})
        return Generation(
            text=text,
            input_tokens=int(usage.get("inputTokens", 0) or 0),
            output_tokens=int(usage.get("outputTokens", 0) or 0),
            model_id=self.model_id,
            mode="bedrock",
        )

    def _hint(self, exc: Exception) -> str:
        code = ""
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
        base = f"Bedrock generation call failed ({code or type(exc).__name__}). "
        if code in {"AccessDeniedException", "UnauthorizedException"}:
            hint = (
                "Check IAM (bedrock:InvokeModel), model access for this account, and "
                "that the model/inference profile is available in the region."
            )
        elif code in {"ResourceNotFoundException", "ValidationException"}:
            hint = (
                f"Model '{self.model_id}' is not usable in '{self.settings.aws_region}'. "
                "Run `python -m scripts.check_bedrock` to list available models and "
                "inference profiles."
            )
        elif code == "ThrottlingException":
            hint = "Bedrock throttled the request; retry or raise the quota."
        elif isinstance(exc, (NoCredentialsError, PartialCredentialsError)):
            hint = (
                "No AWS credentials found. Configure a local profile and pass AWS_PROFILE; "
                "mount ~/.aws read-only at runtime."
            )
        elif isinstance(exc, EndpointConnectionError):
            hint = f"Cannot reach the Bedrock endpoint in {self.settings.aws_region}."
        else:
            hint = "See the Bedrock troubleshooting guide; run scripts/check_bedrock.py."
        return base + hint


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "within",
}
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _terms(text: str) -> set[str]:
    return {
        word for word in _WORD.findall(text.lower()) if len(word) > 2 and word not in _STOPWORDS
    }


class LocalChat(ChatModel):
    """Offline extractive answerer: copies the most query-relevant sentences.

    It cannot paraphrase, but it is deterministic, free, and always cites the
    passage it copies — useful for CI and for the Docker demo without AWS.
    Sentence scoring uses IDF over the retrieved context so rare query terms
    outrank generic words such as "framework" or "risk".
    """

    name = "local"
    model_id = "local-extractive/1"

    def generate(self, question: str, context_blocks: list[str]) -> Generation:
        question_terms = _terms(question)
        input_tokens = sum(len(block) // 4 for block in context_blocks)
        if not question_terms:
            return self._not_covered(input_tokens)

        # Flatten to (block, sentence index, sentence) with a minimum length.
        candidates: list[tuple[int, int, str]] = []
        for block_index, block in enumerate(context_blocks, start=1):
            sentences = _SENTENCE.split(block)
            for sentence_index, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 25:
                    continue
                candidates.append((block_index, sentence_index, sentence))

        if not candidates:
            return self._not_covered(input_tokens)

        # IDF over candidate sentences: rare query terms carry more weight.
        document_frequency: dict[str, int] = {}
        for _, _, sentence in candidates:
            for term in _terms(sentence):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        total = len(candidates)

        def idf(term: str) -> float:
            return math.log((total + 1) / (document_frequency.get(term, 0) + 1)) + 1.0

        query_weight = sum(idf(term) for term in question_terms) or 1.0

        scored: list[tuple[float, int, int, str]] = []
        for block_index, sentence_index, sentence in candidates:
            sentence_terms = _terms(sentence)
            overlap = question_terms & sentence_terms
            if not overlap:
                continue
            score = sum(idf(term) for term in overlap) / query_weight
            specificity = len(overlap) / max(len(sentence_terms), 1)
            scored.append((score + 0.1 * specificity, block_index, sentence_index, sentence))

        if not scored:
            return self._not_covered(input_tokens)

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen: list[tuple[int, str]] = []
        seen_blocks: set[int] = set()
        for _, block_index, sentence_index, sentence in scored:
            if block_index in seen_blocks:
                continue
            seen_blocks.add(block_index)
            # Include the next sentence too: lists and definitions often continue
            # on the following line even when it shares fewer query terms.
            sentences = [s.strip() for s in _SENTENCE.split(context_blocks[block_index - 1])]
            picked = [sentence]
            if sentence_index + 1 < len(sentences):
                neighbour = sentences[sentence_index + 1].strip()
                if 25 <= len(neighbour) <= 400:
                    picked.append(neighbour)
            for candidate_sentence in picked:
                if len(candidate_sentence) > 400:
                    candidate_sentence = candidate_sentence[:400].rsplit(" ", 1)[0] + " […]"
                chosen.append((block_index, candidate_sentence))
            if len(seen_blocks) == 3:
                break

        chosen.sort(key=lambda item: item[0])
        body = " ".join(
            f"{sentence.rstrip('.')} [{citation}]." for citation, sentence in chosen[:5]
        )
        answer = f"Based on the ingested documents: {body}"
        return Generation(
            text=answer,
            input_tokens=input_tokens,
            output_tokens=len(answer) // 4,
            model_id=self.model_id,
            mode="local-extractive",
        )

    @staticmethod
    def _not_covered(input_tokens: int) -> Generation:
        return Generation(
            text="The ingested documents do not cover this.",
            input_tokens=input_tokens,
            output_tokens=8,
            model_id="local-extractive/1",
            mode="local-extractive",
        )


def build_chat(settings: Settings) -> BedrockChat | LocalChat:
    if settings.llm_provider == "bedrock":
        log.info("Generation: Bedrock model=%s", settings.llm_model_id)
        return BedrockChat(settings)
    log.info("Generation: local extractive (offline mode)")
    return LocalChat()
