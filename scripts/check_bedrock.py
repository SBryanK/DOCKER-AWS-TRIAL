"""Verify Bedrock access, model IDs, inference profiles and budget state.

Run this BEFORE the first real Bedrock call:
    python -m scripts.check_bedrock --profile your-sso-profile

It never prints credentials. If a model ID is unavailable it shows what the
account can actually see in the configured region.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from app.config import Settings
from app.embeddings import BedrockEmbedder
from app.llm import BedrockChat

INTERESTING = ("titan", "cohere", "nova", "claude", "llama")


def _session(profile: str | None, region: str) -> boto3.Session:
    kwargs: dict[str, object] = {"region_name": region}
    if profile:
        kwargs["profile_name"] = profile
    return boto3.Session(**kwargs)


def _show_identity(session: boto3.Session) -> None:
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    print(f"AWS identity : {identity['Arn']} (account {identity['Account']})")


def _list_models(session: boto3.Session) -> tuple[list[dict], list[dict]]:
    bedrock = session.client("bedrock")
    models = bedrock.list_foundation_models().get("modelSummaries", [])
    profiles = bedrock.list_inference_profiles().get("inferenceProfileSummaries", [])
    return models, profiles


def _print_models(models: list[dict], profiles: list[dict]) -> None:
    print("\n== Foundation models visible in this region ==")
    for model in sorted(models, key=lambda item: item["modelId"]):
        if not any(key in model["modelId"].lower() for key in INTERESTING):
            continue
        lifecycle = (model.get("modelLifecycle") or {}).get("status", "?")
        print(
            f"  {model['modelId']:<55} {model.get('providerName', ''):<12} "
            f"lifecycle={lifecycle:<12} streaming={model.get('responseStreamingSupported')}"
        )

    print("\n== Cross-region inference profiles ==")
    for profile in sorted(profiles, key=lambda item: item["inferenceProfileId"]):
        if not any(key in profile["inferenceProfileId"].lower() for key in INTERESTING):
            continue
        regions = sorted(
            model["modelArn"].split("/")[-1].split(":")[0]
            if ":" in model.get("modelArn", "")
            else ""
            for model in profile.get("models", [])
        )
        print(
            f"  {profile['inferenceProfileId']:<55} status={profile.get('status', '?'):<10} "
            f"routes={','.join(region for region in regions if region)}"
        )


def _check_id(configured: str, models: list[dict], profiles: list[dict], label: str) -> bool:
    model_ids = {model["modelId"] for model in models}
    profile_ids = {profile["inferenceProfileId"] for profile in profiles}
    found = configured in model_ids or configured in profile_ids
    where = (
        "model"
        if configured in model_ids
        else ("inference profile" if configured in profile_ids else "NOT FOUND")
    )
    status = "CONFIRMED" if found else "NOT FOUND"
    print(f"\n{label}: {configured} -> {status} ({where})")
    return found


def _try_calls(session: boto3.Session, args: argparse.Namespace, profile: str | None) -> bool:
    settings = Settings(
        aws_region=args.region,
        aws_profile=profile,
        embeddings_provider="bedrock",
        llm_provider="bedrock",
        embedding_model_id=args.embedding_model,
        embedding_style=args.embedding_style,
        embedding_dim=args.embedding_dim,
        llm_model_id=args.llm_model,
        bedrock_max_tokens=200,
        log_level="WARNING",
    )
    ok = True

    print("\n== Live embedding call ==")
    try:
        started = time.perf_counter()
        embedder = BedrockEmbedder(settings)
        batch = embedder.embed(
            ["AI risk management is the process of identifying and mitigating AI risks."],
            kind="query",
        )
        elapsed = (time.perf_counter() - started) * 1000
        vector = batch.vectors[0]
        print(
            f"  OK: dim={len(vector)} tokens={batch.tokens} latency={elapsed:.0f}ms "
            f"first_values={[round(float(value), 4) for value in vector[:4]]}"
        )
    except Exception as exc:  # noqa: BLE001 - this diagnostic must never crash
        ok = False
        print(f"  FAILED: {exc}")

    print("\n== Live Converse call ==")
    try:
        started = time.perf_counter()
        chat = BedrockChat(settings)
        generation = chat.generate(
            "What is AI risk management?",
            [
                "[1] Source: sample, page 1. AI risk management is the process of "
                "identifying, assessing and mitigating risks from AI systems."
            ],
        )
        elapsed = (time.perf_counter() - started) * 1000
        print(
            f"  OK: model={generation.model_id} latency={elapsed:.0f}ms "
            f"tokens_in={generation.input_tokens} tokens_out={generation.output_tokens}"
        )
        print(f"  answer: {generation.text[:300]}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAILED: {exc}")

    try:
        budgets = session.client("budgets", region_name="us-east-1")
        account = session.client("sts").get_caller_identity()["Account"]
        response = budgets.describe_budgets(AccountId=account)
        print("\n== AWS Budgets ==")
        for budget in response.get("Budgets", []):
            limit = budget.get("BudgetLimit", {})
            print(
                f"  {budget['BudgetName']}: {limit.get('Amount', '?')} {limit.get('Unit', '')} "
                f"({budget.get('TimeUnit', '')})"
            )
        if not response.get("Budgets"):
            print("  No budgets found. Run scripts/setup_budget.py --email you@example.com")
    except (BotoCoreError, ClientError) as exc:
        print(f"\n== AWS Budgets ==\n  Could not read budgets: {exc}")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="ap-southeast-1")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--embedding-model", default="amazon.titan-embed-text-v2:0")
    parser.add_argument("--embedding-style", default="auto", choices=["auto", "titan", "cohere"])
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--llm-model", default="apac.amazon.nova-lite-v1:0")
    parser.add_argument("--skip-calls", action="store_true", help="Only list models, do not invoke")
    args = parser.parse_args()

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"Bedrock verification · {stamp} · region={args.region}")
    session = _session(args.profile, args.region)
    try:
        _show_identity(session)
    except NoCredentialsError:
        print("No AWS credentials found. Configure a profile first:")
        print("  aws configure sso        # then export AWS_PROFILE=...")
        raise SystemExit(2) from None
    except (BotoCoreError, ClientError) as exc:
        print(f"Cannot call AWS: {exc}")
        raise SystemExit(2) from None

    models, profiles = _list_models(session)
    _print_models(models, profiles)
    embeddding_ok = _check_id(args.embedding_model, models, profiles, "Embedding model")
    llm_ok = _check_id(args.llm_model, models, profiles, "LLM model")

    if args.skip_calls:
        print("\nSkipped live calls (--skip-calls).")
        return

    calls_ok = _try_calls(session, args, args.profile)
    if not (embeddding_ok and llm_ok and calls_ok):
        print(
            "\nOne or more checks failed. Options: verify the model IDs in the Bedrock "
            "console, try the Cohere embedding fallback (--embedding-model "
            "cohere.embed-multilingual-v3), or use another supported region."
        )
        raise SystemExit(1)
    print("\nAll Bedrock checks passed.")


if __name__ == "__main__":
    main()
