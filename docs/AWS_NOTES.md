# AWS notes — what is verified, what is not, and what to check

Scope: Amazon Bedrock from **ap-southeast-1 (Singapore)** for a small, cheap text model
plus an embedding model. Researched 2026-11-25/26 from official AWS documentation, the
AWS Price List API, and the AWS API reference. **No AWS calls were made** during the
build (no credentials), so anything marked UNVERIFIED must be confirmed in your own
account before you rely on it — `python -m scripts.check_bedrock` does that for you.

## 1. Model IDs and availability

| Purpose | ID in this repo | Status | Notes |
|---|---|---|---|
| Text generation | `apac.amazon.nova-lite-v1:0` | **CONFIRMED (docs)** | Nova Lite via APAC cross-region inference profile; Converse and Invoke supported; Singapore is a source and destination region. |
| Embeddings | `amazon.titan-embed-text-v2:0` | **CONFLICTING** | Model card's regional table and the cross-model compatibility page do **not** list ap-southeast-1; the Singapore Price List nevertheless publishes a Titan V2 embedding SKU. Verify with `check_bedrock` before designing around it. |
| Embeddings fallback | `cohere.embed-multilingual-v3` | **CONFIRMED in-region (docs)** | 1024 dimensions, 512-token window, request uses `texts` + `input_type`. The repo supports it via `EMBEDDING_MODEL_ID=cohere.embed-multilingual-v3` and `EMBEDDING_STYLE=cohere`. |
| Small generation fallback | `amazon.nova-micro-v1:0` | **UNVERIFIED** | No published `apac.` inference-profile ID in the docs at research time; `us.`/`eu.` profiles exist. Check `aws bedrock list-inference-profiles --region ap-southeast-1`. |
| Cheap Claude fallback | Claude 3 Haiku | **END-OF-LIFE** | `anthropic.claude-3-haiku-20240307-v1:0` reached EOL in ap-southeast-1 on **2026-11-13**; calls fail with `ResourceNotFoundException`. Do not use old tutorials. |
| Claude 3.5 Haiku | — | **NOT OFFERED** | Not listed in "Models at a glance" at research time. |
| Claude Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` only | **UNVERIFIED for SG** | Global profile only; no bare model ID; Singapore in-region ✗ and Geo ✗ in docs. |

Converse API details (verified for Nova Lite): request `system`, `messages`,
`inferenceConfig`; response `output.message.content[].text`, `stopReason`,
`usage.inputTokens`/`outputTokens`, `metrics.latencyMs`.

- Converse API reference: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
- Nova Lite model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html
- Titan V2 model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-titan-text-embeddings-v2.html
- Region compatibility: https://docs.aws.amazon.com/bedrock/latest/userguide/models-region-compatibility.html
- Cohere multilingual card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-cohere-embed-multilingual.html
- Claude 3 Haiku card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-3-haiku.html

## 2. Model access and IAM

- In commercial regions there is **no "Model access" request step** for most models; the
  console "Model access" flow mainly applies to AWS GovCloud. First invocation
  auto-subscribes (can take up to ~15 minutes to propagate; transient `AccessDeniedException`
  may appear). Amazon models (Nova/Titan) are not Marketplace products.
  https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html
- IAM actions: `bedrock:InvokeModel` (also authorizes Converse);
  `bedrock:InvokeModelWithResponseStream` only if streaming. `AmazonBedrockFullAccess`
  is the easy setup policy, scoped down for production.
- Third-party models may need `aws-marketplace:Subscribe` etc.; Amazon models do not.

## 3. Pricing (ap-southeast-1, from AWS Price List API)

Source file (official, publication date 2026-11-23):
https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/ap-southeast-1/index.json

| Model | Input / 1M tokens | Output / 1M tokens |
|---|---|---|
| Amazon Nova Lite | **$0.081** | **$0.324** |
| Amazon Nova Micro (profile unverified) | $0.047 | $0.188 |
| Titan Text Embeddings V2 (availability conflicting) | **$0.028** | n/a |
| Amazon Nova Pro (reference) | $1.08 | $4.32 |
| Nova 2 Lite (global profile only) | $0.41 | $3.39 |

- Anthropic SKUs are **absent** from the Singapore price file → no on-demand Claude in
  ap-southeast-1, corroborating the docs.
- Cross-region inference-profile calls are priced at source-region rates, no surcharge.
- Cost formula: `tokens/1e6 × price`. Repo estimate for one full run (120k embedding
  tokens + 10 questions × ~1.1k input + ~200 output): **≈ $0.005**.

## 4. Quotas

- Bedrock runtime quotas are per model per region: requests/minute plus on-demand
  input+output tokens/minute; embeddings are throttled by RPM.
- **Exact numeric defaults are UNVERIFIED** in the public docs used here; check
  Service Quotas → Bedrock in the console and the runtime quotas page:
  https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-runtime.html
- New accounts may have lower quotas and slower first-token availability.

## 5. AWS Budgets (USD 5 alert)

`scripts/setup_budget.py` creates a COST/MONTHLY budget with:
- ACTUAL ≥ 80% → email;
- FORECASTED ≥ 100% → email.

Required IAM: `budgets:CreateBudget`, `budgets:UpdateBudget`,
`budgets:CreateNotification`, `budgets:DescribeBudgets`, plus `sts:GetCallerIdentity`.
`AWSBudgetsActionsWithAWSResourceControlAccess` is the managed policy to model after.

References:
- CLI/API structure: https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_budgets_CreateBudget.html
- Managed policy: https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AWSBudgetsActionsWithAWSResourceControlAccess.html

Console path: **Billing and Cost Management → Budgets → Create budget → Cost budget →
USD 5 monthly → alert at 80% actual / 100% forecasted → confirm the subscription email.**

## 6. Data residency gotcha for a policy product

`apac.` does not mean "Singapore only". The Nova Lite APAC profile can route requests to
Tokyo, Seoul, Osaka, Mumbai, Singapore and Sydney. For a policy/compliance workload that
requires Singapore-only processing, no Nova option exists in-region today; you would need
an in-region model (or a different architecture). Document this in the DPA / security
review. SCPs and IAM policies must allow the destination regions.

## 7. Verification checklist (run after configuring SSO)

```bash
brew install awscli                    # AWS CLI was not installed on this machine
aws configure sso                      # region ap-southeast-1
python -m scripts.setup_budget --email you@example.com --amount 5
python -m scripts.check_bedrock --profile <profile>
# Then:
EMBEDDINGS_PROVIDER=bedrock LLM_PROVIDER=bedrock AWS_PROFILE=<profile> \
  docker compose up -d --build
python -m scripts.run_eval --base-url http://localhost:8000
```

`check_bedrock` lists foundation models visible in the account, cross-region inference
profiles, confirms the configured IDs, makes one embedding call and one Converse call,
prints token usage/latency, and lists existing budgets. It exits non-zero if a check
fails, with a concrete hint.
