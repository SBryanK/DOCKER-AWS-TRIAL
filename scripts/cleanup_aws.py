"""List (and optionally delete) the AWS resources used by this project.

Bedrock is pay-per-token: it has no persistent resource to delete. The only
billable resource this repo creates is the AWS Budget (free, but created by you).
Optional extras you may have added manually are listed as a checklist.

    python -m scripts.cleanup_aws                      # inspect only
    python -m scripts.cleanup_aws --delete-budget      # delete the budget
"""

from __future__ import annotations

import argparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--budget-name", default="policy-rag-monthly")
    parser.add_argument("--delete-budget", action="store_true")
    args = parser.parse_args()

    session_kwargs: dict[str, object] = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)

    try:
        account_id = session.client("sts").get_caller_identity()["Account"]
        budgets = session.client("budgets")
        found = budgets.describe_budgets(AccountId=account_id).get("Budgets", [])
    except (NoCredentialsError, BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Cannot list AWS resources: {exc}") from exc

    matches = [budget for budget in found if budget["BudgetName"] == args.budget_name]
    print(f"Account {account_id} · budgets matching '{args.budget_name}': {len(matches)}")
    for budget in matches:
        limit = budget.get("BudgetLimit", {})
        print(
            f"  - {budget['BudgetName']}: {limit.get('Amount')} {limit.get('Unit')} "
            f"per {budget.get('TimeUnit', '').lower()}"
        )

    if args.delete_budget and matches:
        for budget in matches:
            budgets.delete_budget(AccountId=account_id, BudgetName=budget["BudgetName"])
            print(f"Deleted budget '{budget['BudgetName']}'.")
    elif args.delete_budget:
        print("Nothing to delete.")

    print(
        "\nChecklist for manual extras (this project does not create them by default):"
        "\n  [ ] IAM user/access keys created for the demo        -> IAM console > Users"
        "\n  [ ] CloudWatch log groups starting with /aws/bedrock  -> CloudWatch > Log groups"
        "\n  [ ] S3 buckets created for evaluation artefacts       -> S3 console"
        "\n  [ ] Bedrock custom/provisioned throughput"
        "             -> Bedrock console > Provisioned throughput"
        "\n  [ ] Docker volumes on your laptop (not AWS): docker volume rm "
        "<name>_qdrant_storage <name>_index_data"
        "\nBedrock model invocation itself is pay-per-token and leaves no resource to delete."
    )


if __name__ == "__main__":
    main()
