"""Create or update a USD 5/month AWS Budget with email alerts.

Run locally with credentials that can manage Budgets (never from the container):
    python -m scripts.setup_budget --email you@example.com

The script uses the AWS Budgets API directly (boto3), so the AWS CLI is optional.
Requires: sts:GetCallerIdentity, budgets:CreateBudget, budgets:UpdateBudget,
budgets:CreateNotification, budgets:DescribeBudgets, budgets:ViewBudget.
"""

from __future__ import annotations

import argparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


def _build_notifications(email: str, actual_threshold: int, forecast_threshold: int) -> list[dict]:
    return [
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": actual_threshold,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"Address": email, "SubscriptionType": "EMAIL"}],
        },
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": forecast_threshold,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"Address": email, "SubscriptionType": "EMAIL"}],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Email that receives the budget alerts")
    parser.add_argument(
        "--amount", type=float, default=5.0, help="Monthly limit in USD (default: 5)"
    )
    parser.add_argument("--name", default="policy-rag-monthly", help="Budget name")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--region", default="us-east-1", help="Budgets API region (global service)")
    parser.add_argument("--actual-threshold", type=int, default=80)
    parser.add_argument("--forecast-threshold", type=int, default=100)
    args = parser.parse_args()

    session_kwargs: dict[str, object] = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)

    try:
        account_id = session.client("sts").get_caller_identity()["Account"]
    except (NoCredentialsError, BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Cannot authenticate to AWS: {exc}") from exc

    budgets = session.client("budgets")
    budget = {
        "BudgetName": args.name,
        "BudgetLimit": {"Amount": str(args.amount), "Unit": "USD"},
        "BudgetType": "COST",
        "TimeUnit": "MONTHLY",
    }
    notifications = _build_notifications(args.email, args.actual_threshold, args.forecast_threshold)

    created = False
    try:
        budgets.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=notifications,
        )
        created = True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code != "DuplicateRecordException":
            raise SystemExit(f"Could not create budget: {exc}") from exc
        budgets.update_budget(AccountId=account_id, NewBudget=budget)
        for notification in notifications:
            try:
                budgets.create_notification(
                    AccountId=account_id,
                    BudgetName=args.name,
                    Notification=notification["Notification"],
                    Subscribers=notification["Subscribers"],
                )
            except ClientError as notification_error:
                code = notification_error.response.get("Error", {}).get("Code", "")
                if code != "DuplicateRecordException":
                    print(f"Warning: could not create notification: {notification_error}")

    action = "Created" if created else "Updated"
    print(f"{action} budget '{args.name}' ({args.amount} USD/month) for account {account_id}.")
    print(
        f"Alerts sent to {args.email} at {args.actual_threshold}% actual and "
        f"{args.forecast_threshold}% forecasted spend."
    )
    print("Confirm the subscription email from AWS Budgets before relying on alerts.")
    print("Console: https://console.aws.amazon.com/billing/home#/budgets")


if __name__ == "__main__":
    main()
