import argparse
import sqlite3
from contextlib import closing

from vay.tools.db_queries import init_db
from vay.tools.db_schema import DB_PATH


def status(conn: sqlite3.Connection) -> None:
    print(f"\nnexatel_customers.db ready at {DB_PATH.resolve()}")
    for table in (
        "customers",
        "plans",
        "subscriptions",
        "bills",
        "payments",
        "tickets",
        "coverage",
    ):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:15} {n} rows")

    print("\nSample customers:")
    for row in conn.execute(
        "SELECT c.phone_number, c.name, c.account_type, s.plan_id "
        "FROM customers c LEFT JOIN subscriptions s ON s.phone_number = c.phone_number "
        "ORDER BY c.phone_number"
    ):
        print(
            f"  {row['phone_number']}  {row['name']:18} {row['account_type']:10} plan={row['plan_id']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexatel mock customer DB setup / status check")
    parser.add_argument("--reset", action="store_true", help="Wipe and reseed the database")
    args = parser.parse_args()

    with closing(init_db(reset=args.reset)) as conn:
        status(conn)
