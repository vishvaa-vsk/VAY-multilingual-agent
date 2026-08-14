"""
customer_db.py

Mock Nexatel customer/operational database (SQLite) backing the sub-agent
tool layer (tools.py). This stands in for the real billing/CRM/network APIs
the mentor doc's Billing/Plans/Complaints/Coverage tools would call in
production.

Schema:
    customers      -- phone_number (10-digit) is the primary key / account id
    plans          -- Nexatel's plan catalog (mirrors kb_docs/product_catalog.md)
    subscriptions  -- which plan a customer currently has
    bills          -- postpaid/broadband bills per customer
    payments       -- payment history
    tickets        -- complaints / service requests
    coverage       -- pincode -> signal/technology/outage status

CLI (mirrors chroma_setup.py's status/--reset pattern):
    python customer_db.py            # create + seed if empty, print status
    python customer_db.py --reset    # wipe and reseed from scratch
"""

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "nexatel_customers.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    phone_number   TEXT PRIMARY KEY CHECK (length(phone_number) = 10),
    name           TEXT NOT NULL,
    dob            TEXT,
    kyc_verified   INTEGER NOT NULL DEFAULT 1,
    city           TEXT,
    pincode        TEXT,
    account_type   TEXT NOT NULL,      -- prepaid | postpaid | broadband
    language_pref  TEXT NOT NULL DEFAULT 'en',
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id        TEXT PRIMARY KEY,
    plan_name      TEXT NOT NULL,
    plan_type      TEXT NOT NULL,      -- prepaid | postpaid | broadband
    price          REAL NOT NULL,
    validity_days  INTEGER,
    data_limit     TEXT,
    voice_minutes  TEXT,
    sms            TEXT,
    description    TEXT
);

CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number    TEXT NOT NULL REFERENCES customers(phone_number),
    plan_id         TEXT NOT NULL REFERENCES plans(plan_id),
    activated_on    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',   -- active | expired | cancelled
    addons          TEXT DEFAULT ''                    -- comma-separated add-on names
);

CREATE TABLE IF NOT EXISTS bills (
    bill_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number   TEXT NOT NULL REFERENCES customers(phone_number),
    billing_period TEXT NOT NULL,       -- e.g. '2026-07'
    amount         REAL NOT NULL,
    due_date       TEXT NOT NULL,
    status         TEXT NOT NULL,       -- paid | unpaid | overdue
    breakup_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number   TEXT NOT NULL REFERENCES customers(phone_number),
    amount         REAL NOT NULL,
    paid_on        TEXT NOT NULL,
    method         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id        TEXT PRIMARY KEY,
    phone_number     TEXT NOT NULL REFERENCES customers(phone_number),
    category         TEXT NOT NULL,     -- network | billing | service_request | technical | other
    description      TEXT NOT NULL,
    status           TEXT NOT NULL,     -- open | in_progress | resolved | escalated
    created_at       TEXT NOT NULL,
    sla_due          TEXT NOT NULL,
    resolution_notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS coverage (
    pincode         TEXT PRIMARY KEY,
    area            TEXT NOT NULL,
    signal_strength TEXT NOT NULL,      -- Excellent | Good | Fair | Poor | No Coverage
    technology      TEXT NOT NULL,      -- 4G | 5G | 4G/5G
    outage_status   TEXT NOT NULL DEFAULT 'none'   -- none | planned_maintenance | fault | resolved
);
"""

PLANS = [
    ("PPD_LITE", "Prepaid Lite", "prepaid", 149, 20, "1 GB total", "300 min", "100 SMS", "Entry-level short validity pack"),
    ("PPD_BASIC", "Prepaid Basic", "prepaid", 239, 28, "1.5 GB/day", "Unlimited", "100 SMS/day", "Everyday basic pack"),
    ("PPD_VALUE", "Prepaid Value", "prepaid", 299, 28, "2 GB/day", "Unlimited", "100 SMS/day", "Most popular everyday pack"),
    ("PPD_PLUS", "Prepaid Plus", "prepaid", 399, 28, "3 GB/day", "Unlimited", "100 SMS/day", "Higher data everyday pack"),
    ("PPD_84_VALUE", "Prepaid 84-Day Value", "prepaid", 859, 84, "2 GB/day", "Unlimited", "100 SMS/day", "Quarterly value pack"),
    ("PPD_84_PLUS", "Prepaid 84-Day Plus", "prepaid", 1099, 84, "3 GB/day", "Unlimited", "100 SMS/day", "Quarterly plus pack"),
    ("PPD_ANNUAL", "Prepaid Annual", "prepaid", 3599, 365, "2 GB/day", "Unlimited", "100 SMS/day", "Annual long-validity pack"),
    ("DATA_SMALL", "Data Pack Small", "prepaid", 49, 1, "2 GB", "-", "-", "1-day data-only top-up"),
    ("DATA_MEDIUM", "Data Pack Medium", "prepaid", 98, 7, "6 GB", "-", "-", "7-day data-only top-up"),
    ("YOUTH_UNL", "Youth Unlimited", "prepaid", 349, 28, "4 GB/day", "Unlimited", "100 SMS/day", "Youth/student plan (age 18-25)"),
    ("POST_SOLO", "Postpaid Solo", "postpaid", 399, 30, "40 GB", "Unlimited", "100/day", "Single-line postpaid"),
    ("POST_FAMILY", "Postpaid Family", "postpaid", 699, 30, "75 GB shared (up to 4 lines)", "Unlimited", "100/day/line", "Family sharing postpaid"),
    ("POST_PRO", "Postpaid Pro", "postpaid", 999, 30, "100 GB + unlimited 5G", "Unlimited", "100/day", "Pro postpaid with 5G"),
    ("POST_INFINITY", "Postpaid Infinity", "postpaid", 1499, 30, "Unlimited (FUP 200GB)", "Unlimited + 250 min ISD", "100/day", "Top-tier unlimited postpaid"),
    ("FIBER_BASIC", "Fiber Basic", "broadband", 599, 30, "Unlimited @100Mbps", "-", "-", "Entry broadband"),
    ("FIBER_PLUS", "Fiber Plus", "broadband", 899, 30, "Unlimited @300Mbps", "-", "-", "Mid broadband + 2 OTT"),
    ("FIBER_MAX", "Fiber Max", "broadband", 1299, 30, "Unlimited @500Mbps", "-", "-", "High broadband + 4 OTT"),
    ("FIBER_ULTRA", "Fiber Ultra", "broadband", 1999, 30, "Unlimited @1Gbps", "-", "-", "Top broadband + 6 OTT + static IP eligible"),
]

# 10-digit sample customers covering every sub-agent demo path
CUSTOMERS = [
    # phone, name, dob, kyc, city, pincode, account_type, lang
    ("9876500001", "Aditi Sharma",     "1994-03-12", 1, "Chennai",   "600001", "prepaid",   "ta"),
    ("9876500002", "Ramesh Kumar",     "1988-07-21", 1, "Delhi",     "110001", "postpaid",  "hi"),
    ("9876500003", "Priya Natarajan",  "1996-11-05", 1, "Chennai",   "600042", "prepaid",   "ta"),
    ("9876500004", "Vikram Singh",     "1985-01-30", 1, "Mumbai",    "400001", "postpaid",  "hi"),
    ("9876500005", "Sneha Reddy",      "1999-05-18", 1, "Hyderabad", "500001", "prepaid",   "en"),
    ("9876500006", "Arjun Menon",      "1991-09-09", 1, "Kochi",     "682001", "postpaid",  "en"),
    ("9876500007", "Kavya Iyer",       "2001-02-14", 1, "Chennai",   "600020", "prepaid",   "ta"),
    ("9876500008", "Sanjay Gupta",     "1980-12-25", 0, "Delhi",     "110002", "postpaid",  "hi"),
    ("9876500009", "Meena Pillai",     "1993-06-30", 1, "Chennai",   "600001", "broadband", "en"),
    ("9876500010", "Rahul Verma",      "1997-08-08", 1, "Pune",      "411001", "prepaid",   "hi"),
]

# phone -> (plan_id, activated_on, addons)
SUBSCRIPTIONS = {
    "9876500001": ("PPD_VALUE", "2026-06-20", ""),
    "9876500002": ("POST_PRO", "2025-11-01", "OTT Super Bundle"),
    "9876500003": ("YOUTH_UNL", "2026-07-25", ""),
    "9876500004": ("POST_INFINITY", "2024-03-15", "Device Protection Plan,OTT Super Bundle"),
    "9876500005": ("PPD_PLUS", "2026-07-30", ""),
    "9876500006": ("POST_FAMILY", "2025-01-10", "Caller Tune"),
    "9876500007": ("PPD_BASIC", "2026-06-01", ""),
    "9876500008": ("POST_SOLO", "2025-05-05", ""),
    "9876500009": ("FIBER_PLUS", "2025-09-01", ""),
    "9876500010": ("PPD_84_VALUE", "2026-05-10", ""),
}

# a few tickets across statuses/categories
TICKETS = [
    ("NXT-100234", "9876500004", "network", "Frequent call drops in Andheri West area", "open", -1, 2),
    ("NXT-100235", "9876500006", "billing", "Disputed roaming charge of Rs 850 on last bill", "in_progress", -2, 5),
    ("NXT-100236", "9876500002", "technical", "5G not showing despite Postpaid Pro plan", "resolved", -10, 1),
    ("NXT-100237", "9876500008", "service_request", "Requesting SIM replacement, SIM damaged", "open", 0, 1),
]

# outage/coverage sample rows (pincode aligned with sample customers)
COVERAGE = [
    ("600001", "Chennai Central", "Excellent", "4G/5G", "none"),
    ("600042", "Chennai - Velachery", "Good", "4G/5G", "none"),
    ("600020", "Chennai - Adyar", "Fair", "4G", "planned_maintenance"),
    ("110001", "Delhi - Connaught Place", "Good", "4G/5G", "none"),
    ("110002", "Delhi - Daryaganj", "Poor", "4G", "fault"),
    ("400001", "Mumbai - Fort", "Excellent", "4G/5G", "none"),
    ("500001", "Hyderabad - Abids", "Good", "4G/5G", "none"),
    ("682001", "Kochi - Ernakulam", "Fair", "4G", "none"),
    ("411001", "Pune - Camp", "Good", "4G/5G", "none"),
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _iso_days_from_today(offset_days: int) -> str:
    return (date.today() + timedelta(days=offset_days)).isoformat()


def _bill_breakup(base: float, extra_data: float = 0, extra_voice: float = 0, roaming: float = 0, vas: float = 0) -> dict:
    subtotal = base + extra_data + extra_voice + roaming + vas
    gst = round(subtotal * 0.18, 2)
    return {
        "plan_rental": base,
        "extra_data_charge": extra_data,
        "extra_voice_charge": extra_voice,
        "roaming_surcharge": roaming,
        "vas_charge": vas,
        "gst_18pct": gst,
        "total": round(subtotal + gst, 2),
    }


def seed(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.executemany(
        "INSERT OR IGNORE INTO plans VALUES (?,?,?,?,?,?,?,?,?)",
        PLANS,
    )

    now = datetime.now().isoformat()
    cur.executemany(
        "INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?,?,?,?)",
        [(phone, name, dob, kyc, city, pincode, acct, lang, now)
         for (phone, name, dob, kyc, city, pincode, acct, lang) in CUSTOMERS],
    )

    for phone, (plan_id, activated_on, addons) in SUBSCRIPTIONS.items():
        cur.execute(
            "INSERT INTO subscriptions (phone_number, plan_id, activated_on, status, addons) VALUES (?,?,?,?,?)",
            (phone, plan_id, activated_on, "active", addons),
        )

    # Bills: give postpaid/broadband customers one paid + one current bill;
    # make 9876500002's current bill overdue and 9876500006's have a roaming dispute.
    bill_rows = [
        ("9876500002", "2026-06", 999 * 1.18, _iso_days_from_today(-5), "overdue",
         json.dumps(_bill_breakup(999, extra_data=150))),
        ("9876500002", "2026-05", 999 * 1.18, _iso_days_from_today(-35), "paid",
         json.dumps(_bill_breakup(999))),
        ("9876500004", "2026-07", 1499 * 1.18, _iso_days_from_today(10), "unpaid",
         json.dumps(_bill_breakup(1499, extra_data=299, vas=99 + 149))),
        ("9876500006", "2026-07", 699 * 1.18 + 850, _iso_days_from_today(8), "unpaid",
         json.dumps(_bill_breakup(699, roaming=850, vas=15))),
        ("9876500008", "2026-07", 399 * 1.18, _iso_days_from_today(12), "unpaid",
         json.dumps(_bill_breakup(399))),
        ("9876500009", "2026-07", 899 * 1.18, _iso_days_from_today(9), "unpaid",
         json.dumps(_bill_breakup(899))),
    ]
    cur.executemany(
        "INSERT INTO bills (phone_number, billing_period, amount, due_date, status, breakup_json) VALUES (?,?,?,?,?,?)",
        bill_rows,
    )

    payment_rows = [
        ("9876500002", 999 * 1.18, _iso_days_from_today(-33), "UPI Autopay"),
        ("9876500004", 1499 * 1.18, _iso_days_from_today(-40), "Credit Card"),
    ]
    cur.executemany(
        "INSERT INTO payments (phone_number, amount, paid_on, method) VALUES (?,?,?,?)",
        payment_rows,
    )

    for ticket_id, phone, category, desc, status, created_offset, sla_days in TICKETS:
        created_at = _iso_days_from_today(created_offset)
        sla_due = _iso_days_from_today(created_offset + sla_days)
        notes = "Field team dispatched; awaiting confirmation." if status == "in_progress" else (
            "Resolved: VoLTE re-provisioned on account." if status == "resolved" else ""
        )
        cur.execute(
            "INSERT OR IGNORE INTO tickets VALUES (?,?,?,?,?,?,?,?)",
            (ticket_id, phone, category, desc, status, created_at, sla_due, notes),
        )

    cur.executemany(
        "INSERT OR IGNORE INTO coverage VALUES (?,?,?,?,?)",
        COVERAGE,
    )

    conn.commit()


def init_db(reset: bool = False) -> sqlite3.Connection:
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Deleted existing DB at {DB_PATH}")

    conn = _connect()
    conn.executescript(SCHEMA)

    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    if count == 0:
        seed(conn)
        print("Seeded mock Nexatel customer database.")
    else:
        print("Database already seeded — leaving existing data as-is (use --reset to reseed).")

    return conn


def status(conn: sqlite3.Connection) -> None:
    print(f"\nnexatel_customers.db ready at {DB_PATH.resolve()}")
    for table in ("customers", "plans", "subscriptions", "bills", "payments", "tickets", "coverage"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:15} {n} rows")

    print("\nSample customers:")
    for row in conn.execute(
        "SELECT c.phone_number, c.name, c.account_type, s.plan_id "
        "FROM customers c LEFT JOIN subscriptions s ON s.phone_number = c.phone_number "
        "ORDER BY c.phone_number"
    ):
        print(f"  {row['phone_number']}  {row['name']:18} {row['account_type']:10} plan={row['plan_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexatel mock customer DB setup / status check")
    parser.add_argument("--reset", action="store_true", help="Wipe and reseed the database")
    args = parser.parse_args()

    with closing(init_db(reset=args.reset)) as conn:
        status(conn)
