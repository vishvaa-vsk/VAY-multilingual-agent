import json
import sqlite3
from datetime import date, datetime, timedelta

from vay.tools.db_schema import DB_PATH, SCHEMA
from vay.tools.db_seed_data import COVERAGE, CUSTOMERS, PLANS, SUBSCRIPTIONS, TICKETS


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _iso_days_from_today(offset_days: int) -> str:
    return (date.today() + timedelta(days=offset_days)).isoformat()


def _bill_breakup(
    base: float, extra_data: float = 0, extra_voice: float = 0, roaming: float = 0, vas: float = 0
) -> dict:
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
        [
            (phone, name, dob, kyc, city, pincode, acct, lang, now)
            for (phone, name, dob, kyc, city, pincode, acct, lang) in CUSTOMERS
        ],
    )

    for phone, (plan_id, activated_on, addons) in SUBSCRIPTIONS.items():
        cur.execute(
            "INSERT INTO subscriptions (phone_number, plan_id, activated_on, status, addons) VALUES (?,?,?,?,?)",
            (phone, plan_id, activated_on, "active", addons),
        )

    # Bills: give postpaid/broadband customers one paid + one current bill;
    # make 9876500002's current bill overdue and 9876500006's have a roaming dispute.
    bill_rows = [
        (
            "9876500002",
            "2026-06",
            999 * 1.18,
            _iso_days_from_today(-5),
            "overdue",
            json.dumps(_bill_breakup(999, extra_data=150)),
        ),
        (
            "9876500002",
            "2026-05",
            999 * 1.18,
            _iso_days_from_today(-35),
            "paid",
            json.dumps(_bill_breakup(999)),
        ),
        (
            "9876500004",
            "2026-07",
            1499 * 1.18,
            _iso_days_from_today(10),
            "unpaid",
            json.dumps(_bill_breakup(1499, extra_data=299, vas=99 + 149)),
        ),
        (
            "9876500006",
            "2026-07",
            699 * 1.18 + 850,
            _iso_days_from_today(8),
            "unpaid",
            json.dumps(_bill_breakup(699, roaming=850, vas=15)),
        ),
        (
            "9876500008",
            "2026-07",
            399 * 1.18,
            _iso_days_from_today(12),
            "unpaid",
            json.dumps(_bill_breakup(399)),
        ),
        (
            "9876500009",
            "2026-07",
            899 * 1.18,
            _iso_days_from_today(9),
            "unpaid",
            json.dumps(_bill_breakup(899)),
        ),
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
        notes = (
            "Field team dispatched; awaiting confirmation."
            if status == "in_progress"
            else ("Resolved: VoLTE re-provisioned on account." if status == "resolved" else "")
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
