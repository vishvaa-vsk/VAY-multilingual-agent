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
