from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "nexatel_customers.db"


def _days_ago(n: int) -> str:
    """ISO date `n` days before whenever this module is imported/seeded.

    Prepaid subscriptions below used to hardcode a fixed calendar date (e.g.
    "2026-06-20"). Since prepaid validity = activated_on + validity_days, a
    fixed past date silently drifts into "expired" as real time moves past it
    -- 3 of 5 demo prepaid accounts were showing as EXPIRED by the time this
    was caught (see rag-tts-evaluuation.md). Computing relative to seed time
    keeps a freshly-reset DB demo-ready regardless of when --reset is run.
    """
    return (date.today() - timedelta(days=n)).isoformat()

PLANS = [
    (
        "PPD_LITE",
        "Prepaid Lite",
        "prepaid",
        149,
        20,
        "1 GB total",
        "300 min",
        "100 SMS",
        "Entry-level short validity pack",
    ),
    (
        "PPD_BASIC",
        "Prepaid Basic",
        "prepaid",
        239,
        28,
        "1.5 GB/day",
        "Unlimited",
        "100 SMS/day",
        "Everyday basic pack",
    ),
    (
        "PPD_VALUE",
        "Prepaid Value",
        "prepaid",
        299,
        28,
        "2 GB/day",
        "Unlimited",
        "100 SMS/day",
        "Most popular everyday pack",
    ),
    (
        "PPD_PLUS",
        "Prepaid Plus",
        "prepaid",
        399,
        28,
        "3 GB/day",
        "Unlimited",
        "100 SMS/day",
        "Higher data everyday pack",
    ),
    (
        "PPD_84_VALUE",
        "Prepaid 84-Day Value",
        "prepaid",
        859,
        84,
        "2 GB/day",
        "Unlimited",
        "100 SMS/day",
        "Quarterly value pack",
    ),
    (
        "PPD_84_PLUS",
        "Prepaid 84-Day Plus",
        "prepaid",
        1099,
        84,
        "3 GB/day",
        "Unlimited",
        "100 SMS/day",
        "Quarterly plus pack",
    ),
    (
        "PPD_ANNUAL",
        "Prepaid Annual",
        "prepaid",
        3599,
        365,
        "2 GB/day",
        "Unlimited",
        "100 SMS/day",
        "Annual long-validity pack",
    ),
    ("DATA_SMALL", "Data Pack Small", "prepaid", 49, 1, "2 GB", "-", "-", "1-day data-only top-up"),
    (
        "DATA_MEDIUM",
        "Data Pack Medium",
        "prepaid",
        98,
        7,
        "6 GB",
        "-",
        "-",
        "7-day data-only top-up",
    ),
    (
        "YOUTH_UNL",
        "Youth Unlimited",
        "prepaid",
        349,
        28,
        "4 GB/day",
        "Unlimited",
        "100 SMS/day",
        "Youth/student plan (age 18-25)",
    ),
    (
        "POST_SOLO",
        "Postpaid Solo",
        "postpaid",
        399,
        30,
        "40 GB",
        "Unlimited",
        "100/day",
        "Single-line postpaid",
    ),
    (
        "POST_FAMILY",
        "Postpaid Family",
        "postpaid",
        699,
        30,
        "75 GB shared (up to 4 lines)",
        "Unlimited",
        "100/day/line",
        "Family sharing postpaid",
    ),
    (
        "POST_PRO",
        "Postpaid Pro",
        "postpaid",
        999,
        30,
        "100 GB + unlimited 5G",
        "Unlimited",
        "100/day",
        "Pro postpaid with 5G",
    ),
    (
        "POST_INFINITY",
        "Postpaid Infinity",
        "postpaid",
        1499,
        30,
        "Unlimited (FUP 200GB)",
        "Unlimited + 250 min ISD",
        "100/day",
        "Top-tier unlimited postpaid",
    ),
    (
        "FIBER_BASIC",
        "Fiber Basic",
        "broadband",
        599,
        30,
        "Unlimited @100Mbps",
        "-",
        "-",
        "Entry broadband",
    ),
    (
        "FIBER_PLUS",
        "Fiber Plus",
        "broadband",
        899,
        30,
        "Unlimited @300Mbps",
        "-",
        "-",
        "Mid broadband + 2 OTT",
    ),
    (
        "FIBER_MAX",
        "Fiber Max",
        "broadband",
        1299,
        30,
        "Unlimited @500Mbps",
        "-",
        "-",
        "High broadband + 4 OTT",
    ),
    (
        "FIBER_ULTRA",
        "Fiber Ultra",
        "broadband",
        1999,
        30,
        "Unlimited @1Gbps",
        "-",
        "-",
        "Top broadband + 6 OTT + static IP eligible",
    ),
]

# 10-digit sample customers covering every sub-agent demo path
CUSTOMERS = [
    # phone, name, dob, kyc, city, pincode, account_type, lang
    ("9876500001", "Aditi Sharma", "1994-03-12", 1, "Chennai", "600001", "prepaid", "ta"),
    ("9876500002", "Ramesh Kumar", "1988-07-21", 1, "Delhi", "110001", "postpaid", "hi"),
    ("9876500003", "Priya Natarajan", "1996-11-05", 1, "Chennai", "600042", "prepaid", "ta"),
    ("9876500004", "Vikram Singh", "1985-01-30", 1, "Mumbai", "400001", "postpaid", "hi"),
    ("9876500005", "Sneha Reddy", "1999-05-18", 1, "Hyderabad", "500001", "prepaid", "en"),
    ("9876500006", "Arjun Menon", "1991-09-09", 1, "Kochi", "682001", "postpaid", "en"),
    ("9876500007", "Kavya Iyer", "2001-02-14", 1, "Chennai", "600020", "prepaid", "ta"),
    ("9876500008", "Sanjay Gupta", "1980-12-25", 0, "Delhi", "110002", "postpaid", "hi"),
    ("9876500009", "Meena Pillai", "1993-06-30", 1, "Chennai", "600001", "broadband", "en"),
    ("9876500010", "Rahul Verma", "1997-08-08", 1, "Pune", "411001", "prepaid", "hi"),
    # Demo / development test account — used with --phone 9876543210 in run_voice.py
    ("9876543210", "Vishwa Raj", "1995-06-15", 1, "Chennai", "600001", "prepaid", "ta"),
]

# phone -> (plan_id, activated_on, addons)
# Prepaid entries use _days_ago() (relative to validity_days below) so a freshly
# reseeded DB never starts with an already-expired demo plan. Postpaid/broadband
# entries keep fixed historical dates -- they only convey "customer since"
# tenure, which isn't validity-sensitive the way prepaid plans are.
SUBSCRIPTIONS = {
    "9876500001": ("PPD_VALUE", _days_ago(5), ""),  # 28-day validity
    "9876500002": ("POST_PRO", "2025-11-01", "OTT Super Bundle"),
    "9876500003": ("YOUTH_UNL", _days_ago(10), ""),  # 28-day validity
    "9876500004": ("POST_INFINITY", "2024-03-15", "Device Protection Plan,OTT Super Bundle"),
    "9876500005": ("PPD_PLUS", _days_ago(3), ""),  # 28-day validity
    "9876500006": ("POST_FAMILY", "2025-01-10", "Caller Tune"),
    "9876500007": ("PPD_BASIC", _days_ago(7), ""),  # 28-day validity
    "9876500008": ("POST_SOLO", "2025-05-05", ""),
    "9876500009": ("FIBER_PLUS", "2025-09-01", ""),
    "9876500010": ("PPD_84_VALUE", _days_ago(20), ""),  # 84-day validity
    # Demo / development test account
    "9876543210": ("PPD_VALUE", _days_ago(8), ""),  # 28-day validity, Prepaid Value
}

# a few tickets across statuses/categories
TICKETS = [
    (
        "NXT-100234",
        "9876500004",
        "network",
        "Frequent call drops in Andheri West area",
        "open",
        -1,
        2,
    ),
    (
        "NXT-100235",
        "9876500006",
        "billing",
        "Disputed roaming charge of Rs 850 on last bill",
        "in_progress",
        -2,
        5,
    ),
    (
        "NXT-100236",
        "9876500002",
        "technical",
        "5G not showing despite Postpaid Pro plan",
        "resolved",
        -10,
        1,
    ),
    (
        "NXT-100237",
        "9876500008",
        "service_request",
        "Requesting SIM replacement, SIM damaged",
        "open",
        0,
        1,
    ),
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

