# Nexatel Support Knowledge Base — Troubleshooting, FAQs & Complaint Policy

This is the authoritative Support-KB / FAQ knowledge base for the Nexatel customer-care voice
assistant's Complaints & Service-Request Agent. It covers troubleshooting guides, known-issue
articles, and SLA/complaint policy.

## Complaint & Ticket Policy

### Complaint Categories
Nexatel classifies complaints into: **Network** (call drops, no signal, slow data),
**Billing** (disputed charges, incorrect bill), **Service Request** (SIM swap, address
change, plan issue), **Device/Technical** (APN, handset settings), and **Other**.

### Service Level Agreements (SLA)
| Category | Acknowledgement | Target Resolution |
|---|---|---|
| Network issue (single customer) | Immediate (auto-ticket) | 48 hours |
| Network issue (area-wide outage) | Immediate | Per outage ETA (see Technical-KB) |
| Billing dispute | Within 4 hours | 5 working days |
| SIM swap / replacement | Immediate | 24 hours (after ID verification) |
| Plan-related issue | Within 4 hours | 24 hours |
| General service request | Within 24 hours | 3 working days |

If a ticket is not resolved within its SLA, it is automatically escalated to the next support
tier, and the customer is proactively notified with a revised ETA.

### Escalation Levels
1. **Level 1 — Assistant / Front-line agent**: handles routine queries and standard
   troubleshooting.
2. **Level 2 — Specialist team**: handles unresolved technical issues, repeated complaints on
   the same issue, or billing disputes above ₹1,000.
3. **Level 3 — Nodal Officer**: per TRAI regulation, every telecom operator must provide a
   Nodal Officer for complaints unresolved after Level 2, reachable via a dedicated grievance
   channel; response required within 3 days.
4. **Appellate Authority**: the final internal escalation level if the Nodal Officer's
   resolution is unsatisfactory, per TRAI's Telecom Consumers Complaint Redressal
   Regulations.

### When to Escalate to a Human Agent Immediately
- Customer explicitly asks for a human/agent/manager.
- Customer expresses anger, frustration, or repeats the same unresolved complaint a second
  time in the same call.
- The issue involves a cancellation request or a billing dispute above ₹1,000.
- The assistant's confidence in resolving the issue is low, or the customer's issue doesn't
  match any known troubleshooting flow.

## Troubleshooting Guides

### Call Drops / Poor Call Quality
1. Confirm the customer's location is not a known low-coverage area (cross-check via the
   Coverage & Technical agent / `checkCoverage`).
2. Ask the customer to toggle Airplane Mode on/off to force a network re-registration.
3. Confirm the SIM is properly seated and not damaged (common after phone case changes).
4. If using VoLTE, confirm VoLTE is enabled in phone settings — 4G/5G calls fail over to
   weaker legacy networks if VoLTE is off.
5. If unresolved after these steps, log a Network complaint ticket for field-team review.

### Slow or No Mobile Data
1. Confirm a data plan is active and not expired/exhausted (`getBalance`/`getBillBreakup`
   context).
2. Ask the customer to check APN settings match Nexatel's standard APN (see Technical-KB for
   exact values).
3. Ask the customer to restart the device and toggle mobile data off/on.
4. Check for an area outage (`getOutageStatus`) before assuming a device-side issue.
5. If data was reduced to 2G/throttled speed, check whether the plan's FUP data cap for the
   day/month has been crossed.

### SMS Not Sending/Receiving
1. Confirm SMS balance/plan is active.
2. Confirm the SMS center number (SMSC) is correctly configured (standard Nexatel SMSC:
   +91-98100-XXXXX region-specific, provided via the app under Settings → SIM Info).
3. If receiving but not sending, check if the number has hit the daily SMS FUP cap
   (100/day standard).

### Cannot Make/Receive Calls (Not a Coverage Issue)
1. Confirm the account isn't suspended for non-payment (`getDueDate`/billing status check).
2. Confirm Do Not Disturb (DND) or call-barring isn't accidentally enabled on the device.
3. Confirm the SIM hasn't been reported lost/blocked accidentally.
4. Escalate to Network team if the above are all ruled out.

### Recharge/Payment Not Reflecting
1. Ask for the payment reference/transaction ID and time of payment.
2. Most payments reflect within 15 minutes; if beyond 30 minutes, log a Billing complaint
   with the transaction reference so the payment gateway team can trace it.
3. Never ask the customer to pay again while a trace is pending.

## Frequently Asked Questions

**Q: How do I check my current balance/data usage?**
A: Say "check my balance" or dial *123# (prepaid) / check the MyNexatel app dashboard
(postpaid/broadband).

**Q: How do I port my number to Nexatel (MNP)?**
A: SMS "PORT <10-digit number>" to 1900 to receive a Unique Porting Code (UPC), then visit
any Nexatel retail outlet with ID/address proof. Porting typically completes within 3–7 working
days per TRAI's Mobile Number Portability regulations.

**Q: How do I cancel my Nexatel connection?**
A: Cancellation requests must be raised as a Service Request ticket; postpaid/broadband
requires clearing any outstanding dues first. Processing takes up to 7 working days. This is
treated as a sensitive request and is routed to a human agent for confirmation.

**Q: My complaint was already logged once — why do I have to explain again?**
A: This should not happen — if a customer references a prior ticket, look it up via
`getTicketStatus` using the ticket ID or phone number rather than asking them to repeat
everything from scratch.

**Q: A customer is asking for a status update on a dispute/ticket they already raised (e.g.
"any update on my roaming charge dispute?") — is this a sensitive new dispute?**
A: No. Checking the STATUS of an existing dispute/ticket is a normal, answerable request —
call `getTicketStatus` and tell the customer the current status/notes directly. Only RAISING
a brand-new billing dispute, a cancellation request, or a suspected fraud case needs to be
treated as sensitive and handed to a human. Do not escalate a status-check question without
first trying to look up the actual ticket.

**Q: A customer says they already have a SIM replacement ticket and wants it approved/
fast-tracked right now — can the assistant approve it?**
A: No. SIM/eSIM replacement always requires an identity-verification step per compliance
policy (see Technical-KB "SIM Swap Fraud Prevention" and the Compliance-Policy KB) — it can
never be approved purely on verbal request in this call. Look up the ticket status if
possible so the customer at least hears where it stands, explain plainly that approval needs
a verification step the assistant can't perform itself, and escalate to a human agent for
that step. This is not a failure to understand the request — it is the correct, compliant
outcome.

**Q: What is the standard SIM replacement process?**
A: See Technical-KB for the full "Guide SIM Swap" flow; a replacement SIM requires ID
verification and takes effect within 24 hours of activation at a retail outlet or via
doorstep verification.

**Q: Can I get a refund for a service outage?**
A: Yes, for verified outages over 24 continuous hours — see Billing-Policy KB's "Refund
Rules" section for exact eligibility and process.

## Known Issues (Current)

- **Area-wide outage tracking**: Ongoing/recent outages by pincode are tracked live — always
  check `getOutageStatus(pincode)` before troubleshooting a "no signal" complaint as a
  device issue, since it may simply be a known, already-being-fixed outage.
- **OTT bundle activation delay**: Newly purchased OTT bundle add-ons can take up to 2 hours
  to reflect in the partner app login — this is expected behavior, not a fault, and does not
  need a ticket unless it exceeds 6 hours.
