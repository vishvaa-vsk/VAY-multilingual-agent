# Nexatel Technical Knowledge Base — Devices, APN, SIM/eSIM & Coverage

This is the authoritative Technical-KB knowledge base for the Nexatel customer-care voice
assistant's Coverage & Technical Agent. It covers device/APN setup, SIM/eSIM guides, and
coverage FAQs.

## APN (Access Point Name) Settings

Nexatel's standard APN settings for data connectivity:

| Field | Value |
|---|---|
| APN Name | Nexatel Internet |
| APN | nexatel.data |
| Proxy | Not set |
| Port | Not set |
| Username | (leave blank) |
| Password | (leave blank) |
| MMSC | http://mms.nexatel.in |
| MMS Proxy | 10.10.10.10 |
| MMS Port | 8080 |
| MCC | 404 |
| MNC | 45 |
| Authentication Type | Not set / PAP or CHAP |
| APN Type | default,supl,mms |
| APN Protocol | IPv4/IPv6 |

**How to set APN manually (Android)**: Settings → Network & Internet → Mobile Network → Access
Point Names → "+" (Add new) → enter the values above → Save → select "Nexatel Internet" as the
active APN.

**How to set APN manually (iPhone)**: Settings → Mobile Data → Mobile Data Network → enter the
APN value under "Mobile Data" section. Most iPhones auto-configure APN via carrier settings
update — if data doesn't work after inserting a Nexatel SIM, prompt the customer to check for
a "Carrier Settings Update" under Settings → General → About.

Most modern devices auto-configure APN via the SIM's embedded carrier profile; manual entry is
only needed on unlocked/imported devices or after a factory reset that didn't restore carrier
settings.

## VoLTE / 5G Configuration

- VoLTE (Voice over LTE) must be enabled for HD voice calls and to avoid call setup delays.
  Enable via Settings → Network & Internet → Mobile Network → toggle "VoLTE calls" / "4G
  Calling".
  On iPhone: Settings → Mobile Data → Mobile Data Options → Voice & Data → select "4G" or
  "5G Auto" (enables VoLTE automatically).
- 5G requires: (1) a 5G-capable device, (2) an active plan with 5G eligibility (see
  Product-Catalog KB), (3) being within a 5G-covered area (`checkCoverage`), and (4) "5G Auto"
  or "5G On" selected in network mode settings.

## SIM & eSIM Guides

### Physical SIM Replacement (Damaged/Lost SIM)
1. Verify identity: full name, registered address, and either the last recharge/bill amount
   or an OTP sent to an alternate registered contact.
2. Log the request as a Service Request ticket (`createComplaint` category="sim_replacement"
   or use the dedicated SIM-swap flow).
3. Customer visits the nearest retail outlet with original ID proof (matching KYC on file) to
   collect the replacement SIM, OR requests doorstep delivery (available in select cities,
   ₹49 fee).
4. The new SIM is activated within 2–4 hours of physical verification; the old SIM stops
   working immediately upon new SIM activation for security.
5. Number, balance, and active plan carry over automatically — no re-recharge needed.

### eSIM Activation
1. Available on eSIM-compatible devices (most flagship phones from 2020 onward).
2. Customer requests an eSIM via the MyNexatel app or a retail outlet; identity verification
   is required exactly as for a physical SIM swap.
3. An eSIM QR code / activation code is generated and sent via email or shown at the outlet.
4. Customer scans the QR code under Settings → Mobile/Cellular → Add eSIM, and the eSIM
   profile downloads and activates within minutes.
5. eSIM can also be transferred to a new device (e.g. after a phone upgrade) using the same
   flow — the old device's eSIM profile is deactivated automatically once the new one
   activates.

### SIM Swap Fraud Prevention
Nexatel never processes a SIM swap without identity verification. If a customer reports they
did NOT request a SIM swap but their SIM stopped working, this is treated as a **suspected
fraud/security case** and is escalated to a human agent immediately — it is not resolved
through standard self-service troubleshooting.

## Coverage & Network FAQs

- **Checking coverage in an area**: Coverage is tracked by pincode with a signal-strength
  rating (Excellent / Good / Fair / Poor / No Coverage) and available technology (4G/5G).
  Always use `checkCoverage(pincode)` rather than a general answer, since coverage varies
  significantly street-to-street in dense urban areas.
- **Reporting an outage**: If multiple customers in the same pincode report "no signal"
  around the same time, check `getOutageStatus(pincode)` — an area-wide outage is usually
  already logged with an estimated resolution time (planned maintenance or fault repair) and
  does NOT require the customer to do individual troubleshooting.
- **Indoor signal issues**: Weak indoor signal in otherwise "Good"/"Excellent" coverage areas
  is often building-material related (steel/concrete attenuation); recommend the customer try
  near a window, or consider Nexatel's WiFi Calling feature (enable in phone settings —
  routes calls over any WiFi network when cellular signal is weak).
- **Rural/highway coverage**: Nexatel's rural and highway coverage is expanding under an
  ongoing network rollout; some rural pincodes may show "Fair" or "No Coverage" with a
  planned upgrade date — check `checkCoverage` for the specific pincode's status and any
  listed rollout ETA.
- **International coverage**: Nexatel has roaming partnerships in 190+ countries; coverage
  and network quality abroad depend on the local partner operator, not Nexatel directly —
  advise checking the specific destination's partner network status before travel for
  business-critical use.

## Device Settings Quick Reference

| Setting | Where to find it (Android) | Where to find it (iPhone) |
|---|---|---|
| APN | Settings → Network & Internet → Mobile Network → APNs | Settings → Mobile Data → Mobile Data Network |
| VoLTE | Settings → Network & Internet → Mobile Network → VoLTE toggle | Settings → Mobile Data → Voice & Data |
| WiFi Calling | Settings → Network & Internet → WiFi Calling | Settings → Mobile Data → WiFi Calling |
| Network Mode (4G/5G) | Settings → Network & Internet → Preferred Network Type | Settings → Mobile Data → Voice & Data |
| SIM/eSIM Info | Settings → Network & Internet → SIMs | Settings → Mobile/Cellular Plans |
