# Compliance, Multi-Layer Guardrails & Human Handoff

This document provides a comprehensive technical breakdown of the safety, compliance, identity verification, multi-layer guardrails, and human escalation mechanisms implemented in VAY.

---

## 1. Multi-Layer Guardrail Architecture

To ensure zero hallucination on sensitive customer accounts, prevent credential leakage, and maintain strict telecommunications regulatory compliance, VAY implements a **4-Layer Defense-in-Depth Guardrail System**.

```mermaid
flowchart TD
    CustomerUtterance([Customer Utterance / Transcript]) --> L1[Layer 1: Input & NLU Guardrails]
    
    subgraph L1_Detail ["Layer 1: Input & NLU"]
        PIIScan{Sensitive PII in Transcript?<br/>Aadhaar, Card, Bank Acc} -->|Yes| PIIHandoff[Force Redacted Human Handoff]
        HumanReq{Explicit Human Request?} -->|Yes| DirectHandoff[Route to Human Handoff]
        AbuseScan{Dual-Gate Abuse Detection<br/>LLM + Regex Match} -->|Strike 1: Warning<br/>Strike 2: Call Cut| AbuseAction[Warning / Closing Node]
        ConfFloor{NLU Confidence >= 0.40?} -->|No| ClarifyEscalate[Clarify or Escalate]
    end
    
    L1 --> L2[Layer 2: Identity & Tool Authorization]
    
    subgraph L2_Detail ["Layer 2: Identity & Tool Execution"]
        IdentityCheck{Entity Phone == Session Phone?} -->|Mismatch| IdentityRefusal[identity_mismatch_node<br/>Deterministic Refusal]
        ToolAuth{Session Verified?} -->|No| SensitiveDenial[Reject Sensitive Tools]
        TwoPhaseConsent{Sensitive Action Requested?<br/>changePlan / sendPaymentLink} -->|Stage Action| StopAndSay["STOP_AND_SAY Sentinel<br/>(Bypasses LLM paraphrasing)"]
        DupQueryCheck{Jaccard Token Overlap >= 0.50?} -->|Yes| NudgeLLM[Block Tool Repeat & Nudge LLM]
    end
    
    L2 --> L3[Layer 3: Output & Retrieval Grounding]
    
    subgraph L3_Detail ["Layer 3: Output Guardrail Node"]
        RetCheck{Retrieval Score >= min_similarity?} -->|No / Low Conf| HandoffConfidence[Route to Human Handoff]
        UncertaintyCheck{Uncertainty Phrase AND Score < 0.50?} -->|Yes| HandoffUncertainty[Route to Human Handoff]
        OutPIICheck{PII / Token Leak in Draft?} -->|Yes| BlockDraft[Block Draft & Route to Handoff]
        ComplianceKB{Sensitive Action Keywords?} -->|Verify| PolicySearch[compliance_policy_search]
        DetoxCheck{Repetition Loop or Fragment?} -->|Yes| DetoxOrFallback[Detoxify or Localized Fallback]
    end
    
    L3 --> L4[Layer 4: Human Escalation & Audit Queue]
    
    subgraph L4_Detail ["Layer 4: Escalation & Audit"]
        RedactPII[PII Redaction Engine] --> AppendLog[Append Structured Record to handoff_log.jsonl]
        AppendLog --> SpokenHandoff[Play Localized Handoff Audio]
        SpokenHandoff --> CleanSession[Reset Session Context for Next Caller]
    end
```

---

## 2. Layer 1: Input & NLU Guardrails

Input guardrails inspect the raw customer transcript before any domain sub-agent, tool, or database query is executed.

### 2.1 Sensitive PII Disclosure Guardrail (`_contains_sensitive_pii`)
- **Inspection Target**: Raw transcript string.
- **Pattern Coverage**:
  - **Aadhaar Numbers**: 12-digit Indian national identity numbers (`\b\d{4}\s?\d{4}\s?\d{4}\b`).
  - **Payment Card Numbers**: 13-19 digit Visa, MasterCard, RuPay, and Amex numbers validated against regex and format checks.
  - **CVV / Security Codes**: 3-4 digit card verification values.
  - **Bank Account / Passwords**: Raw credential patterns.
- **Enforcement**: If sensitive PII is spoken by the customer, the orchestrator immediately marks `sensitive = True`, bypassing all domain sub-agents and routing directly to `human_handoff_node`. The raw transcript is redacted prior to escalation logging.

### 2.2 Dual-Gate Abuse & Toxicity Policy
To protect against false positives from small LLM classifiers while maintaining a safe environment:
1. **Gate 1**: The orchestrator LLM outputs `"aggressive": true`.
2. **Gate 2 (Deterministic)**: The raw transcript must match `ABUSIVE_LANGUAGE_PATTERN` (profanity, hostile insults, or harassment terms).
- **Strike 1 (Warning)**: The assistant issues a polite, firm warning (`warning_node`) asking the customer to maintain professional communication.
- **Strike 2+ (Call Termination)**: The assistant politely ends the call (`closing_node`) using deterministic `CALL_CUT_TEMPLATES`. Abusive callers are **not** transferred to human agents.

### 2.3 Low-Confidence & Clarification Gate
- If orchestrator NLU confidence is below `DEFAULT_NLU_CONFIDENCE` (0.40) or the intent is ambiguous:
  - **Turn 1**: The assistant reprompts the user politely using `clarify_node` (`CLARIFY_TEMPLATES`).
  - **Turn 2+ (`UNCLEAR_ESCALATION_THRESHOLD = 2`)**: If the caller remains unclear across consecutive turns, the call escalates cleanly to a human representative.

---

## 3. Layer 2: Identity & Tool Authorization Guardrails

Layer 2 ensures that sub-agents cannot act beyond their authorization scope or mutate customer accounts without explicit verification.

### 3.1 Identity Mismatch Guardrail
- **Session-Bound Identity**: The caller's phone number is bound once to `SessionContext` at call intake and is immutable.
- **Entity Verification**: If NLU extracts an explicit target phone number from user speech (e.g. *"Change the plan on my friend's number 9876543210"*), the orchestrator compares `_normalize_phone(entity_phone)` against `_normalize_phone(session.phone_number)`.
- **Deterministic Refusal**: If a mismatch is detected, `identity_mismatch_node` executes immediately, speaking a fixed refusal (`IDENTITY_MISMATCH_TEMPLATES`) and logging the audit event without executing any backend tools.

### 3.2 Two-Phase Code-Enforced Consent Gate

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant SubAgent as Sub-Agent Tool Loop
    participant Tool as Sensitive Backend Tool
    participant Session as SessionContext (State)
    participant Graph as LangGraph Orchestrator

    Customer->>SubAgent: "Upgrade me to Prepaid Plus"
    SubAgent->>Tool: changePlan(new_plan_id="PPD_PLUS")
    Note over Tool: Step 1: Stage Action Only
    Tool->>Session: session.pending_action = {"tool": "changePlan", "args": {...}}
    Tool-->>SubAgent: STOP_AND_SAY: <Verbatim Consent Script>
    Note over SubAgent: Bypasses LLM paraphrasing
    SubAgent-->>Customer: "You are switching to Prepaid Plus at Rs 399/mo. Say YES to confirm or NO to cancel."
    
    Customer->>Graph: "Yes, please proceed"
    Note over Graph: Step 2: Code Confirmation Regex (AFFIRMATION_PATTERN)
    Graph->>Tool: confirm_pending_action(session)
    Note over Tool: Commit DB Mutation in SQLite
    Tool-->>Graph: Success Confirmation
    Graph-->>Customer: "Your plan has been successfully upgraded to Prepaid Plus."
```

- **Staging Only**: Sensitive tools (`changePlan`, `sendPaymentLink`) never mutate the database on their initial call.
- **Verbatim Delivery**: The `STOP_AND_SAY:` sentinel bypasses LLM prompt formatting, ensuring mandatory regulatory terms are delivered verbatim.
- **Deterministic Confirmation**: The next turn's transcript is evaluated with exact regular expressions:
  - `AFFIRMATION_PATTERN`: `r"^\s*(yes|confirm|agree|proceed|sure|ok|ஆம்|சரி|हाँ|स्वीकार|ha|haan)\b"`
  - `NEGATION_PATTERN`: `r"^\s*(no|cancel|stop|dont|don't|இல்லை|வேண்டாம்|नहीं|रद्द)\b"`
  The decision to commit the transaction is strictly code-driven and never delegated to LLM interpretation.

---

## 4. Layer 3: Output & Retrieval Grounding Guardrails (`guardrail_node`)

Before any draft reply is approved for voice synthesis, `guardrail_node` executes comprehensive safety checks:

| Guardrail Check | Trigger Condition | Enforcement Action |
|---|---|---|
| **Retrieval Confidence Gate** | `retrieval_score < DEFAULT_MIN_SIMILARITY` (0.30) | Mark `handoff = True`, route to `human_handoff_node` |
| **Grounded Uncertainty Check** | `UNCERTAINTY_PATTERNS.search(draft)` AND `retrieval_score < 0.50` | Route to `human_handoff_node` (allows appropriate caveating if score >= 0.50) |
| **Output PII Leakage** | `PII_LEAK_PATTERNS.search(draft)` (API keys, tokens, OTPs, raw DB rows) | Suppress draft, route to `human_handoff_node` |
| **Compliance Policy Check** | Draft contains sensitive keywords (`change plan`, `payment link`, `cancel`) | Queries `compliance_policy` via `compliance_policy_search()` to verify consent terms |
| **Anti-Repetition Detox** | `_detoxify_repetition` detects token loop; `_is_complete_reply` validates punctuation | Truncates loop or substitutes safe localized fallback |

---

## 5. Layer 4: Human Escalation & Audit Queue

When an escalation occurs from any layer, `human_handoff_node` logs a complete, auditable incident packet to `handoff_log.jsonl`.

### 5.1 Redacted Context Packet Structure

```json
{
  "timestamp": "2026-08-18T14:32:01.452120",
  "phone_number": "9876543210",
  "language": "ta",
  "intent": "dispute_charge",
  "route": "billing",
  "reason": "Low retrieval confidence (0.24 < 0.30).",
  "transcript": "Why was I charged 150 rupees extra on my bill?",
  "entities": {
    "charge_type": "roaming"
  },
  "normalized_query": "Explain roaming charge dispute",
  "draft_reply_at_handoff": "..."
}
```

### 5.2 Clean Session Reset
Upon completing handoff speech synthesis:
1. The call loop automatically clears active conversation history, state variables, and `SessionContext`.
2. The user interface resets to the ready state, preventing context bleeding between calls.
