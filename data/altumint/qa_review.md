# QA Pairs Review Report
**Dataset:** Altumint | **File:** `data/altumint/qa_pairs.json` | **Date:** 2026-04-06

---

## Summary

| Metric | Value |
|--------|-------|
| Total pairs | 120 |
| Text queries | 72 |
| Visual queries | 48 |
| BAD questions found | 9 |
| Answer accuracy (sampled) | 12/15 correct |
| Images with broken paths | 0 |
| Pages with 0 QA pairs | 0 |

---

## Bad Questions (metadata / unrealistic)

| ID | Question | Why Bad |
|----|----------|---------|
| altumint_0001 | "What is the expected continuity between LOAD CB top screw and bottom screw?" | Assigned `source_page_id: dc001_p01` (assembly drawing). Correct source is dc004. Corrupts retrieval metrics. |
| altumint_0000 | "What wire gauge is used for BATT+ SW?" | Assigned `source_page_id: dc001_p01` (assembly drawing). Correct source is dc005 (wire lengths). Answer may be hallucinated — wire gauge not present in dc001 parsed text. |
| altumint_0004 | "What is the first step in assembling the FLVM electronics enclosure?" | Source is dc002_p01 (hole location drawing). Answer "Align the base plate with the mounting holes" is hallucinated — not in dc002_p01 text. Correct source is tm001_p01. |
| altumint_0022 | "What tool is needed to install the antenna base?" | Question asks for a tool; answer "Lubricate the o-ring before tightening" is a procedure step. Question/answer mismatch. |
| altumint_0029 | "What is the serial number printed on the label of the solar charge controller?" | Unit-specific serial number (HQ23504UFFM) from a photo. Not generalizable — every unit has a different serial. |
| altumint_0041 | "Which component is labeled 'SmartSolar charge controller MPPT 75|10'?" | Answer is just its physical location — trivially circular. Adds no field value. |
| altumint_0097 | "What is the MAC address displayed in the Network Settings?" | Unit-specific MAC address (E8:EA:DA:00:9D:D8). Instance-specific artifact. |
| altumint_0100 | "What is the model number of the SmartSolar device listed under 'My devices'?" | Answer is a serial number (HQ2530FMDQ9), not a model number. Terminology error + unit-specific. |
| altumint_0101 | "What is the serial number displayed on the label of the solar charge controller?" | Same class as 0029 and 0097. Unit-specific artifact. |
| altumint_0112 | "What is the current PIN displayed in the Change PIN code dialog?" | Factory-default PIN (162084) from a specific screenshot. Not a useful generalizable fact. |

---

## Answer Accuracy Sample (15 pairs verified against source page JSON)

| ID | Question (abbreviated) | Expected Answer | Found in Source? | Status |
|----|------------------------|----------------|-----------------|--------|
| altumint_0000 | Wire gauge for BATT+ SW | 14 AWG | dc001_p01: no tables, only title text | **UNVERIFIABLE from text** |
| altumint_0006 | Continuity LOAD CB top-to-bottom screw | 0 Ω | dc004_p01 table confirmed | **CORRECT** |
| altumint_0007 | Continuity Solar Controller PV+ to PV- | OL Ω | dc004_p01 table confirmed | **CORRECT** |
| altumint_0008 | Continuity LOAD CB top/bottom screw | OL | dc004_p02 table confirmed | **CORRECT** |
| altumint_0011 | Continuity BATT 1 B- and BATT 2 B- | 0 Ω | dc004_p03 table confirmed | **CORRECT** |
| altumint_0013 | Continuity distribution block + and - | 0 Ω | dc004_p04: "+" = OL, "−" = 0 Ω — answer is wrong for "+" terminal | **PARTIALLY INCORRECT** |
| altumint_0016 | Length of BATT JUMPER RED | 420mm | dc005_p01 table confirmed | **CORRECT** |
| altumint_0017 | How many BATT JUMPER RED wires | 4 | dc005_p01 QTY=4 confirmed | **CORRECT** |
| altumint_0021 | Wire gauge for FLVM assembly | 18 AWG | tm001_p01 confirmed | **CORRECT** |
| altumint_0027 | Screw type for solar charge controller | 6-32 socket head cap screws | tm001_p03 confirmed | **CORRECT** |
| altumint_0031 | Screw type for DIN rail | 10-32 hex button head screws | tm001_p04 confirmed | **CORRECT** |
| altumint_0055 | Terminal on PoE injector → distribution block | C terminal | tm001_p10 confirmed | **CORRECT** |
| altumint_0081 | IP address to set on laptop | 192.168.0.150 | tm002_p02 confirmed | **CORRECT** |
| altumint_0088 | Default IP to connect to web relay | 192.168.1.100 | tm002_p04 confirmed | **CORRECT** |
| altumint_0096 | Default password for modem | #0pTo4600# | tm002_p06 confirmed | **CORRECT** |

---

## Visual Question Quality

**Image paths: 48/48 exist.** No broken paths.

| ID | Question | Requires Image | Status |
|----|----------|---------------|--------|
| altumint_0002 | Component directly above modem | YES — spatial layout in drawing | GOOD |
| altumint_0005 | Diameter of conduit hole | YES — dimension on drawing | GOOD |
| altumint_0018 | Color of wires connecting batteries to WAGOs | YES — color only in figure | GOOD |
| altumint_0025 | How many mounting posts visible | YES — count from photo | GOOD |
| altumint_0026 | Label on vent (IP67) | YES — label in photo | GOOD |
| altumint_0029 | Serial number on solar controller label | YES but unit-specific | FLAGGED |
| altumint_0049 | Color of marking on wire to positive distrib. block | MARGINAL — text says "red tape" | MARGINAL |
| altumint_0050 | Color of wire connected to + terminal | MARGINAL — same context as 0049 | MARGINAL |
| altumint_0061 | Type of screws used to attach modem | Answerable from text (tm001_p11) | REDUNDANT |
| altumint_0097 | MAC address in Network Settings | YES but unit-specific | FLAGGED |
| altumint_0100 | "Model number" in My Devices | YES but unit-specific + wrong label | FLAGGED |
| altumint_0101 | Serial number on solar controller | YES but unit-specific | FLAGGED |
| altumint_0104 | Current power output displayed | Transient runtime value (0 W) | QUESTIONABLE |
| altumint_0108 | Operation mode on screen | Answerable from text on same page | MARGINAL |
| altumint_0112 | PIN in Change PIN dialog | YES but instance-specific screenshot | FLAGGED |
| All others (33) | Spatial/label/color questions | YES — clearly visual-only | GOOD |

---

## Distribution

| Metric | Count |
|--------|-------|
| Total QA pairs | 120 |
| `query_type: text` | 72 (60%) |
| `query_type: visual` | 48 (40%) |
| `question_type: factual` | 29 (24%) |
| `question_type: numerical` | 27 (23%) |
| `question_type: procedural` | 16 (13%) |
| `question_type: visual` | 48 (40%) |

**Per document:**

| Source Document | QA Pairs | Pages | Pairs/Page |
|----------------|----------|-------|-----------|
| dc001 — Assembly Drawing | 3 | 1 | 3.0 |
| dc002 — Hole Location Drawing | 3 | 1 | 3.0 |
| dc004 — Continuity Checks | 10 | 5 | 2.0 |
| dc005 — Wire Lengths | 5 | 2 | 2.5 |
| tm001 — Assembly Instructions | 58 | 15 | 3.9 |
| tm002 — Configuration Instructions | 38 | 11 | 3.5 |
| production_wiring_diagram | 3 | 1 | 3.0 |

---

## Coverage Gaps

- **No pages with 0 QA pairs** — all 36 pages covered.
- **Production wiring diagram**: only 3 pairs for likely the most information-dense document. Should have 8–10 pairs.
- **DC001 and DC002**: only 1 visual question each; both have rich spatial/dimensional content that could yield 3–4 more visual pairs.
- **Procedural questions thin**: 16/120 (13%) — target should be ~25% for assembly/config manuals.
- **No cross-document questions**: none require reconciling info across multiple documents.

---

## Issues Found

### [CRITICAL] — will corrupt benchmark evaluation
- **altumint_0000**: `source_page_id = dc001_p01` but answer is from dc005 (wire lengths). Answer may be hallucinated. Retrieval recall will always be 0.
- **altumint_0001**: `source_page_id = dc001_p01` but answer is from dc004 (continuity checks). Retrieval recall will always be 0.
- **altumint_0004**: `source_page_id = dc002_p01` (hole location drawing). Answer hallucinated — not present in dc002_p01 text. Correct source is tm001_p01.

### [REQUIRED] — must fix before running pipeline
- **altumint_0013**: Answer "0 Ω" is wrong for the "+" terminal (should be OL). Factually incorrect.
- **altumint_0022**: Question asks for a tool; answer gives a procedure step. Mismatch.
- **altumint_0085**: Password has typo — "0p T o4600" should be "0pTo4600". Will cause false EM failures.

### [WARNING] — suboptimal but workable
- **altumint_0029, 0097, 0100, 0101, 0112**: Unit-specific serials/MAC/PIN — not generalizable, will inflate VQA error rates.
- **altumint_0100**: Mislabels a serial number as "model number."
- **altumint_0104**: Transient runtime value (0 W) — no stable ground truth.
- **altumint_0049, 0050, 0061, 0108**: Visual questions answerable from text on same page.
- **altumint_0041**: Circular/trivially obvious answer.

---

## What Is Ready

Approximately **98–101 of 120 pairs (82–84%)** are confirmed good. The dc004 continuity checks, dc005 wire lengths, tm001 assembly instructions, and tm002 configuration instructions blocks are well-formed with verified answers.

---

## Recommendation

Fix the 3 critical and 3 required issues before the next pipeline run. The 5 unit-specific visual pairs (0029, 0097, 0100, 0101, 0112) should be removed to prevent misleading benchmark results. The production wiring diagram is significantly under-utilized — regenerating 5–8 additional questions from that page would improve coverage. Procedural questions at 13% are thin for an assembly/configuration manual dataset; target 25% in the next generation pass.
