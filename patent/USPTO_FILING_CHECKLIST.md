# USPTO Provisional Patent Filing Checklist

**Inventor:** Nikolas Yoo
**Entity Type:** Micro Entity
**Fee:** $160 USD
**Deadline:** File within 12 months from today to lock this priority date.

---

## What You Are Filing

A **Provisional Patent Application (PPA)** under 35 U.S.C. §111(b).

A provisional:
- Establishes your **priority date** (today)
- Does NOT get examined by the USPTO
- Does NOT issue as a patent
- Gives you **12 months** to file a full Non-Provisional Application
- Costs $160 for micro entity

---

## Documents You Need

| Document | Status | Notes |
|---|---|---|
| Technical specification | ✅ DONE | `provisional_patent_specification.txt` |
| USPTO SB/16 cover sheet | ⬜ Fill out | Link below |
| Payment | ⬜ Credit card at filing | $160 micro entity |

---

## Step-by-Step Filing (30 minutes)

### Step 1: Verify Micro Entity Status
You qualify as a micro entity if:
- You are an individual (not a large company)
- Gross income in prior year was under ~$239,000 (2025 threshold)
- You have not been named inventor on more than 4 prior patent applications
- You have not assigned, licensed, or obligated the invention to a non-micro entity

If all apply → file as micro entity, fee = $160.

### Step 2: Create a USPTO Account
Go to: https://my.uspto.gov
Create an account if you don't have one. Free.

### Step 3: Fill Out Cover Sheet (SB/16)
Download form SB/16 from:
https://www.uspto.gov/sites/default/files/documents/sb0016.pdf

Fill in:
- Title: "System and Method for Topology-Aware Invariant Indexing, Cross-Domain
  Dynamical Memory, and Energy-Conditioned Regime Retrieval in Nonlinear
  Dynamical Systems"
- Inventor: Nikolas Yoo
- Correspondence address: your address
- Check: "Micro Entity" box
- Check: "This is a PROVISIONAL APPLICATION" box

### Step 4: File via EFS-Web (Patent Center)
Go to: https://patentcenter.uspto.gov

1. Click "File a new application"
2. Select "Provisional"
3. Upload documents:
   - `provisional_patent_specification.txt` (or PDF version)
   - Completed SB/16 cover sheet
4. Pay $160 by credit card
5. Download your **confirmation receipt with Application Number**

**Save the Application Number.** It is your priority date proof.

### Step 5: Add "Patent Pending" to your code/product
Once filed, you may legally state:
> "Patent Pending — US Provisional Patent Application [your app number]"

---

## What to Do in the Next 12 Months

| Timeline | Action |
|---|---|
| Today | File provisional — locks priority date |
| Month 1-2 | Build MVP SDK + demo dashboard |
| Month 3-4 | Approach 1-2 pilot customers |
| Month 10-11 | Engage a patent attorney for non-provisional |
| Month 12 | File non-provisional (full patent) — must cite this provisional |

If you do not file a non-provisional within 12 months, the provisional expires
and you lose the priority date. The technology is NOT protected beyond that.

---

## What the Specification Covers (4 Claims Areas)

1. **Domain-invariant triple retrieval** — the (log_ω₀_norm, log_Q_norm, ζ)
   indexing system for cross-domain warm-start optimization

2. **Energy-conditioned separatrix gate** — E₀/E_sep regime filtering before
   retrieval to prevent topology-mismatched transfer

3. **Topology discriminator** — curvature-profile cosine similarity achieving
   0.0 between hardening and softening Duffing regimes

4. **Bifurcation-aware trust metric** — EDMD reconstruction error as periodic/
   chaotic classifier + orbit-range-normalized Poincaré period detection

---

## Important Notes

- This document was drafted as technical disclosure. **Not legal advice.**
  If resources allow, have a patent attorney review before the non-provisional.
  For the provisional itself, a well-written pro-se filing is valid and common.

- The provisional specification does NOT need formal claims — the informal
  claims in Section "CLAIMS" are for scope reference only.

- Keep this repo **private** until you decide to publish (or publish strategically
  after filing). Public disclosure before filing would not affect US rights
  (1-year grace period) but would eliminate foreign patent rights permanently.

- If you plan to file internationally (PCT application), you must do so within
  12 months of today's priority date. PCT filing fee is ~$3,000-5,000.

---

## Quick Check Before Filing

- [ ] Specification file reviewed and accurate
- [ ] No obvious typos in inventor name (Nikolas Yoo)
- [ ] Cover sheet (SB/16) complete
- [ ] Micro entity eligibility confirmed
- [ ] USPTO account created
- [ ] $160 payment method ready
- [ ] Address for correspondence filled in specification


                                                                                                               
---                                                             
  What you have and what to do right now 

  The specification covers all four core inventions precisely:

  1. Domain-invariant triple — the (log_ω₀_norm, log_Q_norm, ζ) indexing with cross-domain retrieval. Includes the 2708× and 3.7×
  empirical results as supporting evidence.
  2. Energy-conditioned separatrix gate — E₀/E_sep regime filtering. The key claim is that the detection is in energy space, not
  position space — this is the topologically correct formulation and it's specific enough to defend.
  3. Topology discriminator — curvature profile cosine similarity = 0.0 between hardening and softening regimes. This is the sharpest
  novel claim — I haven't seen this in the Koopman literature.
  4. Bifurcation-aware trust metric — reconstruction error replacing spectral gap for driven systems + orbit-range-normalized Poincaré
   period classification.

  To file right now (30 minutes):

  1. Open patent/provisional_patent_specification.txt — fill in your address on the inventor declaration page
  2. Download form SB/16 from USPTO.gov — fill in your name and address
  3. Go to patentcenter.uspto.gov → File new application → Provisional → upload both files → pay $160

  That's it. You get an application number immediately, which is your priority date.

  One real risk to flag: The spec references "2708×" and "3.7×" and "cosine similarity = 0.0" as empirical results. Make sure those numbers are reproducible from your test suite before anyone asks to verify them. The demo.py and test_bifurcation_regimes.py
  currently back those claims. Don't change the architecture in a way that breaks those results before the non-provisional filing.
---                                                             



*Draft prepared: 2026-02-21*
*Priority date will be: date of USPTO submission*
