# Annotation Guide: AI Discourse Specificity

**Construct:** Where does this snippet fall on the exploratory/aspirational <-> operationally
specific axis?

## Labels

**0 — Exploratory / aspirational**
Forward-looking, strategic-intent language with no concrete deployment detail.
- "We're excited about the potential of AI to transform our business."
- "We believe generative AI will be a key theme for the industry going forward."
- "We're evaluating opportunities in this space."

**1 — Mixed**
Names a direction or initiative but stops short of quantified outcomes or deployment status.
- "We've begun piloting AI tools in our customer service function."
- "We're investing in AI capabilities across the organization."

**2 — Operationally specific**
Names a tool/system/vendor, states deployment status, and/or quantifies an outcome.
- "We deployed [tool] across our claims processing pipeline in Q2, reducing average
  handling time by 20%."
- "We integrated GPT-4 into our customer support workflow; it now handles 30% of
  first-contact resolutions."

## Rules of thumb

- Label the snippet as written, not what you infer the firm probably means.
- A named tool/vendor without a quantified outcome or deployment statement -> usually a 1, not a 2.
- Boilerplate risk-factor language ("AI could pose risks to our business") is scored on the
  same axis: it's exploratory unless it's specific about what's being done about it.
- If a snippet is ambiguous or cut off by the keyword window, label based on what's present.
  Note the truncation in `annotator_notes` rather than guessing at missing context.

## Calibration rules (added after labeling the training set)

These resolve cases the rules above leave ambiguous. Applied consistently across the
LLM-assisted first pass on `annotation_train.csv` (200/200, human-spot-checked and
confirmed) -- apply the same rules when labeling `annotation_holdout.csv`, or deliberately
override them and note why.

- **Industry/market-level commentary that isn't about the firm's own use** (TAM sizing,
  competitor/analyst discussion of AI trends, macro commentary about AI-driven power
  demand or regulation) -> 0, even if it contains numbers, since the numbers describe the
  market, not the firm's own discourse.
- **Analyst questions with no management answer content in the snippet** -> 0, UNLESS the
  question restates a figure management already disclosed elsewhere (e.g. an analyst
  referencing "the 30% of signings you mentioned are AI") -- in that case, score the
  restated figure as if the firm had stated it directly.
- **A named entity/company + only a vague benefit claim, no deployment status, no
  number** -> 0, not 1. A name alone isn't enough; per the base rule above, a named
  tool/vendor needs a quantified outcome or deployment statement to clear even a 1.
- **Snippets with no substantive AI content visible** (the keyword match likely sits just
  outside the sentence window) -> 0 by default, per "label what's present."
- **Future/upcoming products explicitly framed as not-yet-shipped**, or **projected/target
  figures that are not yet realized** (a market-opportunity estimate, an "aim to double"
  target) -> 1, not 2. Only REALIZED outcomes count toward 2.
- **POC/pilot-stage language**, even with a number attached to interest level rather than
  an outcome (e.g. "20 customers interested in a proof of concept") -> 1.
- **A quantified outcome alone counts as 2, even without a named external tool/vendor** --
  e.g. "reduced training time by 80%" with no vendor name still scores 2. The number is
  what does the work, not the brand name.
- **Explicit deployment-status language alone counts as 2, even without a percentage** --
  "generally available yesterday," "shipped," "rolled out," "now live," "field deployed,"
  "integrated" all satisfy the "and/or" in the base 2 definition without needing an
  accompanying number.
- **A named customer/case-study example with a described concrete outcome** counts as 2
  even if the number belongs to the customer, not the reporting firm (e.g. "Wiley is
  resolving cases 40% faster with Agentforce").

**Two judgment calls flagged as still open, not fully settled by the rules above** --
decide explicitly before holdout labeling if you want holdout scored differently:

1. Should a quantified outcome with NO named tool really score the same "2" as a
   named-tool-plus-number case, or does it deserve a distinct treatment? Applied as
   equivalent in the training set.
2. Should investment/spend figures (capex/opex on AI, not revenue or efficiency outcomes)
   count as a "quantified outcome" for 2, or should pure spend stay at 1 since it's an
   input, not a result? Applied leaning toward 2 when the dollar figure was specific and
   tied to a named initiative -- but this arguably conflates spend with outcome and is the
   single most defensible place to tighten the rule if holdout agreement is weak here.

## Process notes

- Label the `annotation_train.csv` sample first — this trains the classifier.
- Label `annotation_holdout.csv` independently, ideally in a separate sitting, without looking
  back at your train-set labels — this is what generalization gets checked against.
- Keep `ticker` and `quarter` attached throughout (already done for you) so you can later check
  whether the classifier's errors cluster in particular firms, sectors, or time periods.
