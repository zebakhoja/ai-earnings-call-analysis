# ai-earnings-call-analysis
My dissertation project uses natural language processing to analyse how 852 firms discussed AI across 23,500 earnings calls, examining how the specificity of their claims changed over time and how it relates to R&amp;D intensity.

Let’s Talk AI
Earnings Call Discourse as AI Capability Signalling & Strategic Communication
When does AI talk signal capability?
I analysed roughly 23,500 earnings calls from 852 firms across 22 countries for my UCL MSc thesis in Data Science and Public Policy. I wanted to understand when companies’ AI claims might signal genuine capability and when they might reflect incentives to appear active in AI.

Findings
Generic AI talk and operationally specific claims appear to serve different purposes. After ChatGPT’s launch, AI mentions rose sharply without the expected shift towards more advanced AI categories. That pattern is consistent with cheap talk: firms can signal that they are paying attention to AI through broad claims that are difficult to verify. By contrast, firms with higher R&D intensity tended to describe AI in more operationally specific terms. That pattern is consistent with capability signalling: concrete claims give analysts something to probe, making those claims riskier for firms that cannot substantiate them.
Where firms made those claims also mattered. Prepared remarks were more operationally specific than Q&A, and the gap was wider among firms with higher R&D intensity, contrary to my hypothesis.

The study measures what firms say publicly. The findings support the signalling explanation without independently verifying firms’ AI capabilities or establishing what motivated any individual claim.

Methodology
I built an NLP pipeline on Python that used a curated dictionary to identify AI-related passages and group them by technology. I trained a supervised classifier to score each passage on a scale from a generic mention to an operationally specific description. I then aggregated the results by firm and quarter, linked them to accounting data (revenue, R&D expense), and used structural break tests and regression models to examine changes over time and differences across firms.

Code and data
The code guide documents the scripts, run order, and inputs and outputs. Reproducing the full analysis requires Financial Modeling Prep earnings-call transcripts, Compustat data via WRDS, and the mapping and annotation files described in the guide.

