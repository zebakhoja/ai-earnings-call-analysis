"""
Curated dictionary of AI-related terms for matching against earnings call transcripts.

Structure is driven by the thesis's theoretical framework (see Dissertation Outline,
section 5 "Categorising AI technologies" and section 5.3 signalling discussion):

1. TAXONOMY (`category`): each term is tagged with the AI technology type used in the
   J-curve / technology-taxonomy analysis:
     - "general_ai"                : umbrella AI language that doesn't indicate a
                                      specific technology type (e.g. "AI strategy").
                                      Used for AI-flagging/volume measures, but
                                      EXCLUDED from the infra->generative taxonomy
                                      transition analysis since it carries no type
                                      information.
     - "infrastructure_ai"         : cloud compute / data platforms -- foundational,
                                      early-stage, often shared across firms.
     - "predictive_analytical_ai"  : ML / forecasting -- mature, measurable outcomes.
     - "generative_ai"             : LLMs, image generation, coding assistants --
                                      post-2022, rapid diffusion.
     - "autonomous_ai"             : robotics, agentic systems -- emerging.

2. SPECIFICITY (`specificity`): a signalling-theory tag (Spence 1973; Crawford & Sobel
   1982; "AI Washing" ScienceDirect 2026) distinguishing:
     - "generic"  : cheap-to-produce, unverifiable buzzwords (cheap-talk risk --
                    a firm can say "AI-powered" without knowing anything about AI).
     - "specific" : named tools, models, or technical methods that are costly to
                    fabricate credibly (Spence's separating equilibrium) -- these are
                    the terms that feed a "specificity ratio" as an early,
                    pre-topic-model proxy for the substantiveness measure / H3.

Matching conventions (unchanged, enforced by the extraction script, not here):
- All terms matched case-insensitively EXCEPT those in `case_sensitive_terms`
  (short acronyms like "AI", "ML", "LLM" are ambiguous in lowercase).
- Terms matched on word boundaries (\\b) by the extraction script.
- Hyphens are normalized by the extraction script (matches "AI-powered" and "AI powered").
"""

GENERAL_AI = "general_ai"
INFRASTRUCTURE_AI = "infrastructure_ai"
PREDICTIVE_ANALYTICAL_AI = "predictive_analytical_ai"
GENERATIVE_AI = "generative_ai"
AUTONOMOUS_AI = "autonomous_ai"

GENERIC = "generic"
SPECIFIC = "specific"

AI_TERMS = [
    ("artificial intelligence", GENERAL_AI, GENERIC),
    ("AI", GENERAL_AI, GENERIC),
    ("AI-powered", GENERAL_AI, GENERIC),
    ("AI-driven", GENERAL_AI, GENERIC),
    ("AI-enabled", GENERAL_AI, GENERIC),
    ("AI capabilities", GENERAL_AI, GENERIC),
    ("AI technology", GENERAL_AI, GENERIC),
    ("AI technologies", GENERAL_AI, GENERIC),
    ("AI solution", GENERAL_AI, GENERIC),
    ("AI solutions", GENERAL_AI, GENERIC),
    ("AI product", GENERAL_AI, GENERIC),
    ("AI products", GENERAL_AI, GENERIC),
    ("AI feature", GENERAL_AI, GENERIC),
    ("AI features", GENERAL_AI, GENERIC),
    ("AI system", GENERAL_AI, GENERIC),
    ("AI systems", GENERAL_AI, GENERIC),
    ("AI application", GENERAL_AI, GENERIC),
    ("AI applications", GENERAL_AI, GENERIC),
    ("AI investment", GENERAL_AI, GENERIC),
    ("AI investments", GENERAL_AI, GENERIC),
    ("AI strategy", GENERAL_AI, GENERIC),
    ("AI adoption", GENERAL_AI, GENERIC),
    ("AI model", GENERAL_AI, SPECIFIC),
    ("AI models", GENERAL_AI, SPECIFIC),
    ("AI tool", GENERAL_AI, SPECIFIC),
    ("AI tools", GENERAL_AI, SPECIFIC),

    ("data center", INFRASTRUCTURE_AI, GENERIC),
    ("data centers", INFRASTRUCTURE_AI, GENERIC),
    ("data centre", INFRASTRUCTURE_AI, GENERIC),
    ("data centres", INFRASTRUCTURE_AI, GENERIC),
    ("hybrid cloud", INFRASTRUCTURE_AI, GENERIC),
    ("cloud infrastructure", INFRASTRUCTURE_AI, GENERIC),
    ("compute infrastructure", INFRASTRUCTURE_AI, GENERIC),
    ("AI infrastructure", INFRASTRUCTURE_AI, GENERIC),
    ("data platform", INFRASTRUCTURE_AI, GENERIC),
    ("data platforms", INFRASTRUCTURE_AI, GENERIC),
    ("GPU", INFRASTRUCTURE_AI, SPECIFIC),
    ("GPUs", INFRASTRUCTURE_AI, SPECIFIC),
    ("Nvidia", INFRASTRUCTURE_AI, SPECIFIC),

    ("machine learning", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("ML", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("deep learning", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("neural network", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("neural networks", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("reinforcement learning", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("supervised learning", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("unsupervised learning", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("predictive analytics", PREDICTIVE_ANALYTICAL_AI, GENERIC),
    ("predictive modeling", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("predictive model", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("natural language processing", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("NLP", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("computer vision", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("image recognition", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("speech recognition", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("voice recognition", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),
    ("facial recognition", PREDICTIVE_ANALYTICAL_AI, SPECIFIC),

    ("generative AI", GENERATIVE_AI, GENERIC),
    ("genAI", GENERATIVE_AI, GENERIC),
    ("large language model", GENERATIVE_AI, SPECIFIC),
    ("large language models", GENERATIVE_AI, SPECIFIC),
    ("LLM", GENERATIVE_AI, SPECIFIC),
    ("LLMs", GENERATIVE_AI, SPECIFIC),
    ("foundation model", GENERATIVE_AI, SPECIFIC),
    ("foundation models", GENERATIVE_AI, SPECIFIC),
    ("transformer model", GENERATIVE_AI, SPECIFIC),
    ("transformer models", GENERATIVE_AI, SPECIFIC),
    ("diffusion model", GENERATIVE_AI, SPECIFIC),
    ("diffusion models", GENERATIVE_AI, SPECIFIC),
    ("multimodal AI", GENERATIVE_AI, SPECIFIC),
    ("multimodal model", GENERATIVE_AI, SPECIFIC),
    ("reasoning model", GENERATIVE_AI, SPECIFIC),
    ("prompt engineering", GENERATIVE_AI, SPECIFIC),
    ("fine-tuning", GENERATIVE_AI, SPECIFIC),
    ("chatbot", GENERATIVE_AI, GENERIC),
    ("chatbots", GENERATIVE_AI, GENERIC),
    ("conversational AI", GENERATIVE_AI, GENERIC),
    ("ChatGPT", GENERATIVE_AI, SPECIFIC),
    ("GPT", GENERATIVE_AI, SPECIFIC),
    ("GPT-4", GENERATIVE_AI, SPECIFIC),
    ("Copilot", GENERATIVE_AI, SPECIFIC),
    ("Microsoft Copilot", GENERATIVE_AI, SPECIFIC),
    ("Gemini", GENERATIVE_AI, SPECIFIC),
    ("Google Gemini", GENERATIVE_AI, SPECIFIC),
    ("Bard", GENERATIVE_AI, SPECIFIC),
    ("Watson", GENERATIVE_AI, SPECIFIC),
    ("IBM Watson", GENERATIVE_AI, SPECIFIC),
    ("Bedrock", GENERATIVE_AI, SPECIFIC),
    ("AWS Bedrock", GENERATIVE_AI, SPECIFIC),
    ("Azure OpenAI", GENERATIVE_AI, SPECIFIC),
    ("OpenAI", GENERATIVE_AI, SPECIFIC),
    ("Anthropic", GENERATIVE_AI, SPECIFIC),
    ("Claude", GENERATIVE_AI, SPECIFIC),
    ("DeepSeek", GENERATIVE_AI, SPECIFIC),
    ("Llama", GENERATIVE_AI, SPECIFIC),

    ("AI agent", AUTONOMOUS_AI, GENERIC),
    ("AI agents", AUTONOMOUS_AI, GENERIC),
    ("agentic AI", AUTONOMOUS_AI, GENERIC),
    ("autonomous agent", AUTONOMOUS_AI, SPECIFIC),
    ("autonomous agents", AUTONOMOUS_AI, SPECIFIC),
    ("autonomous system", AUTONOMOUS_AI, SPECIFIC),
    ("autonomous systems", AUTONOMOUS_AI, SPECIFIC),
    ("robotic process automation", AUTONOMOUS_AI, SPECIFIC),
    ("intelligent automation", AUTONOMOUS_AI, GENERIC),
    ("cognitive automation", AUTONOMOUS_AI, GENERIC),
]

CASE_SENSITIVE_TERMS = {
    "AI", "ML", "LLM", "LLMs", "NLP", "GPT", "GPT-4", "Claude",
    "Bard", "Watson", "Copilot", "Gemini", "Bedrock", "Llama", "GPU", "GPUs",
}
