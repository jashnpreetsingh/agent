"""System prompts and few-shot examples, one module per agent role.

Prompts live apart from control flow so they can be reviewed, diffed, and
tuned without touching graph logic - and so a reader can see exactly what the
model was told.
"""

from src.prompts.critic import CRITIC_SYSTEM, build_critic_prompt
from src.prompts.planner import PLANNER_SYSTEM, build_planner_prompt
from src.prompts.researcher import RESEARCHER_SYSTEM, build_researcher_prompt
from src.prompts.synthesizer import SYNTHESIZER_SYSTEM, build_synthesis_prompt

__all__ = [
    "CRITIC_SYSTEM",
    "PLANNER_SYSTEM",
    "RESEARCHER_SYSTEM",
    "SYNTHESIZER_SYSTEM",
    "build_critic_prompt",
    "build_planner_prompt",
    "build_researcher_prompt",
    "build_synthesis_prompt",
]

#: Prepended to every role. Keeping the shared identity in one place stops the
#: four prompts from drifting into four different personas.
SHARED_PREAMBLE = """You are part of a biomedical evidence-retrieval system that answers \
questions using published literature from PubMed.

Non-negotiable rules for every role in this system:
- Published evidence is the only source of truth. Never rely on unstated background knowledge.
- Never invent a PMID, author, journal, statistic, or finding. If it is not in the retrieved \
records, it does not exist for your purposes.
- You inform; you do not treat. Never give personalised medical advice, dosing, or diagnosis.
- Uncertainty is information. Say plainly when evidence is thin, mixed, or absent."""
