from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


class ProfileUpdate(BaseModel):
    """Fields to be merged into the long-term user_profile."""

    updated_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value pairs representing stable user traits, preferences, "
            "constraints, or background details inferred from the conversation."
        ),
    )


class ProfileUpdater:
    """Agent responsible for maintaining long-term user profile memory."""

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are the Long-Term Profile Updater for Remiro AI.
You maintain a structured, long-lived `user_profile` about the user.

Your job is to:
- Read the recent conversation.
- Compare it with the existing profile.
- Decide what *stable* information should be added or updated.

ONLY store information that is likely to stay true over time, such as:
- Personality traits (e.g., introvert, highly conscientious).
- Stable preferences (e.g., prefers remote work, values stability over risk).
- Background facts (e.g., years of experience, current role, location).
- Hard constraints (e.g., must earn at least X per month, cannot relocate).

DO NOT store:
- One-off emotions ("I'm tired today").
- Temporary states ("I'm stressed this week").
- Very vague or speculative guesses.

You will output ONLY the fields that should be added or updated, not the full profile.
If there is nothing new or important to add, return an empty `updated_profile`.
"""

    def get_chain(self):
        structured_llm = self.llm.with_structured_output(ProfileUpdate)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Current profile: {current_profile}\n\nRecent conversation:\n{conversation_text}"),
            ]
        )
        return prompt | structured_llm
