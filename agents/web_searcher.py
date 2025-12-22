import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class WebSearcher:
    def __init__(self, llm):
        self.llm = llm
        # Configure Serper with an explicit API key so failures are easier to diagnose.
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            self.search = None
            self._config_error = (
                "[Web search unavailable] SERPER_API_KEY is not set in the environment. "
                "Please add SERPER_API_KEY to your .env (or OS env vars) "
                "with a valid key from https://serper.dev/."
            )
        else:
            self.search = GoogleSerperAPIWrapper(serper_api_key=api_key)
            self._config_error = None
        self.system_prompt = """You are the Web Searcher for the Remiro AI system.
    Your sole purpose is to fetch accurate, up-to-date information from the internet to support the other specialist agents.

    ### YOUR ROLE:
    - You do NOT give career advice.
    - You ONLY provide facts, data, links, and current market trends.

    ### INPUTS:
    You will receive a specific search query or a request for information.

    ### PROCESS:
    1. Analyze the request.
    2. Use your search tool to find the answer.
    3. Summarize the findings clearly, citing sources/links where possible.

    ### OUTPUT CONSTRAINTS (TO SAVE TOKENS):
    - Return a *very concise* summary: at most 8 bullet points and roughly 250 words.
    - Focus only on the 3–5 most relevant, high-quality sources.
    - Omit low-signal or redundant details.
    """

    def run(self, query: str, history):
        """Execute a web search and return a summarized result string.

        This implementation does NOT rely on OpenAI tool-calling. Instead it:
        1. Calls GoogleSerperAPIWrapper directly.
        2. Asks the LLM to summarize the raw results using the system prompt.
        """
        # If the wrapper could not be configured at init time, surface that clearly.
        if self.search is None:
            return self._config_error

        try:
            raw_results = self.search.run(query)
        except Exception as e:
            # Fallback: if Serper fails (invalid key, 403, quota, etc.),
            # return a clear message instead of crashing the whole graph.
            return (
                "[Web search unavailable] There was an error calling the "
                "external search service (e.g., SERPER_API_KEY missing/invalid "
                f"or quota exceeded). Technical details: {e}"
            )

        # Limit raw results length so the summarization prompt stays compact.
        raw_str = str(raw_results)
        if len(raw_str) > 3000:
            raw_str = raw_str[:3000] + "... (truncated)"

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "User query: {query}\n\nRaw search results (truncated if necessary):\n{raw_results}\n\n"
                "Summarize only the most relevant, reliable information for the other "
                "specialist agents. Do NOT give career advice yourself; just present "
                "facts, figures, links, and trends.",
            ),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "history": history,
            "query": query,
            "raw_results": raw_str,
        })
        return getattr(response, "content", str(response))
