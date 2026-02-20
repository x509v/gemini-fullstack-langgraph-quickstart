import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from langchain_groq import ChatGroq

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)

from agent.configuration import Configuration

from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)

from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
    search_dir_semantic_local,
)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_groq_llm(temperature: float):
    return ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_retries=2,
    )


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = get_groq_llm(
        temperature=1.0
    )
    
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    output = {"search_query": result.query}
    # If a local search directory was provided in overall state, pass it on
    if state.get("search_dir") is not None:
        output["search_dir"] = state.get("search_dir")
    return output


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the appropriate research node.

    This is used to spawn n number of research nodes, one for each search query.
    If `search_dir` is present, spawn `local_research` branches that search the filesystem.
    """
    # If a local directory is provided, run local filesystem search instead of web search
    if state.get("search_dir"):
        return [
            Send(
                "local_research",
                {"search_query": search_query, "id": int(idx), "dir": state["search_dir"]},
            )
            for idx, search_query in enumerate(state["search_query"])
        ]

    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]



def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    Perform web research using Gemini Google Search ONLY.
    Returns structured search results, not synthesized text.
    """

    configurable = Configuration.from_runnable_config(config)
    query = state["search_query"]

    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,  # gemini-2.0-flash
        contents=query,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )

    grounding = response.candidates[0].grounding_metadata
    if not grounding or not grounding.grounding_chunks:
        return {
            "sources_gathered": [],
            "search_query": [query],
            "web_research_result": [],
        }

    resolved_urls = resolve_urls(
        grounding.grounding_chunks,
        state["id"],
    )

    results = []
    sources = []

    for chunk in grounding.grounding_chunks:
        web = chunk.web
        if not web:
            continue

        url = resolved_urls.get(web.uri)
        if not url:
            continue

        snippet = (
            getattr(chunk, "text", None)
            or getattr(chunk, "content", None)
            or ""
        )

        results.append({
            "title": web.title,
            "snippet": snippet,
            "url": url,
            "confidence": getattr(chunk, "confidence", None),
        })

        sources.append(url)

    return {
        "sources_gathered": sources,
        "search_query": [query],
        # Structured, raw search results
        "web_research_result": results,
    }


def local_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Search a local directory of Markdown files for the given query using TF-IDF ranking.

    Uses `search_dir_tfidf` helper to return ranked results with `title`, `snippet`, `url`, and
    `confidence` (0..1). The shape of the returned state mirrors `web_research` so the rest
    of the graph is unchanged.
    """
    query = state["search_query"]
    dir_path = state.get("dir") or state.get("search_dir")

    if not dir_path:
        return {
            "sources_gathered": [],
            "search_query": [query],
            "web_research_result": [],
        }

    # Use embedding-based semantic search implemented locally with TF-IDF fallback
    ranked = search_dir_semantic_local(dir_path, query, top_k=5)

    results = []
    sources = []

    for r in ranked:
        results.append({
            "title": r["title"],
            "snippet": r["snippet"],
            "url": r["url"],
            "confidence": r["confidence"],
        })
        sources.append(r["url"])  # Keep as string for simplicity

    return {
        "sources_gathered": sources,
        "search_query": [query],
        "web_research_result": results,
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(
            f"{r['title']}\n{r['snippet']}\n{r['url']}"
            for r in state["web_research_result"]
        )
        ,
    )

    llm = get_groq_llm(
        temperature=1.0
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    out = {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }
    # propagate search_dir so downstream routing (evaluate_research) can continue local search
    if state.get("search_dir"):
        out["search_dir"] = state.get("search_dir")
    return out


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        # If a local search directory is present, continue with local_research branches
        if state.get("search_dir"):
            return [
                Send(
                    "local_research",
                    {
                        "search_query": follow_up_query,
                        "id": state["number_of_ran_queries"] + int(idx),
                        "dir": state.get("search_dir"),
                    },
                )
                for idx, follow_up_query in enumerate(state["follow_up_queries"])
            ]

        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]

def finalize_answer(state: OverallState, config: RunnableConfig):
    """
    Finalizes the research by producing a structured report and cleaning up sources.
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 1. Prepare the prompt
    current_date = get_current_date()
    # We convert results to strings to ensure the LLM can process them
    summaries_text = "\n---\n\n".join([str(r) for r in state["web_research_result"]])
    
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries=summaries_text,
    )

    llm = get_groq_llm(temperature=0)
    result = llm.invoke(formatted_prompt)
    content = result.content

    # 2. Deduplicate and Filter Sources
    # state["sources_gathered"] contains all links from all parallel branches.
    # We use a dictionary keyed by URL to remove duplicates.
    seen_urls = set()
    unique_sources = []
    
    for source in state.get("sources_gathered", []):
        # Handle both dict and string formats depending on how your state is stored
        url = source["value"] if isinstance(source, dict) else source
        
        if url not in seen_urls:
            # OPTIONAL: Only include the source if the LLM actually mentioned it/its placeholder
            # If you use "short_urls" (like [1], [2]), check if they exist in content.
            short_url = source.get("short_url") if isinstance(source, dict) else None
            
            if short_url and short_url in content:
                # Replace placeholder with formatted markdown link
                content = content.replace(short_url, f"[{source['title']}]({url})")
                unique_sources.append(source)
                seen_urls.add(url)
            elif not short_url:
                # If not using placeholders, just deduplicate the master list
                unique_sources.append(source)
                seen_urls.add(url)

    # 3. Build a clean "Sources" section
    if "## Sources" not in content and unique_sources:
        source_list = "\n".join([f"- [{s['title']}]({s['value']})" if isinstance(s, dict) else f"- {s}" for s in unique_sources])
        content += f"\n\n## Sources\n{source_list}"

    return {
        "messages": [AIMessage(content=content)],
        "sources_gathered": unique_sources, # Returns the clean list back to state
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("local_research", local_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
# The continue function will decide whether to spawn `web_research` or `local_research`
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research", "local_research"]
)
# Reflect on the web or local research
builder.add_edge("web_research", "reflection")
builder.add_edge("local_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "local_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")