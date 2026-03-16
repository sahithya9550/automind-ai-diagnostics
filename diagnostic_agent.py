import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool


@tool
def estimate_repair_cost(issue: str) -> str:
    """Estimate repair cost for common vehicle issues."""
    cost_db = {
        "engine overheating": {
            "parts": "$80-$400",
            "labor": "$150-$400",
            "urgency": "HIGH",
        },
        "battery replacement": {
            "parts": "$150-$350",
            "labor": "$50-$100",
            "urgency": "MEDIUM",
        },
        "brake pad replacement": {
            "parts": "$80-$200",
            "labor": "$100-$200",
            "urgency": "HIGH",
        },
        "oil change": {
            "parts": "$30-$80",
            "labor": "$25-$50",
            "urgency": "LOW",
        },
        "transmission": {
            "parts": "$400-$2500",
            "labor": "$500-$1500",
            "urgency": "CRITICAL",
        },
    }

    for key, data in cost_db.items():
        if any(w in issue.lower() for w in key.split()):
            return (
                f"Issue: {key.title()}\n"
                f"Parts: {data['parts']}\n"
                f"Labor: {data['labor']}\n"
                f"Urgency: {data['urgency']}"
            )

    return f"Custom estimate needed for: {issue}"


@tool
def get_maintenance_schedule(mileage: str) -> str:
    """Get recommended maintenance tasks based on mileage."""
    miles = int(mileage.replace(",", ""))
    tasks = []

    if miles % 5000 < 500:
        tasks.append("Oil Change")
    if miles % 15000 < 500:
        tasks.append("Tire Rotation + Air Filter")
    if miles % 30000 < 500:
        tasks.append("Brake Inspection")
    if miles % 60000 < 500:
        tasks.append("Spark Plugs + Transmission Fluid")

    return f"At {miles} miles: {', '.join(tasks) if tasks else 'No scheduled maintenance due'}"


def build_diagnostic_agent():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to your .env file in the project root."
        )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key,
    )

    tools = [estimate_repair_cost, get_maintenance_schedule]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are AutoMind, an expert vehicle diagnostic AI. Given sensor anomaly data: "
                "provide root cause analysis, severity, repair recommendations, "
                "cost estimates, and failure timeline. Prioritize safety.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)