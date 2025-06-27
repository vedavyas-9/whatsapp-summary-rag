# app/controller/analyst_controller.py

from langsmith import traceable
from app.service.langstream_service import run_traced_claude_task

@traceable(name="Analyst Agent")
def run_analyst_agent(structured_json: str) -> str:
    """
    Stage 2 Analyst Agent
    Analyzes structured JSON from law enforcement datasets (CDRs, IPDRs, FIRs, etc.)
    to detect patterns, anomalies, and correlations between entities.

    Parameters:
    - structured_json (str): Pre-parsed and summarized information extracted from multiple sources.

    Returns:
    - str: Bullet-point summary of insights for further narrative generation (Stage 3).
    """

    prompt = f"""
You are an Investigative Analyst AI working with law enforcement data from Call Detail Records (CDRs), Internet Protocol Detail Records (IPDRs), and First Information Reports (FIRs).

You are given the following structured data:

{structured_json}

Your responsibilities:
1. Identify and list anomalies, such as:
   - Self-calls (same number calling itself)
   - Repeated or high-frequency communication with suspects
   - Overlapping or unusually long session times in IPDR
2. Detect behavioral patterns, such as:
   - Frequent presence near specific cell towers
   - High mobile data usage within small time spans
   - Consistent timing of suspicious activity
3. Correlate entities across datasets:
   - Suspects or numbers appearing in multiple FIRs
   - Tower IDs appearing across different incidents
   - Commonly contacted numbers between suspects

---

🎯 Output Format:
Respond using clear, concise bullet points using these symbols:
- 📌 for entity insights
- 📶 for network behavior or tower-based observations
- 🔗 for correlations between suspects or FIRs
- ⚠️ for anomalies or red flags

Avoid repeating input. Just focus on insights.
"""

    return run_traced_claude_task(prompt, agent_name="Analyst Agent")
