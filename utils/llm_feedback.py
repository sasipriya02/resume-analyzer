import os

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

PROMPT = """You are an experienced technical recruiter.
Compare the RESUME with the JOB DESCRIPTION and reply in markdown with:
1. A 2-line overall fit summary.
2. Five specific, actionable resume improvements for this job.
3. Three project ideas that would close the biggest skill gaps.
Be concise and concrete. Do not invent experience the candidate does not have.

Missing skills detected: {missing}

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""


def get_feedback(resume: str, jd: str, missing: list[str], api_key: str | None = None) -> str:
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return rule_based_feedback(missing)
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=MODEL,
            max_tokens=900,
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    missing=", ".join(missing) or "none",
                    resume=resume[:6000],
                    jd=jd[:4000],
                ),
            }],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    except Exception as e:  # network/key problems should not crash the app
        return f"LLM call failed ({e}).\n\n" + rule_based_feedback(missing)


def rule_based_feedback(missing: list[str]) -> str:
    if not missing:
        return "Your resume covers all the skills found in the job description. Quantify your achievements to stand out."
    lines = ["**No API key set, so here is basic feedback:**", ""]
    lines.append("- Add these missing skills if you genuinely have them: " + ", ".join(missing))
    lines.append("- Build one small project per missing skill and link the GitHub repo.")
    lines.append("- Use numbers in bullet points (e.g. 'reduced load time by 30%').")
    lines.append("- Mirror the job description's keywords in your summary and skills section.")
    lines.append("- Keep the resume to one page with clear sections.")
    return "\n".join(lines)
