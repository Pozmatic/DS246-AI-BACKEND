"""from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3-32b"""

def call_llm_batch(batch_inputs):
    """
    batch_inputs: list of dicts of the form:
        { "section_index": int, "sentences": [str, ...] }

    Returns: dict mapping section_index -> result dict (or {} on any parsing failure)
    """

    if not batch_inputs:
        return {}

    # Build the numbered SECTIONS block
    blocks = []
    for entry in batch_inputs:
        idx = entry["section_index"]
        sent_list = entry["sentences"]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sent_list))
        block = f"SECTION {idx}:\n{numbered}"
        blocks.append(block)

    sections_block = "\n\n".join(blocks)

    prompt = PROMPT_TEMPLATE_BATCH.format(sections_block=sections_block)

    try:
        # Groq Chat Completion request
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a structured legal information extractor. Respond ONLY using strict JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"},  # forces JSON object response
        )

        # Extract model output
        text = response.choices[0].message.content

        # Fail-safe: empty or None output → skip
        if not text or not isinstance(text, str):
            return {}

        # Parse JSON safely
        try:
            parsed = json.loads(text)
        except Exception as e:
            print("LLM JSON parse error:", type(e).__name__, e)
            return {}

        # expected: list of objects, each with section_index + semantic data
        if not isinstance(parsed, list):
            parsed = [parsed]

        out = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("section_index")
            if idx is None:
                continue

            out[idx] = {
                "roles": item.get("roles") or [],
                "obligations": item.get("obligations") or [],
                "powers": item.get("powers") or [],
                "penalties": item.get("penalties") or [],
                "rights": item.get("rights") or [],
            }

        return out

    except Exception as e:
        print("LLM batch error:", type(e).__name__, e)
        return {}