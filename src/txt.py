#client = Client(api_key=os.getenv("GEMINI_API_KEY"))
#GEMINI_MODEL = "gemini-2.5-flash"

def call_llm_batch(batch_inputs):
    """
    batch_inputs: list of dicts:
        {"section_index": int, "sentences": [str, ...]}

    Returns: dict mapping section_index -> result dict
    """
    if not batch_inputs:
        return {}

    # Build the sections block
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
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

        text = response.text

        # ---- safe guard against None / invalid ----
        if not text or not isinstance(text, str):
            return {}

        try:
            parsed = json.loads(text)
        except Exception as e:
            print("LLM JSON parse error:", type(e).__name__, e)
            # if JSON can't be parsed, give up for this batch
            return {}

        # We expect an array of objects
        if not isinstance(parsed, list):
            parsed = [parsed]

        out = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("section_index")
            if idx is None:
                continue

            # ensure keys exist
            res = {
                "roles": item.get("roles") or [],
                "obligations": item.get("obligations") or [],
                "powers": item.get("powers") or [],
                "penalties": item.get("penalties") or [],
                "rights": item.get("rights") or [],
            }
            out[idx] = res

        return out

    except Exception as e:
        print("LLM batch error:", type(e).__name__, e)
        return {}

