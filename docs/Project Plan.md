1. Generation Phase

    Use LMQL or DSPy to build 3–5 prompt templates that ask for home DIY Repair Q&A pairs.

    Generate 20 synthetic QA pairs.

    Use Instructor to ensure the output JSON follows a structure:

    {
      "question": "...",
      "answer": "...",
      "equiment-problem": "...",
      "tools-required": ["..."],
      "steps": ["..."],
      "safety_info": "...",
      "tips": "..."
    }

2. Validation Phase

    Use Pydantic or jsonschema to validate that outputs are structurally correct.

    Filter invalid entries before moving to error analysis.

3. Failure Labeling

    Create a Pandas DataFrame with:

        Trace ID (auto-assigned)

        All structured fields (e.g., answer, ...)

        Binary columns for each of the 6 failure modes (0 = success, 1 = failure)

    Manually label 20 entries OR use regex/heuristics to auto-label common failures.

4. Analysis

    Create a heatmap of failure modes across samples.

    Identify:

        Most common failure types

        Correlations (e.g., Overcomplicated Recipes ↔ Missing Equipment)

5. Stretch Goal

    Try generating "corrected" versions of failed examples using a second-pass prompt.

    Re-label and re-analyze if failure rates improve.


    Feel Free to enhance the existing project with libraries like Braintrust to log things, instructor libraries and additional libraries.
