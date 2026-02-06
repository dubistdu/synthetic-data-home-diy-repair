# Home DIY Repair Q&A Synthetic Data Generator

This project generates and validates synthetic Q&A pairs for home DIY repair scenarios using OpenAI's API with structured validation.

## Project Structure

```
├── main.py                         # Pipeline entry (run: python main.py)
├── diy_repair/                     # Pipeline package
│   ├── __init__.py
│   ├── config.py                   # Output filenames, TARGET_FAILURE_RATE, paths
│   ├── models.py                   # Pydantic data models
│   ├── openai_client.py            # OpenAI client and env config
│   ├── generation_phase.py         # Phase 1: LLM generation
│   ├── validation_phase.py         # Phase 2: Pydantic + rule-based validation
│   ├── failure_labeling.py         # Phase 3: LLM-as-judge; failure modes
│   ├── failure_analysis.py         # Phase 4: Stats, heatmaps, target rate
│   └── correction_phase.py         # Correct failed Q&A using judge criteria
├── scripts/                        # Optional, run-by-hand tools
│   ├── human_labeling.py           # CLI to label samples
│   ├── human_vs_llm_comparison.py  # Compare human vs LLM labels
│   └── merge_corrected_qa.py       # Merge corrected Q&A for re-labeling
├── docs/
│   └── Project Plan.md             # Original project plan
├── output/                         # Default output dir (created on first run, in .gitignore)
│   ├── generation_results.json
│   ├── structurally_valid_qa_pairs.json
│   ├── validation_summary.json
│   ├── failure_labeled_data.csv
│   ├── failure_labeled_data.json
│   ├── failure_analysis_report.json
│   ├── failure_heatmap.png
│   ├── failure_rates.png
│   ├── failure_correlations.png
│   ├── corrected_qa_pairs.json
│   ├── qa_after_correction.json    # From scripts/merge_corrected_qa.py
│   ├── human_labels.json
│   └── human_vs_llm_comparison.json
├── requirements.txt
└── README.md
```

### Output directory

All pipeline outputs (and script outputs) go into **`output/`** by default so the project root stays clean. Override with `--output-dir` (e.g. `python main.py --output-dir ./runs/exp1`). `python main.py stats` reads from `output/` unless you pass a path: `python main.py stats ./other_dir`.

### Code layout

| Path | Purpose |
|------|--------|
| `main.py` | Entry point. Full pipeline or single phases (`--generation-only`, etc.); `python main.py stats` for quick counts. |
| `diy_repair/config.py` | `FILENAMES`, `TARGET_FAILURE_RATE`, `DEFAULT_OUTPUT_DIR` (= `output/`). |
| `diy_repair/models.py` | Pydantic models for Q&A and generation results. |
| `diy_repair/openai_client.py` | OpenAI client and API key from env. |
| `diy_repair/generation_phase.py` | Phase 1: LLM generation with templates. |
| `diy_repair/validation_phase.py` | Phase 2: Pydantic + rule-based validation. |
| `diy_repair/failure_labeling.py` | Phase 3: LLM-as-judge; exports `FAILURE_MODE_NAMES` / `FAILURE_MODES_CLI`. |
| `diy_repair/failure_analysis.py` | Phase 4: Stats, heatmaps, target failure rate. |
| `diy_repair/correction_phase.py` | Corrects failed Q&A using judge criteria. |
| `scripts/human_labeling.py` | CLI to label samples. Run from root: `python scripts/human_labeling.py`. |
| `scripts/human_vs_llm_comparison.py` | Compare human vs LLM labels. Run from root: `python scripts/human_vs_llm_comparison.py`. |
| `docs/Project Plan.md` | Original project plan. |

## Features

### Generation Phase
- **5 Diverse Prompt Templates**: Covers appliance repair, plumbing, electrical, HVAC, and general home repair
- **Structured JSON Output**: Enforces consistent format with required fields
- **Error Handling**: Robust error handling with detailed logging
- **Rate Limiting**: Built-in delays to avoid API rate limits

### Validation Phase
- **Pydantic Schema Validation**: Ensures structural correctness
- **Quality Validation Rules**: 6 comprehensive validation rules:
  - Question quality (proper question format, specificity)
  - Answer completeness (step-by-step structure, explanations)
  - Tools realism (realistic and appropriate tools)
  - Steps clarity (actionable instructions, proper sequencing)
  - Safety adequacy (proper warnings and precautions)
  - Tips usefulness (additional practical value)
- **Filtering**: Removes invalid entries before analysis
- **Detailed Reporting**: Comprehensive validation summaries

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your OpenAI API key is configured in `openai_client.py`

## Model Selection

The system supports different OpenAI models:

- **gpt-3.5-turbo** (default): Faster, cheaper, higher rate limits
- **gpt-4**: Higher quality but more expensive and lower rate limits

If you encounter rate limit errors (429), try:
1. Using `gpt-3.5-turbo` instead of `gpt-4`
2. Reducing the number of samples
3. Adding longer delays between requests
4. Upgrading to a paid API plan

## Usage

### Basic Usage
Generate 20 Q&A pairs and validate them:
```bash
python main.py
```

### Advanced Usage
```bash
# Generate 50 samples
python main.py --samples 50

# Use GPT-4 model (if you have access)
python main.py --model gpt-4 --samples 20

# Use GPT-3.5-turbo (default, higher rate limits)
python main.py --model gpt-3.5-turbo --samples 50

# Run only generation phase
python main.py --generation-only --samples 30

# Run only validation phase (requires existing generation_results.json)
python main.py --validation-only

# Use a different output directory (default is output/)
python main.py --output-dir ./results --samples 25

# Reproducible run (same template order each time; LLM output still varies)
python main.py --seed 42 --samples 20

# Quick stats from output/ (or pass a path: python main.py stats ./results)
python main.py stats
```

**Note:** Success rate (structural validation and failure-label rate) varies from run to run because each run generates new samples and the LLM is non-deterministic. The refactor did not change any validation or labeling logic. Use `--seed` to get the same *template* order across runs; for strict reproducibility you’d also need to fix the model’s randomness (e.g. temperature=0 in the API).

## Output Files

By default all of these are written under **`output/`**:

1. **`generation_results.json`**: Raw generation results with metadata
2. **`structurally_valid_qa_pairs.json`**: Validated Q&A pairs used for labeling/analysis
3. **`validation_summary.json`**: Validation statistics and common errors
4. **`failure_labeled_data.json`** / **`.csv`**: Per-sample failure-mode labels
5. **`failure_analysis_report.json`**: Summary and target failure rate
6. **`failure_heatmap.png`**, **`failure_rates.png`**, **`failure_correlations.png`**: Analysis charts
7. **`corrected_qa_pairs.json`**: Output of correction phase (when run)
8. **`human_labels.json`**, **`human_vs_llm_comparison.json`**: From scripts (when run)

## Data Structure

Each Q&A pair follows this structure:
```json
{
  "question": "How do I fix a leaky faucet?",
  "answer": "Detailed step-by-step answer...",
  "equipment_problem": "Kitchen faucet with dripping water",
  "tools_required": ["adjustable wrench", "screwdriver", "plumber's tape"],
  "steps": ["Turn off water supply", "Remove faucet handle", "..."],
  "safety_info": "Always turn off water supply before starting...",
  "tips": "Apply plumber's tape in clockwise direction..."
}
```

## Validation Rules

The validation phase applies these quality checks:

1. **Question Quality**: Proper question format, specificity, not vague
2. **Answer Completeness**: Step-by-step structure, adequate explanation
3. **Tools Realism**: Realistic tools, not overly complex or generic
4. **Steps Clarity**: Clear actionable instructions, proper sequencing
5. **Safety Adequacy**: Proper warnings and protective measures
6. **Tips Usefulness**: Additional practical value beyond basic instructions

## Next Steps (Phase 3 & 4)

After running generation and validation:

1. **Failure Labeling**: Create DataFrame with binary failure mode columns
2. **Analysis**: Generate heatmaps and identify failure patterns
3. **Correction**: Run correction phase for failed Q&A; optionally merge and re-label to measure improvement

## Improving success rate (judge + correction)

When the LLM judge reports e.g. 84% success and you want to push higher:

1. **Fix the failed pairs (correction)**  
   Run the correction phase, then merge corrected content back and re-label to get a new success rate:
   ```bash
   python main.py --correction-only
   python scripts/merge_corrected_qa.py
   python main.py --labeling-only --input-qa output/qa_after_correction.json
   ```
   The new `failure_labeled_data.json` reflects labels for the dataset where failed items were replaced by corrected Q&A.

2. **Tune the judge**  
   If the judge is too strict (false positives), adjust the evaluation prompts in `diy_repair/failure_labeling.py`. Use human labels and `python scripts/human_vs_llm_comparison.py` to see FP/FN per mode and loosen prompts where FP is high.

3. **Improve the generator**  
   Edit `diy_repair/generation_phase.py` (templates, system prompt) so fewer low-quality Q&As are generated in the first place.

## Example Output

```
HOME DIY REPAIR Q&A SYNTHETIC DATA GENERATOR
============================================================
Timestamp: 2024-01-15 14:30:22
Output directory: /path/to/output
Samples to generate: 20
============================================================

🔄 STARTING GENERATION PHASE
----------------------------------------
Generating sample 1/20 using template: appliance_repair
Generating sample 2/20 using template: plumbing_repair
...

Generation Phase Complete:
Total generated: 20
Valid samples: 18
Invalid samples: 2
Success rate: 90.0%

🔍 STARTING VALIDATION PHASE
----------------------------------------

Validation Phase Complete:
Total samples: 20
Valid samples: 16
Invalid samples: 4
Validation rate: 80.0%

📊 FINAL SUMMARY
============================================================
Total samples generated: 20
Valid samples after validation: 16
Overall success rate: 80.0%
```

## Error Handling

The system handles various error scenarios:
- API failures and rate limiting
- JSON parsing errors
- Pydantic validation failures
- File I/O errors
- Network connectivity issues

All errors are logged with detailed messages for debugging.
