"""
Main entry point for the Home DIY Repair Q&A synthetic data pipeline.
Orchestrates: generation → validation → failure labeling → analysis → correction.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from diy_repair.config import DEFAULT_OUTPUT_DIR, FILENAMES
from diy_repair.correction_phase import run_correction_phase
from diy_repair.failure_analysis import run_failure_analysis
from diy_repair.failure_labeling import create_failure_labeled_dataframe, load_structurally_valid_data
from diy_repair.generation_phase import run_generation_phase
from diy_repair.models import DIYRepairQA, GenerationResult
from diy_repair.validation_phase import run_validation_phase, save_valid_data


def main():
    """
    Main function to orchestrate the generation and validation phases
    """
    parser = argparse.ArgumentParser(description="Generate and validate Home DIY Repair Q&A synthetic data")
    parser.add_argument("--samples", type=int, default=20, help="Number of Q&A pairs to generate (default: 20)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--generation-only", action="store_true", help="Run only the generation phase")
    parser.add_argument("--validation-only", action="store_true", help="Run only the validation phase (requires existing generation_results.json)")
    parser.add_argument("--labeling-only", action="store_true", help="Run only the failure labeling phase (requires existing structurally_valid_qa_pairs.json)")
    parser.add_argument("--input-qa", type=str, default=None, help="Path to Q&A JSON for labeling (e.g. output/qa_after_correction.json); default: output/structurally_valid_qa_pairs.json")
    parser.add_argument("--analysis-only", action="store_true", help="Run only the analysis phase (requires existing failure_labeled_data.csv)")
    parser.add_argument("--correction-only", action="store_true", help="Run only the correction phase (requires existing failure_labeled_data.json)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for results")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation (template choice); same seed = same sequence of templates")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("HOME DIY REPAIR Q&A SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Samples to generate: {args.samples}")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    generation_results = None
    
    # Generation Phase
    if not args.validation_only:
        print("\n🔄 STARTING GENERATION PHASE")
        print("-" * 40)
        
        try:
            generation_results = run_generation_phase(args.samples, args.model, seed=args.seed)
            
            # Save generation results
            results_file = output_dir / FILENAMES["generation_results"]
            serializable_results = []
            for result in generation_results:
                result_dict = result.model_dump()
                if result_dict['qa_pair']:
                    result_dict['qa_pair'] = result.qa_pair.model_dump()
                serializable_results.append(result_dict)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Generation results saved to {results_file}")
            
        except Exception as e:
            print(f"❌ Error in generation phase: {str(e)}")
            sys.exit(1)
    
    # Stop here if generation-only mode
    if args.generation_only:
        print("\n✅ Generation phase completed. Exiting (generation-only mode).")
        return

    # Validation Phase (Structural validation only)
    print("\n🔍 STARTING VALIDATION PHASE (STRUCTURAL VALIDATION)")
    print("-" * 40)
    
    try:
        # Load generation results if not already in memory
        if generation_results is None:
            results_file = output_dir / FILENAMES["generation_results"]
            if not results_file.exists():
                print(f"❌ Generation results file not found: {results_file}")
                print("Please run the generation phase first or provide the correct path.")
                sys.exit(1)
            
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to GenerationResult objects
            generation_results = []
            for item in data:
                if item['qa_pair']:
                    item['qa_pair'] = DIYRepairQA(**item['qa_pair'])
                generation_results.append(GenerationResult(**item))
        
        # Run validation
        valid_results, validation_summary = run_validation_phase(generation_results)
        
        # Save structurally valid data
        valid_data_file = output_dir / FILENAMES["structurally_valid_qa"]
        save_valid_data(valid_results, str(valid_data_file))

        # Save validation summary
        summary_file = output_dir / FILENAMES["validation_summary"]
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(validation_summary.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Validation summary saved to {summary_file}")
        
    except Exception as e:
        print(f"❌ Error in validation phase: {str(e)}")
        sys.exit(1)

    # Stop here if validation-only mode
    if args.validation_only:
        print("\n✅ Validation phase completed. Exiting (validation-only mode).")
        return

    # Failure Labeling Phase (Phase 3)
    print("\n🏷️ STARTING FAILURE LABELING PHASE (LLM-AS-JUDGE)")
    print("-" * 40)

    try:
        # Load structurally valid data (or merged post-correction QA when --input-qa is set)
        qa_path = args.input_qa or str(output_dir / FILENAMES["structurally_valid_qa"])
        qa_data = load_structurally_valid_data(qa_path)

        if not qa_data:
            print("❌ No structurally valid data found. Cannot proceed with failure labeling.")
            sys.exit(1)

        # Create failure labeled DataFrame
        failure_df = create_failure_labeled_dataframe(qa_data, args.model)

        # Save failure labeled data
        failure_csv_file = output_dir / FILENAMES["failure_labeled_csv"]
        failure_json_file = output_dir / FILENAMES["failure_labeled_json"]

        failure_df.to_csv(failure_csv_file, index=False)
        failure_df.to_json(failure_json_file, orient="records", indent=2)

        print(f"✅ Failure labeled data saved to {failure_csv_file}")
        print(f"✅ Failure labeled data saved to {failure_json_file}")

    except Exception as e:
        print(f"❌ Error in failure labeling phase: {str(e)}")
        sys.exit(1)

    # Stop here if labeling-only mode
    if args.labeling_only:
        print("\n✅ Failure labeling phase completed. Exiting (labeling-only mode).")
        return

    # Correction-only mode: run just the correction phase and exit
    if args.correction_only:
        print("\n🔧 STARTING CORRECTION PHASE (correction-only mode)")
        print("-" * 40)
        try:
            run_correction_phase(
                model=args.model,
                input_file=str(output_dir / FILENAMES["failure_labeled_json"]),
                output_file=str(output_dir / FILENAMES["corrected_qa"]),
            )
        except FileNotFoundError as e:
            print(f"❌ {e}. Ensure {FILENAMES['failure_labeled_json']} exists in {output_dir}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error in correction phase: {str(e)}")
            sys.exit(1)
        print("\n✅ Correction phase completed. Exiting (correction-only mode).")
        return

    # Analysis Phase (Phase 4)
    print("\n📊 STARTING ANALYSIS PHASE")
    print("-" * 40)

    try:
        # Run failure analysis
        run_failure_analysis(str(failure_csv_file))

    except Exception as e:
        print(f"❌ Error in analysis phase: {str(e)}")
        sys.exit(1)

    # Correction Phase (Stretch: correct failed Q&A)
    print("\n🔧 STARTING CORRECTION PHASE (failed Q&A only)")
    print("-" * 40)
    correction_results = []
    try:
        correction_results = run_correction_phase(
            model=args.model,
            input_file=str(output_dir / FILENAMES["failure_labeled_json"]),
            output_file=str(output_dir / FILENAMES["corrected_qa"]),
        )
    except FileNotFoundError:
        print(f"⚠️ {FILENAMES['failure_labeled_json']} not found; skipping correction.")
    except Exception as e:
        print(f"❌ Error in correction phase: {str(e)}")
        sys.exit(1)

    # Final Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total samples generated: {len(generation_results)}")
    print(f"Structurally valid samples: {len(valid_results)}")
    print(f"Structural validation rate: {len(valid_results)/len(generation_results)*100:.1f}%")

    if 'failure_df' in locals():
        print(f"Failure labeled samples: {len(failure_df)}")
        print(f"Overall failure rate: {failure_df['overall_failure'].mean():.1%}")
        print(f"Overall success rate: {(1 - failure_df['overall_failure'].mean()):.1%}")
    if correction_results:
        valid_corrected = sum(1 for r in correction_results if r.get("is_valid"))
        print(f"Corrected Q&A pairs: {len(correction_results)} (valid: {valid_corrected})")

    if validation_summary.common_errors:
        print(f"\nMost common structural errors:")
        for i, error in enumerate(validation_summary.common_errors[:3], 1):
            print(f"  {i}. {error}")

    print(f"\n📁 Output files:")
    print(f"  • Generation results: {output_dir / FILENAMES['generation_results']}")
    print(f"  • Structurally valid Q&A pairs: {output_dir / FILENAMES['structurally_valid_qa']}")
    print(f"  • Validation summary: {output_dir / FILENAMES['validation_summary']}")
    if "failure_df" in locals():
        print(f"  • Failure labeled data (CSV): {output_dir / FILENAMES['failure_labeled_csv']}")
        print(f"  • Failure labeled data (JSON): {output_dir / FILENAMES['failure_labeled_json']}")
        print(f"  • Analysis visualizations: failure_heatmap.png, failure_rates.png, failure_correlations.png")
        print(f"  • Analysis report: {output_dir / FILENAMES['failure_analysis_report']}")
    if correction_results:
        print(f"  • Corrected Q&A pairs: {output_dir / FILENAMES['corrected_qa']}")

    print("\n✅ All phases completed successfully!")

    if 'failure_df' not in locals():
        print("\n💡 NEXT STEPS:")
        print("1. Run failure labeling: python main.py --labeling-only")
        print("2. Run analysis: python main.py --analysis-only")
        print("3. Review visualizations and reports")


def quick_stats(output_dir: str = None) -> None:
    """Print quick stats from existing result files in the given directory."""
    if output_dir is None:
        output_dir = str(DEFAULT_OUTPUT_DIR)
    out = Path(output_dir)
    try:
        gen_path = out / FILENAMES["generation_results"]
        with open(gen_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total = len(data)
        valid = sum(1 for item in data if item.get("is_valid"))
        print("Generation Results Summary:")
        print(f"  Total generated: {total}")
        print(f"  Initially valid: {valid}")
        print(f"  Initial success rate: {valid / total * 100:.1f}%")
        valid_path = out / FILENAMES["structurally_valid_qa"]
        if valid_path.exists():
            with open(valid_path, "r", encoding="utf-8") as f:
                valid_data = json.load(f)
            n = len(valid_data)
            print(f"  Final valid after validation: {n}")
            print(f"  Final success rate: {n / total * 100:.1f}%")
    except FileNotFoundError:
        print("No results files found. Run the generation phase first.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        quick_stats(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main()
