#!/usr/bin/env python3
"""
AttnSentinel - Automated Attention Benchmark & Leaderboard Daemon
Runs from outside the target repository to prevent git conflicts.
"""

import os
import csv
import time
import subprocess
import logging
import argparse
from datetime import datetime

# ================= Configuration =================
# Define exactly where the repo is relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(SCRIPT_DIR, "attention-test")  # <--- CHANGE THIS IF YOUR REPO FOLDER NAME IS DIFFERENT

# Logs and CSV will be saved safely in the parent directory, away from Git operations
CSV_FILENAME = os.path.join(SCRIPT_DIR, "attention_leaderboard.csv")
LOG_FILENAME = os.path.join(SCRIPT_DIR, "attn_sentinel.log")

POLL_INTERVAL_SECONDS = 60
BENCH_CMD = (
    "python3 bench_attention.py --gpu {gpu} --shapes 1x64x8x131072x128 "
    "--dtype bf16 --causal --prefill-warmup 10 --prefill-iters 10 "
    "--decode-warmup 100 --decode-iters 100"
)

# Fallback baselines
DEFAULT_BASELINE_PREFILL = 3956.011
DEFAULT_BASELINE_DECODE = 5.358
# =================================================

logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_shell(cmd, split_lines=False, cwd=REPO_DIR):
    """Executes a shell command inside the target repo directory and returns combined output."""
    try:
        res = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, cwd=cwd
        )
        output = res.stdout.strip()
        return output.split('\n') if split_lines else output
    except Exception as e:
        error_msg = f"Exception running command '{cmd}': {str(e)}"
        logging.error(error_msg)
        return [] if split_lines else error_msg

def parse_section(text, section_marker):
    """Extracts operator execution times robustly, reading right-to-left to ignore spaces in names."""
    if section_marker not in text:
        return {}
    
    section_text = text.split(section_marker)[1]
    latencies = {}
    in_table = False
    
    for line in section_text.split('\n'):
        if '耗时(ms)' in line:
            in_table = True
            continue
        if in_table and line.startswith('---'):
            continue
        if in_table and line.startswith('==='):
            break
            
        if in_table and line.strip():
            parts = line.strip().split()
            # A valid row has the operator name + 6 trailing metric columns
            if len(parts) >= 7:
                try:
                    # The latency is ALWAYS the 6th token from the right
                    latency = float(parts[-6])
                    op_name = " ".join(parts[:-6])
                    latencies[op_name] = latency
                except ValueError:
                    pass
    return latencies

def get_latencies(latencies_dict):
    customs = [v for k, v in latencies_dict.items() if k != 'baseline']
    custom = min(customs) if customs else None
    return custom

def main():
    parser = argparse.ArgumentParser(description="AttnSentinel Daemon")
    parser.add_argument("--branch", type=str, help="Specify a single branch to run (e.g., origin/my-branch or my-branch)")
    parser.add_argument("--gpu", type=str, default="0", help="Specify GPU ID to use (default: 0)")
    args = parser.parse_args()

    if not os.path.exists(REPO_DIR):
        print(f"❌ ERROR: Repository directory not found at {REPO_DIR}")
        print("Please check the REPO_DIR path in the configuration.")
        return

    mode_msg = f"targeting specific branch: {args.branch}" if args.branch else "in continuous daemon mode"
    start_msg = f"🛡️ Starting AttnSentinel ({mode_msg}) targeting repo: {REPO_DIR} on GPU: {args.gpu}"
    print(start_msg)
    logging.info(start_msg)
    
    results = {}
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[row['Branch']] = row
                
    while True:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{now_str}] Fetching latest remote branches...")
        
        # Fetch directly inside REPO_DIR
        run_shell("git fetch --all")

        if args.branch:
            # Ensure branch starts with 'origin/'
            target_branch = args.branch if args.branch.startswith('origin/') else f"origin/{args.branch}"
            branches_output = [target_branch]
        else:
            branches_output = run_shell("git branch -r", split_lines=True)
        
        for line in branches_output:
            line = line.strip()
            if not line or '->' in line: 
                continue
                
            branch = line
            person_name = branch.replace('origin/', '')
            commit_hash = run_shell(f"git rev-parse {branch}").strip()

            # Handle edge case where specified branch doesn't exist remotely
            if "fatal:" in commit_hash.lower():
                err_msg = f"Could not resolve branch '{branch}'. Ensure the branch exists."
                print(f"❌ ERROR: {err_msg}")
                logging.error(err_msg)
                continue
            
            # Skip if already benchmarked (unless a specific branch was passed, then we force run)
            already_benchmarked = branch in results and results[branch].get('Commit') == commit_hash
            if already_benchmarked and not args.branch:
                continue
            elif already_benchmarked and args.branch:
                print(f"ℹ️ Note: Branch {branch} already benchmarked at this commit, forcing rerun.")
            
            detect_msg = f"Commit detected ({commit_hash[:7]}) on branch: {branch}"
            print(f"🚀 {detect_msg}")
            logging.info(detect_msg)
            
            # Scorched Earth Git Reset: Ensures total cleanliness before checking out
            run_shell("git reset --hard")
            run_shell("git clean -fd") 
            
            checkout_out = run_shell(f"git checkout {branch}")
            
            if "error:" in checkout_out.lower() or "fatal:" in checkout_out.lower():
                logging.error(f"Failed to checkout branch {branch}. Reason:\n{checkout_out}")
                continue
            
            print(f"   Running benchmark and extracting execution times...")
            bench_output = run_shell(BENCH_CMD.format(gpu=args.gpu))
            
            prefill_lats = parse_section(bench_output, "[PREFILL]")
            decode_lats  = parse_section(bench_output, "[DECODE]")
            
            # Debug prints to verify the right-to-left parser is working
            print(f"   [Debug] Parsed Prefill: {prefill_lats}")
            print(f"   [Debug] Parsed Decode:  {decode_lats}")
            
            if not prefill_lats and not decode_lats:
                fail_msg = f"Benchmark failed/crashed on {branch} (Commit: {commit_hash}). Assigning Score = 0."
                print(f"   [!] {fail_msg} Check log file for traceback.")
                
                logging.error(f"{fail_msg}\n--- RAW STDOUT ---\n{bench_output}\n--- END STDOUT ---")
                
                score, A, B, c_pref, c_dec = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                c_pref, b_pref = get_latencies(prefill_lats), DEFAULT_BASELINE_PREFILL
                c_dec, b_dec  = get_latencies(decode_lats), DEFAULT_BASELINE_DECODE
                
                A = (b_pref / c_pref) if (c_pref is not None and c_pref > 0) else 0.0
                B = (b_dec / c_dec) if (c_dec is not None and c_dec > 0) else 0.0
                score = (A / 2) + (B / 35)
                
                # Format for display/logs, avoiding NoneType errors
                disp_pref = f"{c_pref:.3f}" if c_pref else "FAIL"
                disp_dec = f"{c_dec:.3f}" if c_dec else "FAIL"
                
                success_msg = f"Done! Prefill: {disp_pref} ms | Decode: {disp_dec} ms | Score: {score:.4f}"
                print(f"   ✅ {success_msg}")
                logging.info(f"Branch: {branch} | Prefill: {disp_pref} ms (Speedup: {A:.2f}x) | Decode: {disp_dec} ms (Speedup: {B:.2f}x) | Score: {score:.4f}")
                
            results[branch] = {
                'Branch': branch,
                'Person': person_name,
                'Score': round(score, 6),
                'Prefill Latency (ms)': round(c_pref, 4) if c_pref else 0.0,
                'Decode Latency (ms)': round(c_dec, 4) if c_dec else 0.0,
                'A (Prefill Speedup)': round(A, 4),
                'B (Decode Speedup)': round(B, 4),
                'Commit': commit_hash,
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            sorted_rows = sorted(results.values(), key=lambda x: float(x.get('Score', 0)), reverse=True)
            fields = ['Branch', 'Person', 'Score', 'Prefill Latency (ms)', 'Decode Latency (ms)', 
                      'A (Prefill Speedup)', 'B (Decode Speedup)', 'Commit', 'Last Updated']
                      
            with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(sorted_rows)

        # Break out of loop if running a single, specific branch
        if args.branch:
            print(f"🎯 Execution complete for specified branch '{target_branch}'. Exiting.")
            break

        print(f"💤 Sweep complete. Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == '__main__':
    main()