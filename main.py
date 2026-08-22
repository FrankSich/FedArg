import os
import sys
import time
import threading
import subprocess
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "client"))

from client.data_utils import preprocess_all_data
from client.clean_app import clean_all_data
from client.client_app import (
    start_flower_client,
    get_global_outcome_classes,
    reset_experiment,
    set_current_run,
    save_experiment_summary,
    finalize,
    NUM_RUNS,
)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

    # =====================================================
    # PREPROCESS DATA (ONLY ONCE)
    # =====================================================
    print("Starting hospital data preprocessing...\n")
    preprocess_all_data()

    print("\nStarting cleaning and binning data...")
    clean_all_data()

    print("\nAll preprocessing and cleaning completed!\n")

    # =====================================================
    # GLOBAL CLASS MAPPING (ONLY ONCE)
    # =====================================================
    cleaned_dir = Path("data/cleaned")
    class_to_index = get_global_outcome_classes(cleaned_dir) # type: ignore

    for old_file in cleaned_dir.glob("*_mapped.csv"):
        old_file.unlink()

    cleaned_csvs = sorted(
        p for p in cleaned_dir.glob("*.csv")
        if not p.name.endswith("_mapped.csv")
    )

    # =====================================================
    # MULTIPLE FEDERATED LEARNING RUNS
    # =====================================================
    for run in range(1, NUM_RUNS + 1):

        print("\n" + "=" * 70)
        print(f"🚀 STARTING EXPERIMENT {run}/{NUM_RUNS}")
        print("=" * 70)

        # Reset global variables
        set_current_run(run)
        reset_experiment()

        # -------------------------------------------------
        # Start Flower Server
        # -------------------------------------------------
        print("🚀 Starting Federated Learning Server...")

        server = subprocess.Popen(
            [sys.executable, "server/server_app.py"]
        )

        # Give the server time to initialize
        time.sleep(5)

        # -------------------------------------------------
        # Start Clients
        # -------------------------------------------------
        print("\n🚀 Starting Federated Clients...\n")

        threads = []

        for csv in cleaned_csvs:
            print(f"  → Client: {csv.name}")

            t = threading.Thread(
                target=start_flower_client,
                args=(csv.as_posix(), class_to_index),
                daemon=True,
            )

            t.start()
            threads.append(t)

        print("\n🎉 All clients launched. Training is running...\n")

        # Wait for all clients
        for t in threads:
            t.join()

        print(f"\n✅ Experiment {run} completed.")

        # -------------------------------------------------
        # Save experiment outputs and plots
        # -------------------------------------------------
        finalize()

        # -------------------------------------------------
        # Stop Flower Server
        # -------------------------------------------------
        print("\nStopping Flower server...")

        server.terminate()

        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)

        time.sleep(3)

        print("✅ Server stopped.")

        # Small pause before next experiment
        if run < NUM_RUNS:
            print("\nWaiting before next experiment...\n")
            time.sleep(5)

    # =====================================================
    # SAVE SUMMARY OF ALL RUNS
    # =====================================================
    print("\n" + "=" * 70)
    print("Saving experiment summary...")
    print("=" * 70)

    save_experiment_summary("results")

    print("\n🎉 ALL EXPERIMENTS FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()