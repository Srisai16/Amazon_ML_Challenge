"""
Script to package final submission zip archive matching official hackathon structure:

Team_Technocrats_Submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── scripts/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import zipfile
import time

BASE_DIR = r"S:\Amazon_ML_Challenge"
ZIP_PATH = os.path.join(BASE_DIR, "Team_Technocrats_Submission.zip")

def package():
    print("Packaging submission zip archive...")
    t0 = time.time()

    output_files = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    ]

    doc_file = ("Documentation_template.md", "Documentation_template.md")

    code_files = []
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "src")):
        for file in files:
            if not file.endswith(".pyc") and "__pycache__" not in root:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, BASE_DIR)
                code_files.append((full_path, os.path.join("code", "business_entity_resolution", rel_path)))

    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "scripts")):
        for file in files:
            if not file.endswith(".pyc") and "__pycache__" not in root:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, BASE_DIR)
                code_files.append((full_path, os.path.join("code", "business_entity_resolution", rel_path)))

    for file in ["README.md", "requirements.txt"]:
        full_path = os.path.join(BASE_DIR, file)
        if os.path.exists(full_path):
            code_files.append((full_path, os.path.join("code", "business_entity_resolution", file)))

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        print("Adding output files...")
        for src, arcname in output_files:
            full_src = os.path.join(BASE_DIR, src)
            print(f"  Adding {src} ({os.path.getsize(full_src) / (1024**2):.1f} MB)")
            zf.write(full_src, arcname)

        print("Adding methodology document...")
        zf.write(os.path.join(BASE_DIR, doc_file[0]), doc_file[1])

        print("Adding source code & scripts...")
        for src, arcname in code_files:
            print(f"  Adding {arcname}")
            zf.write(src, arcname)

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 ** 2)
    print(f"\nSuccessfully created submission zip: {ZIP_PATH} ({zip_size_mb:.1f} MB in {time.time() - t0:.1f}s)")

if __name__ == "__main__":
    package()
