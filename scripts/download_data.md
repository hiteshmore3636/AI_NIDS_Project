# Dataset Download Guide

AI-NIDS uses publicly benchmarked network intrusion datasets to ensure scientific reproducibility.

---

## 1. Primary Dataset: CIC-IDS2017

### Official Sources:
* **UNB CIC Website**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
* **Kaggle Mirror**: Search `cicids2017` on Kaggle for pre-hosted mirrors of `MachineLearningCVE.zip`.

### Target Directory:
Extract the 8 CSV files directly into `dataset/raw/`:
* `Monday-WorkingHours.pcap_ISCX.csv`
* `Tuesday-WorkingHours.pcap_ISCX.csv`
* `Wednesday-workingHours.pcap_ISCX.csv`
* `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
* `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
* `Friday-WorkingHours-Morning.pcap_ISCX.csv`
* `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
* `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

---

## 2. Secondary Dataset: UNSW-NB15

### Official Source:
* **UNSW Research Portal**: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
* Download the CSV features files into `dataset/raw/unsw/`.

---

## 3. Automated Synthetic Sample for Quick Verification
If large raw datasets are not yet downloaded, run the synthetic dataset generator in Phase 1:
```bash
python -m ml_engine.sample_generator --rows 10000 --output dataset/raw/sample_synthetic.csv
```
This produces a 10,000-flow sample with correct columns for immediate testing of the preprocessing and training pipeline.
