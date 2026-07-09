# Research-Gap Report: Federated Learning

## Domain Topic Landscape

The federated learning literature is heavily concentrated in a core cluster of FL methodology, data, and model research (242 papers, mean year 2021.4), with smaller but more recent satellite clusters in healthcare/medical applications (27 papers, mean year 2022.3), blockchain-based privacy and data sharing (18 papers, mean year 2021.5), and CNN-based crop and plant disease detection (7 papers, mean year 2023.0). The recency gradient — with the disease/CNN cluster averaging 2023.0 versus the core cluster's 2021.4 — suggests that agricultural AI applications are an emerging frontier, while the dominant FL methodology body of work remains the field's gravitational center.

---

## Validated Gap

### Gap: Blockchain-Secured Federated Learning for Agricultural Disease Detection

**Why it is a gap:**
The intersection of the blockchain/privacy cluster (n = 18) and the crop disease/CNN cluster (n = 7) is sparsely bridged. Despite 142 bridge papers touching both topic neighborhoods, targeted searches for direct integration — "blockchain" + "crop disease detection," "privacy-preserving machine learning" + "plant disease," and "secure data sharing" + "apple leaf disease" — returned only **12 recent papers**, yielding a prominence score of just **0.1**. This low density indicates that privacy-preserving, blockchain-anchored mechanisms have not been systematically applied to federated plant pathology models, even as both sub-fields are individually active. The gap carries a **confidence of 0.52**, classifying it as **WEAK** — meaning preliminary evidence of underexploration exists, but the signal is not yet definitive, and any research agenda should treat confirmation of the gap as a first-order objective.

**Research Agenda:**
1. **Systematic mapping study:** Conduct a structured literature review explicitly at the blockchain × federated learning × plant disease tripoint to precisely quantify coverage, identify which privacy threat models (e.g., gradient inversion, model poisoning) have been addressed in agricultural CNN settings, and confirm the gap's boundaries before committing resources.
2. **Prototype framework design:** Develop a privacy-preserving FL architecture — incorporating lightweight blockchain-based audit trails or secure aggregation protocols — tailored to the data-heterogeneity and bandwidth constraints typical of distributed agricultural edge nodes contributing crop disease imagery.
3. **Empirical validation on benchmark datasets:** Evaluate the proposed framework on established plant disease datasets (e.g., PlantVillage, apple leaf corpora) under realistic non-IID partitioning schemes, measuring both model utility (classification accuracy) and privacy/security guarantees, to produce reproducible baselines for the community.

---

## Rejected Candidates

Five candidate gaps were rejected due to substantial existing literature: FL × healthcare (1,078 recent papers), FL × blockchain/privacy (4,781), healthcare × blockchain (43), FL × disease/CNN (41), and healthcare × disease/CNN (41) — all indicating sufficiently active research coverage to disqualify them as genuine gaps.