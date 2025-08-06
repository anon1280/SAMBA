# SAMBA: Toward a Long-Context EEG Foundation Model via Spatial Embedding and Differential Mamba
**Update: July 2025**

**SAMBA** is a scalable self-supervised framework for long-sequence EEG foundation modeling. It integrates 3D spatial-adaptive input embedding and differential Mamba modules to enable robust, efficient, and generalizable EEG representation learning across diverse recording configurations and cognitive tasks.

<p align="center">
  <img src="./Figures/SAMBA.png" alt="SAMBA Overview" width="900"/>
</p>

---

## Abstract

Modeling long EEG sequences is critical for developing generalizable neural representations, particularly due to the high temporal resolution and extended durations often required to capture brain dynamics. While transformer-based models have shown success on short EEG segments, their quadratic complexity prevents effective scaling to long contexts. Additionally, the diversity in EEG montages and subject variability presents significant generalization challenges.

We introduce **SAMBA**, a self-supervised learning framework featuring a U-shaped encoder-decoder architecture built on the linear-time **Mamba** module. SAMBA incorporates:

1. **Temporal Semantic Random Masking** — to reconstruct semantically masked segments in long sequences;
2. **Multi-Head Differential Mamba** — to reduce redundancy and enhance salient temporal features;
3. **Spatial-Adaptive Input Embedding (SAIE)** — to learn robust 3D spatial representations across heterogeneous EEG devices.

Evaluations across **13 EEG datasets** covering a range of tasks, montages, and sequence lengths demonstrate that SAMBA consistently outperforms state-of-the-art baselines, while maintaining low memory usage and fast inference. In addition, the learned spatial weights exhibit strong alignment with task-relevant neurophysiological regions, suggesting that SAMBA is both **learnable** and **interpretable**.

> 📌 The code is available at: https://github.com/anon1280/SAMBA

---

## Learnable Spatial Embedding Visualization

The figure below illustrates the **Spatial-Adaptive Input Embedding (SAIE)** module and its alignment with neurophysiological topology:

<p align="center">
  <img src="./Figures/SAIE.png" alt="SAIE Overview" width="800"/>
</p>


<table align="center">
  <tr>
    <td align="center">
        <img src="./Figures/Topo_gif/EyesState_GIF.gif" alt="Eyes State" width="250"/><br/>
      <b>Eyes Close/Open</b><br/>
      Emphasizes the <b>frontal lobe</b>, consistent with alpha modulation during eye closure[1].
    </td>
    <td align="center">
      <img src="./Figures/Topo_gif/DriverDistraction_GIF.gif" alt="Driver Distraction" width="250"/><br/>
      <b>Driver Distraction</b><br/>
      Highlights the <b>left temporal lobe</b>, associated with auditory processing (conversation during driving) and cognitive control under distraction[2].
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./Figures/Topo_gif/STEW_GIF.gif" alt="Workload Estimation" width="250"/><br/>
      <b>Workload Estimation</b><br/>
      Focuses on the <b>left frontal region</b>, linked to stress-induced workload processing[3].
    </td>
    <td align="center">
      <img src="./Figures/Topo_gif/TUAB_GIF.gif" alt="Seizure" width="250"/><br/>
      <b>Abnormal Detection (Seizure)</b><br/>
      Shows <b>occipital dominance</b>, consistent with findings that occipital electrodes yield better performance for distinguishing abnormal EEG[4].
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="./Figures/Topo_gif/MI_GIF.gif" alt="Motor Imagery" width="250"/><br/>
      <b>Motor Imagery</b><br/>
      Localizes around the <b>motor cortex</b>, reflecting sensorimotor activity during imagery tasks[5].
    </td>
  </tr>
</table>
---

#### References
<div style="font-size: smaller">

[1] Sven Hoffmann and Michael Falkenstein. *The correction of eye blink artefacts in the EEG: a comparison of two prominent methods.* *PLOS ONE*, 3(8):e3004, 2008.  
[2] Guofa Li, Xiaojian Wu, Arno Eichberger, Paul Green, Cristina Olaverri-Monreal, Weiquan Yan, Yechen Qin, and Yuezhi Li. *Drivers’ EEG responses to different distraction tasks.* *Automotive Innovation*, 6(1):20–31, 2023.  
[3] Gesa Berretz, Julian Packheiser, Oliver T. Wolf, and Sebastian Ocklenburg. *Acute stress increases left hemispheric activity measured via changes in frontal alpha asymmetries.* *iScience*, 25(2), 2022.  
[4] Sebas Lopez, G. Suarez, D. Jungreis, I. Obeid, and Joseph Picone. *Automated identification of abnormal adult EEGs.* In *Proceedings of the 2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)*, pages 1–5. IEEE, 2015.  
[5] Jiazhen Hong, Foroogh Shamsi, and Laleh Najafizadeh. *A deep learning framework based on dynamic channel selection for early classification of left and right hand motor imagery tasks.* In *Proceedings of the 2022 44th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)*, pages 3550–3553. IEEE, 2022.
</div>

## Model and Checkpoints

- The full architecture is implemented under `Models/`.
- Two pretrained checkpoints are provided under `Checkpoints/`:
  - `SAMBA-E`: trained using PyTorch Lightning
  - `SAMBA-T`: trained using native PyTorch
- The accompanying paper is currently in preparation. More features and documentation will be released soon.

---

## Repository Structure
```plaintext
SAMBA/
├── Env-requirement/     # Environment configs with dated backups
├── Checkpoints/         # Pretrained SAMBA models (SAMBA-E, SAMBA-T)
├── Models/              # Model architecture implementations
├── Experiments/         # PyTorch Lightning training modules
├── utility/             # Supporting functions: data loading, masking, loss, evaluation
├── Figures/             # Diagrams and visualizations
├── Montage/             # EEG montage metadata for multiple devices
└── README.md            # Project overview