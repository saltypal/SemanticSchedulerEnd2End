# Primary-source acquisition audit

Checked30October2026. Metadata/header-only checks; no media downloaded locally for this audit.

| Source | Reference | Finding |
|---|---|---|
| COCO | https://cocodataset.org/ ; https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/coco.py | Individual image URLs; official image-info archive used for boundedselection. Per-image licenses retained where provided. |
| Open Images V7 | https://storage.googleapis.com/openimages/web/download_v7.html ; https://storage.googleapis.com/openimages/web/factsfigures_v7.html | Official per-ID subset path. Preserve source attribution/license; image-info collection is not an annotation join. |
| Places365 | https://github.com/CSAILVision/places365 ; https://data.csail.mit.edu/places/places365/val_256.tar | HTTPS Range header probe returned206, bytes0–511/525158400; first TARheader `val_256/`. Read at most64MiB, never retain fullarchive. Prefixsamplingbias explicit. |
| CrisisMMD | https://crisisnlp.qcri.org/crisismmd.html | Reuse the existing pinned QCRI/CrisisMMD media mapping; official landing-page fetch returned403 duringaudit. No native SPI labels inferred. |
| BDD100K | https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/download.rst ; https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/license.rst | Imagebundle5.3GB and videosverylarge; managedsmallinput needed. 100Kimages are isolated10secondkeyframes, not continuous video. |
| ImageNet | https://www.image-net.org/download-images.php | Login/agreement required; use authorizedmanagedselection, noaccessbypass. |
| DoTA | https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly ; https://raw.githubusercontent.com/MoonBlvd/Detection-of-Traffic-Anomaly/master/dataset/download_DoTA.py | FullYouTubesources canbelong; authorframearchive55GB. Managedshortclips/frames only, retain originalsourceoffsets. |
| UCF-Crime | https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/ | Officialpageoffers archives/parts; individualvideoaccess notverified. Managedselectionrequired. |
| xBD / xView2 | https://xview2.org/download ; https://github.com/DIUx-xView/xView2_baseline | Access/acceptance requires verification. Managedauthorizedsample; no third-partylicensingassumption. |
| RescueNet | https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation ; https://api.figshare.com/v2/articles/22826369 | ValidationZIP2,373,908,074bytes; headerfetch403. AuthorREADME CC-BY-NC-ND conflicts with FigshareCC0metadata; preserve stricternotice pendingresolution. |
| FloodNet | https://github.com/BinaLab/FloodNet-Supervised_v1.0 | Authorstates CDLA-Permissive; usable per-imageDropboxlisting notverified. Managedselectionrequired; schedulertransferonly. |

None of these dataset categories supplies universal operator-action urgency. Native image/object/
condition annotations and future scheduling reference labels must retain their own scope/provenance.
