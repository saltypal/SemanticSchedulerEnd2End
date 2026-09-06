# Graph Report - D:/Bunker/BaseCamp/WirelessCommunication/SemanticComms/SemanticSchedulerEnd2End  (2026-09-06)

## Corpus Check
- Corpus is ~9,203 words - fits in a single context window. You may not need a graph.

## Summary
- 479 nodes · 504 edges · 87 communities (37 shown, 50 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Component Contracts|Component Contracts]]
- [[_COMMUNITY_Deferred Components|Deferred Components]]
- [[_COMMUNITY_Quality Evaluation|Quality Evaluation]]
- [[_COMMUNITY_AWGN Symbol Mapping|AWGN Symbol Mapping]]
- [[_COMMUNITY_Latent Codebook|Latent Codebook]]
- [[_COMMUNITY_VQ Packet Coding|VQ Packet Coding]]
- [[_COMMUNITY_LDPC QPSK Baseline|LDPC QPSK Baseline]]
- [[_COMMUNITY_JPEG Source Codec|JPEG Source Codec]]
- [[_COMMUNITY_Stage One Documentation|Stage One Documentation]]
- [[_COMMUNITY_Upstream Source Patching|Upstream Source Patching]]
- [[_COMMUNITY_Artifact Integration Tests|Artifact Integration Tests]]
- [[_COMMUNITY_Component Registries|Component Registries]]
- [[_COMMUNITY_Channel Implementations|Channel Implementations]]
- [[_COMMUNITY_Pipeline Orchestration|Pipeline Orchestration]]
- [[_COMMUNITY_Real Image Data|Real Image Data]]
- [[_COMMUNITY_Lifecycle Management|Lifecycle Management]]
- [[_COMMUNITY_Comparison Reporting|Comparison Reporting]]
- [[_COMMUNITY_Artifact Persistence|Artifact Persistence]]
- [[_COMMUNITY_Notebook Generation|Notebook Generation]]
- [[_COMMUNITY_Result Persistence|Result Persistence]]
- [[_COMMUNITY_Checkpoint Metadata|Checkpoint Metadata]]
- [[_COMMUNITY_Project Path Utilities|Project Path Utilities]]
- [[_COMMUNITY_Scheduler Artifact|Scheduler Artifact]]
- [[_COMMUNITY_Codebook Diagnostics|Codebook Diagnostics]]
- [[_COMMUNITY_Bandwidth Metrics|Bandwidth Metrics]]
- [[_COMMUNITY_Native CBR|Native CBR]]
- [[_COMMUNITY_Compression Metrics|Compression Metrics]]
- [[_COMMUNITY_Latent Storage Metric|Latent Storage Metric]]
- [[_COMMUNITY_Raw Image Accounting|Raw Image Accounting]]
- [[_COMMUNITY_Image Cropping|Image Cropping]]
- [[_COMMUNITY_Evaluation Plotting|Evaluation Plotting]]
- [[_COMMUNITY_Runtime Logging|Runtime Logging]]
- [[_COMMUNITY_Timing Profiling|Timing Profiling]]
- [[_COMMUNITY_Identity Denormalization|Identity Denormalization]]
- [[_COMMUNITY_Reconstruction Clamping|Reconstruction Clamping]]
- [[_COMMUNITY_Tensor Image Export|Tensor Image Export]]
- [[_COMMUNITY_Identity Normalization|Identity Normalization]]
- [[_COMMUNITY_Shape Validation|Shape Validation]]
- [[_COMMUNITY_Bit Error Metric|Bit Error Metric]]
- [[_COMMUNITY_Symbol Error Metric|Symbol Error Metric]]
- [[_COMMUNITY_Shared Codebook Storage|Shared Codebook Storage]]
- [[_COMMUNITY_Run Report|Run Report]]
- [[_COMMUNITY_Knowledge Evidence Artifact|Knowledge Evidence Artifact]]
- [[_COMMUNITY_Semantic Parameter Artifact|Semantic Parameter Artifact]]
- [[_COMMUNITY_Execution Plan|Execution Plan]]
- [[_COMMUNITY_Runtime Context|Runtime Context]]
- [[_COMMUNITY_Device Inspection|Device Inspection]]
- [[_COMMUNITY_Reproducible Seeding|Reproducible Seeding]]
- [[_COMMUNITY_JSON Serialization|JSON Serialization]]
- [[_COMMUNITY_Artifact Package|Artifact Package]]
- [[_COMMUNITY_Baseline Package|Baseline Package]]
- [[_COMMUNITY_Channel Package|Channel Package]]
- [[_COMMUNITY_QPSK Wrapper|QPSK Wrapper]]
- [[_COMMUNITY_Evaluation Package|Evaluation Package]]
- [[_COMMUNITY_INFRA Package|INFRA Package]]
- [[_COMMUNITY_Postprocessing Package|Postprocessing Package]]
- [[_COMMUNITY_Preprocessing Package|Preprocessing Package]]
- [[_COMMUNITY_Stage One Notebook|Stage One Notebook]]
- [[_COMMUNITY_Semantic Evidence Separation|Semantic Evidence Separation]]
- [[_COMMUNITY_Registry Package|Registry Package]]
- [[_COMMUNITY_Codebook Package|Codebook Package]]
- [[_COMMUNITY_KMeans Codebook Alias|KMeans Codebook Alias]]
- [[_COMMUNITY_Communication Package|Communication Package]]
- [[_COMMUNITY_Deployment Package|Deployment Package]]
- [[_COMMUNITY_Interface Package|Interface Package]]
- [[_COMMUNITY_Test Configuration|Test Configuration]]
- [[_COMMUNITY_Transformer Pipeline Package|Transformer Pipeline Package]]
- [[_COMMUNITY_Utility Package|Utility Package]]

## God Nodes (most connected - your core abstractions)
1. `ComponentUnavailableError` - 26 edges
2. `MiniBatchKMeansVQ` - 13 edges
3. `require_torch()` - 12 edges
4. `real_to_complex_per_image()` - 9 edges
5. `Stage 1A Operational Runtime` - 8 edges
6. `ComponentRegistry` - 7 edges
7. `frame_payload_bits()` - 6 edges
8. `recover_payload_bits()` - 6 edges
9. `complex_to_real_per_image()` - 6 edges
10. `CodebookPacketHeader` - 6 edges

## Surprising Connections (you probably didn't know these)
- `test_k256_packet_has_eight_bit_indices_and_full_rate_mask_per_image()` --calls--> `CodebookPacketHeader`  [INFERRED]
  Test/tests/test_packet_accounting.py → src/EncDecPipeline/Codebooks/codebook_codec.py
- `register_stage1a_components()` --calls--> `register_minibatch_kmeans_vq()`  [INFERRED]
  E2EMain.py → src/EncDecPipeline/Codebooks/minibatch_kmeans_vq.py
- `validate_stage1a_skeleton()` --calls--> `StageController`  [INFERRED]
  E2EMain.py → src/INFRA/Core/stage_controller.py
- `test_per_image_mapping_matches_independently_sharded_batches()` --calls--> `real_to_complex_per_image()`  [INFERRED]
  Test/tests/test_channel_mapping.py → src/Channels/channel_utils.py
- `test_data_parallel_shards_preserve_per_image_symbols()` --calls--> `real_to_complex_per_image()`  [INFERRED]
  Test/tests/test_channel_mapping.py → src/Channels/channel_utils.py

## Hyperedges (group relationships)
- **Stage 1A Experiment Contract** — default_stage1a_configuration, stage1a_training_configuration, stage1a_awgn_mse_lock, stage1a_k256_codebook, stage1a_div2k_kodak_data [EXTRACTED 1.00]
- **Stage 1A Reproducibility Contract** — readme_stage1a_repository, readme_kaggle_runner_notebook, requirements_local_runtime, readme_swinjscc_artifact_manifest, paths_artifact_layout [EXTRACTED 1.00]

## Communities (87 total, 50 thin omitted)

### Community 0 - "Component Contracts"
Cohesion: 0.06
Nodes (20): ABC, ChannelInterface, Channel contract; implementations belong to the top-level Channels package., CodebookInterface, Contract for vector quantization without coupling the encoder to K-means., DeploymentInterface, Deployment contract reserved for NS-3, Sionna system, and O-RAN adapters., EncDecInterface (+12 more)

### Community 1 - "Deferred Components"
Cohesion: 0.07
Nodes (22): Rician support is deliberately deferred until its CSI assumptions are specified., RicianChannel, Future semantic codebook contracts; distinct from Stage 1A latent VQ., SemanticCodebook, GroupedVQ, ProductVQ, Stage availability guard used by the CLI and notebooks., StageController (+14 more)

### Community 2 - "Quality Evaluation"
Cohesion: 0.07
Nodes (24): EvaluationArtifact, Machine-readable evaluation result artifact., create_channel(), Concrete channel factory; no channel implementation is hosted inside INFRA., register_builtin_channels(), ImageQualityEvaluator, Typed image evaluator registered for dependency-injected pipeline use., register_image_quality_evaluator() (+16 more)

### Community 3 - "AWGN Symbol Mapping"
Cohesion: 0.09
Nodes (27): Artifact crossing a communication channel., Channel input/output and accounting for one independently mapped batch., TransmissionArtifact, complex_to_real_per_image(), esn0_to_noise_variance(), normalize_complex_per_image(), qpsk_bits_from_symbols(), qpsk_symbols_from_bits() (+19 more)

### Community 4 - "Latent Codebook"
Cohesion: 0.1
Nodes (12): LatentArtifact, Encoder-output runtime artifact., Latent tensor and its rate-allocation mask without transport assumptions., CodebookInterface, CodebookPacketHeader, index_width(), Fields that must travel with each VQ packet; codebook storage is separate., MiniBatchKMeansVQ (+4 more)

### Community 5 - "VQ Packet Coding"
Cohesion: 0.12
Nodes (17): _crc32_bits(), decode_codebook_packet_bits(), encode_codebook_packet_bits(), _integer_bits(), _metadata_bits(), pack_fixed_width_indices(), Packet schema and bit accounting for shared-codebook VQ packets., Verify packet CRC and recover indices/mask for one VQ image. (+9 more)

### Community 6 - "LDPC QPSK Baseline"
Cohesion: 0.13
Nodes (16): bits_to_integer(), crc32_bits(), frame_payload_bits(), integer_to_bits(), Transport headers, CRC and padding accounting for digital baselines., recover_payload_bits(), DigitalTransmissionResult, LDPCQPSKAWGN (+8 more)

### Community 7 - "JPEG Source Codec"
Cohesion: 0.16
Nodes (7): bytes_to_bits(), Exact conversion between JPEG bytes and one-dimensional binary payloads., JPEGCodec, JPEG byte codec with no hidden image resizing or quality adaptation., encode_jpeg_for_transport(), JPEGBitstream, JPEG source coding only; transport is injected separately through DigitalPHY.

### Community 8 - "Stage One Documentation"
Cohesion: 0.15
Nodes (13): Default Stage 1A Configuration, Runtime Logging Configuration, Model Result and Upstream Source Paths, Corrected Modern PyTorch Port, INFRA Contract and Orchestration Boundary, Stage 1A Operational Runtime, SwinJSCC Tensor State Artifact Manifest, Local Inference Runtime Dependencies (+5 more)

### Community 9 - "Upstream Source Patching"
Cohesion: 0.33
Nodes (9): apply_patch(), ensure_upstream(), main(), Clone one audited SwinJSCC revision and apply minimal modern-runtime corrections, Replace an audited repeated hunk, accepting a fully patched file on rerun., replace_expected(), replace_once(), repository_root() (+1 more)

### Community 10 - "Artifact Integration Tests"
Cohesion: 0.2
Nodes (5): ImageArtifact, Input-image runtime artifact., A batched image tensor plus provenance that must survive a pipeline run., Runs only when a Kaggle-produced artifact directory is supplied explicitly., test_encoder_only_then_decoder_only_round_trip()

### Community 11 - "Component Registries"
Cohesion: 0.22
Nodes (3): ComponentRegistry, Small explicit registry used for dependency injection and configuration lookup., Maps stable names to constructors; duplicate registration is an error.

### Community 12 - "Channel Implementations"
Cohesion: 0.22
Nodes (7): ChannelInterface, AWGNChannel, AWGN channel whose complex conversion is DataParallel-safe., Unit-energy complex AWGN channel using Es/N0 in dB.      `rate_mask` is applied, Flat Rayleigh channel reserved for controlled ablations beyond Stage 1A., Perfect-CSI flat Rayleigh channel; not part of the locked Stage 1A results., RayleighChannel

### Community 13 - "Pipeline Orchestration"
Cohesion: 0.33
Nodes (4): PipelineManager, PipelineRun, Dependency-injected Stage 1A execution flow., Runs encode -> channel -> decode -> evaluation without importing model internals

### Community 14 - "Real Image Data"
Cohesion: 0.43
Nodes (6): dataset_provenance(), discover_images(), file_sha256(), pil_to_tensor(), Strict real-image discovery: there is intentionally no synthetic fallback., require_torch_and_pillow()

### Community 16 - "Comparison Reporting"
Cohesion: 0.33
Nodes (4): Comparison rows preserve method, SNR, use count, quality, and failure semantics., validate_comparison_row(), CSV output without a pandas dependency., write_result_table()

### Community 18 - "Notebook Generation"
Cohesion: 0.53
Nodes (5): build_notebook(), code(), main(), markdown(), Generate the single Stage 1A Kaggle runner notebook from a stable template.

### Community 21 - "Project Path Utilities"
Cohesion: 0.67
Nodes (3): project_root(), Repository-relative paths only; runtime data paths remain configuration inputs., resolve_project_path()

## Knowledge Gaps
- **11 isolated node(s):** `KnowledgeArtifact`, `SchedulingArtifact`, `SemanticParametersArtifact`, `ExecutionPlan`, `RuntimeContext` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **50 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ComponentUnavailableError` connect `Deferred Components` to `Quality Evaluation`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `TransmissionArtifact` connect `AWGN Symbol Mapping` to `Latent Codebook`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `RicianChannel` connect `Deferred Components` to `Channel Implementations`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `ComponentUnavailableError` (e.g. with `RicianChannel` and `SemanticCodebook`) actually correct?**
  _`ComponentUnavailableError` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `require_torch()` (e.g. with `.transmit()` and `qpsk_llr()`) actually correct?**
  _`require_torch()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `real_to_complex_per_image()` (e.g. with `.transmit()` and `.transmit()`) actually correct?**
  _`real_to_complex_per_image()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Entry point for contract checks and manifest-validated Stage 1A inference setup.`, `Clone one audited SwinJSCC revision and apply minimal modern-runtime corrections`, `Replace an audited repeated hunk, accepting a fully patched file on rerun.` to the rest of the system?**
  _140 weakly-connected nodes found - possible documentation gaps or missing edges._