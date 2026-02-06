# Hunting Gadgets in Neural Networks: Building a Taxonomy of Structural Patterns

**Project Silent Scalpel — Post 2 of 10**

*Previously: In [Post 1](https://github.com/Vect0rdecay/SILENT-SCALPEL/blob/main/1-ONNX-Graphs-Gadgets.md), I introduced ONNX computational graphs, the concept of "gadgets" as structural motifs asspcoated with adversarial behaviors, and hypothesized that graph structure might indicates attack surface (what's possible) but not robustness (whether attacks succeed).*

---

## From Concept to Catalog

Post 1 ended with a question: if these structural motifs exist in neural network graphs, what are they specifically? I had the concept, but I needed to build the actual catalog.

This post covers how I surveyed the adversarial ML literature, extracted architectural patterns associated with successful attacks/robustness research, and formalized them into a **subset** of the gadget taxonomy focused on **input-driven attack surfaces** (20 gadgets). The goal was to create a systematic vocabulary for attack surface analysis that can also act as a map of sorts to help understand and reverse engineer model DAGs. Even though the motifs are a 100% reliable signal if an attack would work, they really helped me and hopefully will help others, better understand how to read model computational graphs at the op level and understand what is happening.

Note: the **full taxonomy** I used to build a tool that includes additional supply-chain motifs are not disclosed here yet for various reasons. This post intentionally does *not* cover those.

---

## Extensive Research Paper Survey

I used Claude to help me gather and analyze over 100 published papers on adversarial attacks and model robustness that cover both vision and audio models. I was looking for structural patterns that could help me determine what a model's graph needed to look like for a particular attack type to work.

Most adversarial ML papers focus on the attack algorithm and things like the optimization procedure, the perturbation bounds, the success rates. The architecture is often overlooked and treated as a given. But buried in the methodology sections, threat models, and experimental setups of most if not all of these papers were implicit assumptions about structural requirements. I found this to be a gold mine. And I started panning for gold.

### Methodology

For each paper, I extracted:

1. **Architectural requirements** — What does the model need to have for this attack to be applicable?
2. **Structural patterns** — What graph properties might determine whether this attack type applies?
3. **Architectural modifications** — What changes to architecture affected attack applicability?

### Emergent Patterns

The papers clustered around some recurring themes:

**Patch attack papers** seemed to consistently target models with global pooling which collapses spatial features into a single vector and no spatial attention mechanisms that could weigh regions differently. The unstated assumption I figured out was that if the model aggregates all spatial information equally without prioritizing areas, a small adversarial patch placed anywhere in the image can dominate and alter the final classification output, making these architectures particularly vulnerable to localized perturbations.

**Physical-world attack papers** emphasized aliasing where high-frequency details are misrepresented as lower frequencies during sampling and downsampling behavior that reduces image resolution. Printed adversarial patterns, like stickers or signs, inherently contain these high-frequency components from edges and textures. Models without anti-aliasing filters, such as low-pass smoothing before downsampling, allow those frequencies to fold into the internal representation, enabling attacks that survive real-world transformations like printing, viewing angles, or lighting changes. I wrote about this earlier in an X [post](https://x.com/jsn_yrty/status/1998136466279723128) about 3D Gaussian Splats Attacks and some Gaussian Splat background information found [here](https://x.com/jsn_yrty/status/1993391572940144828).

**Detector evasion papers** exploited specific head structures in object detection models, including objectness scores that estimate the likelihood of an object in a region, anchor configurations that define predefined bounding box shapes and sizes, and Non-Maximum Suppression thresholds that filter overlapping detections. The attack surface depended heavily on the detection architecture's design choices, such as two-stage proposals or anchor-based matching, rather than just the shared backbone network for feature extraction. This meant adversaries could craft perturbations to suppress detections or create false positives by targeting these specialized components.

**ViT attack papers** found that Vision Transformers weren't inherently more robust than convolutional networks, despite their attention-based design. The patch tokenization process, which divides the image into fixed-size patches treated as discrete tokens, created exploitable boundaries where adversaries could inject perturbations that propagate through layers. Self-attention mechanisms didn't automatically down-weight or isolate adversarial regions, allowing small changes in one patch to influence global decisions via unchecked token interactions.

From these patterns that materialized out of the hundreds of papers, I started building the taxonomy.

---

## Taxonomy (Public Subset): 20 Gadgets Across 4 Categories

I organized the patterns into four categories based on what part of the architecture they describe.

### Category 1: Core CNN Gadgets

These appear in standard convolutional architectures and relate to information flow and aggregation.

**GAP_FC_HEAD** — Global Average Pooling feeding a Fully Connected head.

This is the ending structure of most classifiers (ResNet, VGG, EfficientNet). GAP collapses spatial dimensions (H×W×C → 1×1×C) by averaging across height and width. The security implication: an adversarial patch anywhere in the image contributes to that averaged representation. The network has no structural mechanism to ignore a malicious region.

*Detection: Look for GlobalAveragePool → Flatten/Reshape → Gemm/MatMul sequence.*

**ALIASING_DOWNSAMPLE** — Stride-2 convolutions or pooling without anti-aliasing filters.

Downsampling without low-pass filtering causes high-frequency components to "fold" into lower frequencies (aliasing). For physical-world attacks, this matters because printed adversarial patterns contain high frequencies that would otherwise be filtered during the print-capture pipeline.

*Detection: Stride > 1 in Conv or Pool without preceding blur/anti-alias operation.*

**MAXPOOL_AFTER_FUSION** — MaxPool following a Concat or Add operation.

When max pooling selects from fused features, adversarial perturbations can be crafted to become the maximum in their local region. The perturbation gets selected and propagated, amplifying its effect.

*Detection: MaxPool node with Concat or Add as input.*

**HIGH_FANIN_FUSION** — Three or more feature paths merging.

More inputs to a fusion point means more attack paths. Gradients fan out across all branches, giving adversaries more degrees of freedom for optimization.

*Detection: Concat or Add nodes with 3+ inputs.*

**SKIP_CONNECTION** — Residual connections allowing gradients through multiple paths.

Essential for training deep networks, but skip connections preserve adversarial signals without attenuation. The identity mapping lets perturbations bypass convolutional processing.

*Detection: Add nodes where one input comes from earlier in the graph (graph distance > 1).*

**NORMALIZER** — BatchNorm or LayerNorm layers.

Normalization learns statistics from clean training data. Adversarial inputs cause distribution shift—their statistics differ from learned values. This is especially problematic combined with physical-world conditions.

*Detection: BatchNormalization or LayerNormalization nodes.*

### Category 2: Spatial Attention Gadgets

These relate to whether a network can selectively attend to (or ignore) spatial regions.

**NO_SPATIAL_ATTENTION** — Absence of attention mechanisms (SE blocks, CBAM, self-attention with spatial masking).

Without spatial attention, every input region contributes equally. This is a critical enabler of patch attacks—the model cannot learn to down-weight adversarial regions.

*Detection: Absence of squeeze-excitation patterns, CBAM patterns, or spatial attention mechanisms.*

**HAS_SPATIAL_ATTENTION** — Presence of spatial attention.

Networks with attention can theoretically learn to ignore adversarial regions. However, attention is not a complete defense—adversaries can craft perturbations that attract rather than repel attention.

*Detection: Presence of SE blocks, CBAM, or spatial attention patterns.*

### Category 3: Object Detector Gadgets

Detection architectures have attack-surface patterns beyond classification.

**OBJECTNESS_HEAD** — Single scalar objectness score per proposal.

One perturbation can suppress this score, making objects "disappear" from detection entirely.

**ANCHOR_BASED_DETECTION** — Fixed anchor grid.

Adversaries know exactly which image locations and scales the detector examines, making attacks more efficient and targeted.

**FPN_STRUCTURE** — Feature Pyramid Network (multi-scale features).

Multi-scale pyramids force attacks to work across resolutions. When the pyramid is driven by a shared bottom-up pathway, a single perturbation can propagate across all scales.

**TWO_STAGE_RPN** — Region Proposal Networks (Faster R-CNN family).

If the RPN is compromised, no proposals reach the second stage. RPN-targeted attacks have outsized impact.

**SINGLE_OBJECTNESS_PATH** — No detection redundancy.

If there’s only one objectness pathway with no auxiliary heads or redundancy, suppression attacks have no fallback to “catch” the object.

**NMS_DEPENDENCY** — Non-Maximum Suppression in post-processing.

NMS operates on confidence scores. Manipulating confidences can suppress correct detections or retain false ones without fooling the actual learned features.

**SHARED_BACKBONE** — Single feature extractor for multiple tasks.

One perturbation affects everything downstream—detection, segmentation, tracking all compromised together.

**DETECTION_HEAD_PATTERN** — Multi-output detection head (class/box/objectness coupled).

Shared head features create multiple interacting targets. In practice, post-processing can be manipulated by changing the confidence outputs, even when learned features are unchanged.

### Category 4: ViT-Specific Gadgets

Vision Transformers have characteristic patterns distinct from CNNs.

**VIT_PATCH_EMBEDDING** — Non-overlapping patch tokenization (typically 16×16).

Creates discrete token boundaries. An adversarial patch can align perfectly with token boundaries, giving maximum influence over that token's embedding with no "bleeding" into neighbors.

**UNREGULARIZED_ATTENTION** — Attention without dropout or entropy regularization.

Softmax attention can concentrate entirely on adversarial tokens, giving them disproportionate influence over the output.

**CLS_TOKEN_AGGREGATION** — Classification via a special CLS token.

Analogous to GAP_FC_HEAD—a single aggregation point that cannot spatially filter adversarial content.

**AGGRESSIVE_EARLY_DOWNSAMPLING** — Large initial stride / aggressive early reduction.

Some efficient architectures discard spatial detail early (large stride or patchification). This creates a “small object / small patch” sensitivity region from the very first stages.

---

## Chains: How Gadgets Combine

Individual gadgets are associated with a potential targeted attack surface, whereas **chains** seem to correlate to a more complete attack pattern that in some cases can be used to one-shot predict the type of adversarial attack technique that is most likely to succeed. A chain is a combination of gadgets that together enable a specific attack class.

**CHAIN-PATCH-ATTACK-SURFACE**

Combines: GAP_FC_HEAD + NO_SPATIAL_ATTENTION

The global pooling ensures any patch affects the output. The lack of spatial attention means the network cannot learn to ignore it. Present in most standard CNN classifiers.

**CHAIN-PHYSICAL-WORLD-ATTACK**

Combines: ALIASING_DOWNSAMPLE + NORMALIZER

Aliasing lets high-frequency adversarial patterns survive the print-photograph pipeline. Normalization distribution shift amplifies perturbations under varying real-world conditions.

**CHAIN-VIT-PATCH-ATTACK**

Combines: VIT_PATCH_EMBEDDING + UNREGULARIZED_ATTENTION + CLS_TOKEN_AGGREGATION

Clean token boundaries for precise patch placement, attention concentration on adversarial tokens, and single-point aggregation. The ViT-specific attack surface.

**CHAIN-OBJECT-DISAPPEARANCE**

Combines: OBJECTNESS_HEAD + SINGLE_OBJECTNESS_PATH

Suppress one score, the object vanishes from detection entirely.

**CHAIN-ANCHOR-EXPLOITATION**

Combines: ANCHOR_BASED_DETECTION + NMS_DEPENDENCY

Fixed anchors expose predictable surfaces; NMS gives a second lever via confidence manipulation and post-processing.

The chain concept describes what appears to be complete attack landscapes that have shown up as patterns across many papers. A single gadget indicates partial attack surface, but combined with others, it forms the full structural requirements for a particular attack class to be more likely to work. Always keep in mind though, these motifs and patterns do not predict ASR because of all the other factors involved that must be taken into account. Think of this more like a somewhat foggy oracle through which one might look to get a pretty good idea about what attack type to start with first.

---

## Detection Results on Real Models

I built tooling to detect these gadgets in ONNX models. Here's what common architectures look like through this lens:

### ResNet-50

```
Gadgets detected:
- GAP_FC_HEAD: Present (GlobalAveragePool → Gemm)
- NO_SPATIAL_ATTENTION: Present (no SE/CBAM blocks)
- SKIP_CONNECTION: Present (49 Add operations)
- NORMALIZER: Present (49 BatchNormalization nodes)
- ALIASING_DOWNSAMPLE: Present (stride-2 convolutions without anti-alias)

Chains formed:
- CHAIN-PATCH-ATTACK-SURFACE: Active
- CHAIN-PHYSICAL-WORLD-ATTACK: Active
```

ResNet-50 has the classic CNN attack-surface profile: susceptible to both patch attacks and physical-world attacks based purely on structural analysis.

### ViT-Base

```
Gadgets detected:
- VIT_PATCH_EMBEDDING: Present (16x16 non-overlapping patches)
- UNREGULARIZED_ATTENTION: Present (standard softmax attention)
- CLS_TOKEN_AGGREGATION: Present
- NO_SPATIAL_ATTENTION: Partial (has self-attention but no spatial suppression)

Chains formed:
- CHAIN-VIT-PATCH-ATTACK: Active
```

ViT has different gadgets than ResNet, but still forms potentially exploitable chains. Self-attention exists but isn't protective. Tt doesn't down-weight adversarial regions.

### YOLO (Single-Stage Detector)

```
Gadgets detected:
- OBJECTNESS_HEAD: Present
- ANCHOR_BASED_DETECTION: Present
- NMS_DEPENDENCY: Present (external post-processing)
- SHARED_BACKBONE: Present

Chains formed:
- CHAIN-OBJECT-DISAPPEARANCE: Active
- CHAIN-ANCHOR-EXPLOITATION: Active
```

YOLO's architecture has characteristic detector attack surfaces distinct from classifiers.

---

## The So What of All This

The taxonomy provides a vocabulary for basic graph analysis but also potential attack surface analysis and enables hypothesis-driven red teaming.

**Instead of "try every attack":**
- Model has CHAIN-PATCH-ATTACK-SURFACE? Prioritize patch attacks.
- Model has anti-aliasing filters? Deprioritize physical-world attacks.
- Model uses anchor-free detection? Different attack strategy than anchor-based.

**For comparative analysis:** When choosing between architectures for security-sensitive deployment, the gadget profile is relevant information alongside accuracy metrics.

As established in Post 1, gadgets describe attack surface, i.e. what's structurally enabled, not robustness. Two models with identical gadget profiles can have completely different empirical robustness depending on training and how they're deployed. The taxonomy informs what to test, not whether attacks will succeed.

For the complementary representation-level view (how intent can be carried through multimodal inputs and reconstructed at inference time instead of just vision or just audio), see my exploit carrier modality [framework](https://github.com/Vect0rdecay/lvlm-adversarial-carrier-modality-framework/blob/main/ontology.md).

---

## What's Next

The taxonomy gave me a systematic way to characterize attack surface. But the obvious question remained: if structure doesn't predict robustness, what does?

I started measuring everything I could think of—weight norms, gradient properties, activation statistics. Most showed nothing interesting. But one metric showed a striking pattern that had nothing to do with graph structure.

In Post 3, I'll cover what happened when I started looking at weight distributions instead of only graph topology and then combined the two areas.

---

## Key Takeaways

1. **100+ papers analyzed** — Extract structural patterns, not just attack algorithms.

2. **20 gadgets across 4 categories** — Core CNN, Spatial Attention, Object Detector, and ViT-Specific each have characteristic patterns.

3. **Chains combine gadgets into potential attack paths** — Individual gadgets create surface; chains represent potential *applicability* patterns for an attack class.

4. **Different architectures have different profiles** — ResNet, ViT, and YOLO look distinct through the gadget lens. (This will be an abvious takeaway to anyone who has been looking at these model DAGs for more than a minute.)

5. **Enables hypothesis-driven red teaming** — Use structural analysis to prioritize which attacks are worth testing empirically.

---

## Appendix: Full Gadget Reference

| Category | Gadget | Description |
|----------|--------|-------------|
| Core CNN | GAP_FC_HEAD | Global Average Pooling + FC head |
| Core CNN | ALIASING_DOWNSAMPLE | Stride >1 without anti-aliasing |
| Core CNN | MAXPOOL_AFTER_FUSION | MaxPool following Concat/Add |
| Core CNN | HIGH_FANIN_FUSION | 3+ paths merging |
| Core CNN | SKIP_CONNECTION | Residual connections |
| Core CNN | NORMALIZER | BatchNorm/LayerNorm |
| Spatial | NO_SPATIAL_ATTENTION | No SE/CBAM/spatial attention |
| Spatial | HAS_SPATIAL_ATTENTION | Has attention mechanisms |
| Detector | OBJECTNESS_HEAD | Single objectness score |
| Detector | ANCHOR_BASED_DETECTION | Fixed anchor grid |
| Detector | FPN_STRUCTURE | Feature Pyramid Network |
| Detector | TWO_STAGE_RPN | Region Proposal Network |
| Detector | SINGLE_OBJECTNESS_PATH | No detection redundancy |
| Detector | NMS_DEPENDENCY | Non-Maximum Suppression |
| Detector | SHARED_BACKBONE | Single feature extractor |
| Detector | DETECTION_HEAD_PATTERN | Multi-output detection head |
| ViT | VIT_PATCH_EMBEDDING | Non-overlapping patch tokens |
| ViT | UNREGULARIZED_ATTENTION | No attention regularization |
| ViT | CLS_TOKEN_AGGREGATION | CLS token classification |
| ViT | AGGRESSIVE_EARLY_DOWNSAMPLING | Large initial stride / early reduction |

---

*Project Silent Scalpel is a series documenting my research journey into neural network security. This research was done independently outside of work on personal time, using my own hardware, software, and AI-assisted development resources. Any conclusions or views expressed herein are my own and do not reflect those of my employer.*
