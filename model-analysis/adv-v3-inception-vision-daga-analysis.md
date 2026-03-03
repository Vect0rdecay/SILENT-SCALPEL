# DAG Analysis: ADV Inception-v3 Vision Model

This is an analysis of an Inception-v3 vision model that has been trained with adversarial examples.

**Source:** [adv_inception_v3_Opset16.onnx](https://github.com/onnx/models/blob/main/Computer_Vision/adv_inception_v3_Opset16_timm/adv_inception_v3_Opset16.onnx)

**Date:** 1.18.26

---

The model metadata `.yaml` has some useful information about the input, size, ops counter, and number of parameters.

![Model metadata](media/image4.png)

## Data Flow Overview

The following is a general overview of the flow and layers of the Inception model.

```
Input image
    ↓
Stem (Conv + Pool stack)
    ↓
Inception blocks (many times)
  ┌─ Conv → Conv ─┐
  Input ──────┼─ Conv ────────┼─ Concat → output
  ├─ Conv → Conv ─┤
  └─ Pool → Conv ─┘
    ↓
Global average pool
    ↓
Fully connected / logits
    ↓
Softmax
```

## Summary of Elements

Some of the Netron and NN elements we encounter in this analysis.

![Summary of elements](media/image3.png)

## Input and 1st Layer

Model Input = batch size 1, channels 3 (RGB), height 299, width 299 (`1x3x299x299`)

This is one 299×299 color image, the standard Inception-v3 input size.

Netron labels it as `X`, but semantically this is the `input_image`.

![Input layer](media/image7.png)

### 1st Convolutional Layer

**Conv**

W (32 filters, each filter is 3×3, operating on 3 input channels (RGB), moving 2 pixels at a time.) This layer extracts 32 kinds of low-level features (edges, corners, color transitions) from the image while shrinking its spatial size.

Its input was `1x3x299x299` and because stride 2 downsamples, the output will be `1,32,149,149`.

The weight Tensor `W(32x3x3x3)` represents: `[out_channels, in_channels, kernel_h, kernel_w]` — so 32 output feature maps, each looks at all 3 RGB channels using a 3×3 window.

**kernel_shape** — filter size e.g. `[3,3]`, or `[1,1]`. `[1,1]` looks at a single pixel, `[3,3]` is a standard local pattern detector, and `[5,5]` is a larger spatial pattern.

So this means: use 3×3 kernel filters, move 2 pixels at a time, and don't pad the edges of the shape.

This is an example `[3,3]` kernel filter that produces one output value per window it slides over.

![Kernel filter example](media/image8.png)

**Stride** = how many pixels the kernel jumps each step.

![Stride visualization](media/image2.png)

## Inception Block/Module

An inception module is a multi-path feature extractor block. Instead of doing one convolution, it does several different ones in parallel, then merges the results. Different kernel sizes see different things — small kernels see fine details like edges/textures, while large kernels see bigger shapes like parts/objects. **Inception TL;DR: "Let's look at the same image in different ways."**

How to identify an inception block:

1 tensor in and 3–4 parallel paths, multiple Conv stacks, then one big Concat node.

![Inception block diagram](media/image10.png)

This is what an inception block looks like in Netron. One tensor feeding into multiple Conv chains, one pooling chain, then a big Concat (`axis=1`). This entire subgraph is the Inception Module.

![Inception block in Netron](media/image1.png)

Each Conv, ReLU, Pool, Concat modifies:

- magnitude
- direction
- spatial structure
- channel coupling

**This is what attackers exploit.**

## Model Stem

The section of Conv layers prior to the Inception module is called **the Stem** of Inception-v3. Its purpose is to rapidly reduce image size, increase channel depth, and extract primitive visual features.

For each Conv node think: **"Take these feature maps, slide these filters of this size with this stride, produce this many new feature maps."**

![Model stem](media/image5.png)

## MaxPool

Looking at the MaxPool node below we see the following node properties:

- **kernel_shape** = `(3, 3)`
- **stride** = `(2, 2)`
- **pads** = `(0, 0, 0, 0)`
- **ceil_mode** = `0`

This means: For every channel independently, the layer slides a 3×3 window across the feature map, moves it 2 pixels at a time, and outputs the **maximum value** in each window. So spatial size is reduced roughly by half. In graph terms, MaxPool is a **many-to-one spatial aggregation node.** It collapses a small neighborhood of activations into a single value per channel. You can think of it as:

- Throwing away exact pixel positions
- Keeping only the strongest local response

Models use MaxPool for **downsampling** (reduce spatial resolution, saves compute, increase receptive field of later layers) and **local invariance** (small shifts in input don't change output much — only the strongest feature in each region survives).

### MaxPool Node & Attributes

![MaxPool node and attributes](media/image9.png)

Looking at this module more closely we see an attribute **ceil_mode** that controls **how the output size of the pooling layer is computed when the window does not fit perfectly at the edge of the feature map**. It only affects **shape calculation**, not the pooling values themselves.

![ceil_mode illustration](media/image6.png)

For example, when sliding a pooling window (here 3×3 with stride 2), the layer must decide: "If the window would partially hang off the edge, do we stop early or include one more step?" That decision is controlled by `ceil_mode`.

### Two Cases for ceil_mode

**ceil_mode = 0** — Use floor division (default behavior). Meaning: Only include windows that fully fit inside the input. So the output size is:

```
output = floor((input_size - kernel_size) / stride) + 1
```

This has the effect of dropping border regions that don't fit cleanly, slightly smaller output, more predictable geometry, and is the most common choice.

**ceil_mode = 1** — Use ceiling division. Meaning: Allow one extra pooling step even if the window partially exceeds the input boundary. Output size:

```
output = ceil((input_size - kernel_size) / stride) + 1
```

This has the effect of keeping more border information, slightly larger output, and sometimes requires implicit padding.

**TL;DR:** `ceil_mode` decides whether pooling rounds output size down (safe & common) or up (keeps partial edge windows).

## Security Analysis

From a security perspective, this part of the DAG (the Stem) provides only *partial* robustness. The downsampling step reduces some high-frequency noise, and the larger receptive field helps aggregate information across neighboring pixels, which can slightly weaken very fine-grained perturbations. However, the core operation is still a linear convolution, so adversarial directions in the input space are largely preserved.

Using stride can even alias adversarial patterns instead of destroying them, and there is no explicit smoothing or denoising to actively remove malicious structure. In effect, the layer compresses the perturbation but does not eliminate it. The following BatchNorm and ReLU stabilize activations by normalizing their scale and clipping negative values, but they also fail to remove adversarial directions — only their magnitude or sign — so the attack signal can continue to propagate through the graph.

Let's now try to connect the DAG structure of Inception modules to concrete security intuitions, followed by a conceptual example of a multi-scale attack.

## Why an Inception Block is a "Multi-Path Attack Surface" in a DAG

In a DAG view, an Inception block looks like this:

> one input node → several parallel transformation subgraphs → a concatenation node

Each branch applies a different type of feature extraction:

- Small-kernel convolutions (local, fine detail)
- Larger-kernel convolutions (coarser shapes / textures)
- Pooling or dimensionality-reduction paths (context / invariances)

These branches are **independent computation paths** that later merge.

From a learning perspective, this is great: the model can represent the same object using multiple "views" of the data (edges, textures, shapes, spatial context).

From a security perspective, this has two important implications.

## Attacker-Side Intuition

### 1. Multiple feature scales can be influenced at once

Different branches respond to different spatial frequencies:

- Small kernels → sensitive to pixel-level or edge-level changes
- Large kernels → sensitive to broader patterns and shapes
- Pooling paths → sensitive to regional averages or layout

A single input perturbation can be constructed so that:

- Part of it affects fine edges
- Part of it affects medium textures
- Part of it affects coarse structure

So instead of "breaking" only one feature extractor, the perturbation nudges **several internal representations simultaneously**.

In essence, one malicious signal fans out into several internal feature streams.

### 2. Gradients flow through multiple independent paths

During backpropagation, gradients do not pass through one bottleneck but through **several parallel routes**.

This means:

- The attacker gets richer gradient information
- The optimization landscape is smoother
- Small updates are more likely to affect the final output

This tends to improve:

- Attack reliability
- Transferability

### 3. The concatenation step preserves all corrupted features

When the branches recombine, the model does not choose one representation — it stacks them.

So if even **two out of four branches** are successfully manipulated, their corrupted features are preserved and passed downstream.

## Defender-Side Intuition

Despite the above, Inception is not weak by default. Weights and adversarial training with its higher kurtosis signal can both obviate any architectural benefit an attacker may gain.

### 1. Feature diversity

The model does not depend on:

- One kernel size
- One texture scale
- One type of pattern

This reduces *single-point failure*. An attack that only targets "edges" or only "textures" may fail.

### 2. Ensemble-like behavior

Each branch acts like a weak specialist model:

- Branch A: edges
- Branch B: blobs
- Branch C: context

Concatenation is similar to ensembling features. This often improves robustness against **narrow attacks**.

## The Resulting Trade-Off

Structurally:

- **Robust to single-scale attacks** (e.g., only pixel noise or only texture manipulation)
- **Vulnerable to multi-scale, multi-branch coordinated attacks**

This is why Inception models often perform well under basic adversarial testing but still fail under stronger adaptive methods.

## Conceptual Example: A Multi-Scale Coordinated Attack

Consider an image classifier distinguishing **cat vs dog**.

An attacker constructs a perturbation with three components:

### Fine scale (small kernels)

Tiny, high-frequency changes around whiskers and fur edges:

- Barely visible pixel-level noise
- Shifts edge detectors in the 1×1 or 3×3 conv branches

Effect: *"edges look more dog-like"*

### Medium scale (mid kernels)

Subtle texture changes over patches of fur:

- Alters repeated patterns
- Affects mid-receptive-field branches

Effect: *"fur texture resembles dog fur statistics"*

### Coarse scale (large kernels / pooling)

Very low-frequency brightness or shading changes over the head region:

- Slight shape bias
- Affects large-kernel or pooling paths

Effect: *"overall head structure trends toward dog"*

### What happens in the Inception block

| Branch | What it sees |
|---|---|
| Small kernel branch | altered edges |
| Medium kernel branch | altered textures |
| Large kernel branch | altered shape/context |
| Pooling branch | altered regional averages |

All of these corrupted features are concatenated. Downstream layers now receive a **coherent false narrative**:

> edges + textures + shape all support "dog"

Even though no single change is strong enough alone, together they may overwhelm the classifier.

## How to Recognize This in a DAG Analysis

When reviewing a model graph, look for:

- Parallel convolution branches
- Different kernel sizes
- Concatenation nodes

Then ask:

- "Has this model been adversarially trained?"
- "Do I have access to model weights for analysis?"
- "Can a perturbation influence multiple branches at once?"
- "Are there any explicit denoising or low-pass filters before concatenation?"
- "Does any path dominate, or do they all contribute similarly?"

If the answers are: *No / No / Yes / No / Similar* — then the architecture is potentially **multi-scale expressive but multi-scale attackable**. You have to run some initial adversarial attacks like FGSM and PGD to get a sense of how robust it is.
