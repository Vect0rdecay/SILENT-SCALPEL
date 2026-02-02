# What Is ONNX, and Why I Started Researching Computational Graphs

In early January 2026, a colleague who had co-authored HiddenLayer's ShadowLogic research mentioned it in passing during a meeting. I went and read his team's blog post and was instantly interested in learning more. ShadowLogic demonstrated something I hadn't thought much about, namely that you can inject backdoors directly into neural network computational graphs without retraining and no weight modification needed. Just... edit the graph.

The backdoor injection was interesting, but something else caught my attention. The computational graphs themselves. This post documents how I started treating neural network computational graphs the way I used to treat disassembly, and more importantly whether structural patterns alone can reveal security weaknesses.

I spent years in offensive security looking at code, malware and disassembled programs. One of the things that always fascinated me was seeing actual code patterns like loops, arrays, structs, values being pushed onto the stack etc rendered in assembly. You could recognize patterns. A `for` loop has a recognizable pattern of mnemonics. A switch statement looks different from a series of if-else blocks. Experienced reverse engineers can glance at a chunk of disassembly and get a feel for what the code is doing.

Reading about ShadowLogic, I found myself wondering: *"Are there similar patterns in ONNX computational graphs for vision models?"*

Not backdoor patterns specifically, ShadowLogic already covered that. I was wondering something different: could you look at a graph and predict what kinds of attacks a vision or audio model might be vulnerable to? Could certain structural patterns indicate susceptibility to specific adversarial inputs?

And if you could identify those patterns... could you *create* them deliberately to help an attack along? My thought was that empirical attacks are expensive, incomplete, and often unguided. Graph analysis might inform which attacks are worth running, not necessarily whether attacks should be run at all. But more on this in other posts as we progress.

Before I could answer any of those questions, I needed to understand the ONNX spec and how its behavior depends on the runtime itself. This is important for later when we discuss how graph semantics alone are not the same as runtime semantics. What follows is what I learned, written for offensive security practitioners.

---

## What Is ONNX?

ONNX stands for **Open Neural Network Exchange**. It's an open format spec for representing machine learning models. It's a portable, inspectable, modifiable container that can run across different platforms and runtimes.

More specifically, ONNX defines:

1. **A computational graph format** - the operations and how they connect
2. **A set of standard operators** - Conv, MatMul, Relu, Softmax, etc.
3. **A way to store model weights** (more on this below)

When you export a PyTorch or TensorFlow model to ONNX, you're converting from the framework's internal representation into this standardized intermediate representation (IR). Different runtimes like ONNX Runtime, TensorRT, OpenVINO, CoreML can then execute it.

**ONNX files are just Protocol Buffers.** You can parse them, inspect them, modify them, and save them back out. ONNX has no mandatory signing or integrity enforcement. This is what makes ShadowLogic possible. And it's what made my research possible too. Some pipelines may wrap ONNX in signed containers or checksums, but the format itself does not enforce trust. Anyone with write access to an ONNX file can modify it. Below is an image showing the Wav2Vec2 audio model graph taken from Netron.
<img width="1501" height="722" alt="image" src="https://github.com/user-attachments/assets/24a24dc5-8d06-4f9f-862f-a0f1364abe14" />



---

## What ONNX Files Contain by Default

**Core Structure:** An ONNX file (`.onnx`) is essentially a serialized Protocol Buffers (protobuf) message based on the ONNX schema. It includes:

- **The model's computational graph** - nodes representing operations, inputs, outputs, edges representing data flow
- **Metadata** - model name, producer (PyTorch, TensorFlow, etc.), ONNX version
- **Input/output specifications** - what the model expects and produces
- **And crucially, the model's parameters** - weights, biases, and other constants, stored as tensors in the `initializer` field of the `GraphProto`

**Weights Are Included:** In standard usage, when you export a model to ONNX (e.g., from PyTorch, TensorFlow, or other frameworks), the weights are embedded directly in the `.onnx` file. This makes the file self-contained and portable. For small to medium-sized models, this is the norm. I've seen and worked with countless ONNX files where inspecting them (e.g., via `onnx.load()` in Python) shows the weights right there as raw tensor data.

Here's the basic intuition for loading and inspecting one:

```python
import onnx

model = onnx.load("resnet50.onnx")

# Basic info
print(model.producer_name)      # "pytorch"
print(len(model.graph.node))    # 176 nodes (operations)
print(len(model.graph.initializer))  # 161 initializers (weights)

# What operators does it use?
op_types = set(node.op_type for node in model.graph.node)
# {'Conv', 'BatchNormalization', 'Relu', 'MaxPool', 'Add', 'Gemm', ...}
```

Those nodes are operations—convolutions, activations, pooling. The initializers are the learned weights. The graph structure defines how data flows through them.

For a ResNet-50, you'd see operators like `Conv`, `BatchNormalization`, `Relu`, `MaxPool`, `Add`, `GlobalAveragePool`, and `Gemm` (general matrix multiply, used for the final fully-connected layer).

---

## The Confusion About Weights

This is something that confused me when I started researching, and I've seen it maybe confuse others. Some experienced people told me ONNX files don't contain weights. They said the format is just for representing the computational graph—the architecture—and weights are stored separately. This is partially true, but somewhat misleading for most of the vision models I looked at. Let me explain.

### The Protobuf Size Limit and External Data

**The 2GB Constraint:** Protobuf messages have a practical upper limit of around 2GB (technically, the default max is 64MB in some implementations, but libraries like ONNX Runtime allow up to 2GB). This becomes a bottleneck for large models (e.g., modern transformers like GPT variants or large vision models) where weights alone can exceed several GB.

**How ONNX Handles It:** To work around this, ONNX supports external data storage for tensors. In this mode:

- The `.onnx` file still contains the full graph structure and metadata
- But large tensors (weights) are offloaded to separate external files (often `.bin` or `.data` files in the same directory)
- The `.onnx` file includes references to these external files via fields like `external_data` in the `TensorProto`, specifying the file path, offset, and length
- When loading the model (e.g., in ONNX Runtime), the runtime pulls in the external data automatically

**Resulting Variations:** This is why "some ONNX files have weights and some don't":

- For models under ~2GB total size, weights are **inline**, fully stored in the `.onnx` file
- For larger models, weights are **external**, so if you just look at the `.onnx` file in isolation (e.g., its file size or a hex dump), it might seem like it lacks weights. But it's not that the format can't store them; it's a deliberate split to avoid protobuf limitations
- Additionally, it's possible to export an ONNX file without weights intentionally, e.g., for sharing just the graph topology for inference with custom-initialized parameters. But this isn't the default behavior in most export tools

### Evidence from Practice and Tools

- If you load an ONNX model in Python using the `onnx` library, you can check `model.graph.initializer` and it'll list the tensors if they're inline. For external ones, you'll see placeholders with external references
- Tools like Netron will show the graph and indicate if data is external
- Official ONNX docs confirm this: The format is designed to include weights, but external storage is an optional extension for scalability

If you've primarily worked with LLMs in ONNX format, you may have only ever seen models with external weights. The `.onnx` file looks "empty" because the weights are elsewhere. But for vision models, audio models, and smaller architectures, the weights are typically embedded directly in the file from what I could see during my experiments. The following figure shows how to check:

```python
# Pseudocode for checking weight location
model = onnx.load("your_model.onnx")

embedded_count = len(model.graph.initializer)
external_count = count_external_references(model)

# Vision models: embedded_count high, external_count zero
# LLMs: embedded_count low or zero, external_count high
```

For my research, this matters because if weights are embedded, I can analyze them directly alongside the graph structure. If they're external, graph-only analysis is still possible but weight analysis requires both access to and loading the external files.

---

## How ONNX Files Differ by Model Type

Not all ONNX graphs look the same. The computational graphs for different types of models have dramatically different structures, sizes, and operator patterns. Understanding these differences turned out to be important for my research.

### Vision Models (CNNs like ResNet, VGG, Inception)

Typical characteristics:

- **Size:** 25MB to 500MB (weights embedded)
- **Node count:** 50-500 nodes
- **Dominant operators:** `Conv`, `BatchNormalization`, `Relu`, `MaxPool`, `AveragePool`, `Add`
- **Structure:** Deep sequential chains with skip connections (ResNet) or branches (Inception)

**What the graph looks like:** Relatively linear flow, predictable patterns, easy to visualize. Heavy Conv→BN→Relu→Pool repeating blocks.

### Vision Transformers (ViT, DeiT, Swin)

Typical characteristics:

- **Size:** 100MB to 1GB (weights usually embedded)
- **Node count:** 500-2000 nodes
- **Dominant operators:** `MatMul`, `Softmax`, `LayerNormalization`, `Reshape`, `Transpose`, `Add`
- **Structure:** Patch embedding → N × (Attention + MLP) → Classification head

**What the graph looks like:** Much more complex than CNNs. The attention mechanism creates distinctive patterns with Q/K/V projections and multi-head concatenation.

### Large Language Models (Llama, Phi, etc.)

Typical characteristics:

- **Size:** 1GB to 100GB+ (weights almost always external)
- **Node count:** 1000-10000+ nodes
- **Dominant operators:** `MatMul`, `Softmax`, `LayerNormalization`, `Reshape`, `Gather`, `Where`
- **Structure:** Embedding → N × (Self-Attention + FFN) → Output projection

**What the graph looks like:** The graph file itself may be small because weights are external. Similar to ViT but much deeper, with KV cache handling for generation.

### Audio Models (Whisper, Wav2Vec)

Typical characteristics:

- **Size:** 100MB to 2GB
- **Node count:** 500-3000 nodes
- **Dominant operators:** `Conv` (1D kernels), `LSTM`/`GRU` (older models), `MatMul`, `Softmax`
- **Structure:** Audio encoder (often with spectrogram processing) → Transformer → Decoder

**What the graph looks like:** Heavy use of 1D convolutions for temporal processing. Whisper-style encoder-decoder architectures.

> **Note:** For Whisper / encoder-decoder models, exported ONNX graphs often bake in inference-only paths, which affects node count and structure.

### Summary Table

| Model Type | Typical Size | Weights Location | Key Operators |
|------------|--------------|------------------|---------------|
| Vision CNN | 25–500MB | Embedded | Conv, BN, ReLU, Pool |
| Vision Transformer | 100MB–1GB | Embedded | MatMul, Softmax, LayerNorm |
| Large Language Model | 1–100GB+ | External | MatMul, Softmax, Gather |
| Audio Model | 100MB–2GB | Mixed | Conv1D, LSTM, MatMul |

---

## What I Was Actually Wondering

At this point I understood the format, including its use of Protocol Buffers for serialization and the `GraphProto` structure for representing nodes and tensors. Using tools like Python's `onnx` library, I could load and inspect ONNX files, parsing elements such as operator types, input/output shapes, and initializer tensors. This revealed different structural patterns across model types, like recurring sequences of Conv2D, BatchNormalization, and ReLU ops in CNNs or attention mechanisms in transformers. But here's what I kept coming back to: *"Can these structural patterns predict anything useful about security?"*

In adversarial ML, different attack types exploit different properties:

- **FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent)** — gradient-based white-box attacks primarily for vision models, push inputs across decision boundaries by computing perturbations via backpropagation and iteratively refining them within an epsilon ball (e.g., L∞ norm), often exploiting linearities in convolutional layers

- **Optimization-based attacks like Carlini & Wagner (C&W) or DeepFool** — applicable to both vision and audio models, use iterative solvers (e.g., L2 or L∞ optimization) to craft minimal perturbations that evade detection, targeting the model's loss landscape and vulnerabilities in activation functions or spectrogram processing in audio

- **Patch attacks** — for vision models, exploit how models aggregate spatial information through pooling or attention, often via localized adversarial stickers or digital patches that dominate feature maps in CNNs to fool object detection or classification

- **Physical-world attacks** — robust to real-world distortions in vision (e.g., aliasing from camera sensors, affine transformations like rotation/scaling) and audio (e.g., over-the-air propagation with room acoustics or ultrasonic signals), exploit environmental artifacts to create perturbations that survive printing, playback, or sensor capture

- **Transfer attacks** — exploit shared architectural patterns (e.g., residual blocks in vision or temporal convolutions in audio) in black-box settings, where perturbations crafted on surrogate models transfer to targets due to similar feature representations or waveform sensitivities

- **Universal adversarial perturbations** — data-agnostic for vision (e.g., image-wide noise fooling multiple inputs) and audio (e.g., persistent waveform distortions across clips), target global model vulnerabilities like overparameterized encoders or frequency-domain filters

- **Backdoor/poisoning attacks** — during training, inserting triggers like visual patterns (e.g., specific pixels) or audio signatures (e.g., inaudible tones) to activate misbehavior at inference, exploiting fusion layers or Mel-spectrogram transforms in the graph

- **Imperceptible audio attacks** — (e.g., psychoacoustic masking or frequency-domain perturbations), often using genetic algorithms or hidden commands to fool speech recognition models while remaining inaudible or subtle to humans, targeting temporal dependencies in RNNs or transformers

So, I became curious and wanted to know, as one example, if a model has a **global average pooling (GAP) layer**, which reduces spatial dimensions (e.g., from H×W×C to 1×1×C) by computing the mean across height (H) and width (W) for each channel (C), feeding directly into a fully-connected head, does that make it more susceptible to patch attacks? The pooling aggregates spatial information over the entire feature map, creating a large effective receptive field (often the full input size in later layers), which can let a small adversarial patch with high-norm features (e.g., optimized via Lp-bounded perturbations) dominate the averaged signal and mislead classification. Empirical studies show this vulnerability increases in models with expansive receptive fields, though it can be mitigated by defenses like small receptive field masking or clipping.

Or, if a model uses **max pooling instead of average pooling** after a fusion layer—typically with a 2×2 kernel and stride 2, selecting the maximum value in each window—does that amplify adversarial perturbations? Max pooling propagates the most extreme (potentially perturbed) value forward, preserving high-magnitude noise (e.g., under L∞ attacks) and potentially exacerbating error rates, whereas average pooling computes the mean, which can smooth and dilute perturbations across the window (reducing their L2 norm impact), theoretically enhancing robustness. This seems to hold for vision models (e.g., in CNN feature maps) and extends to audio, where max pooling on spectrograms might amplify frequency-domain attacks like hidden perturbations, while average pooling could better handle temporal smoothing in waveforms.

So, I used Claude to gather over 100 published research papers on adversarial attacks and adversarial robustness and analyze them looking for connections and patterns between characteristics and robustness or susceptibility to specific types of attack.

From there, I started building a tool to detect these patterns that I eventually called **"gadgets,"** borrowing the term from binary exploitation. In binaries, a gadget is a short instruction sequence (e.g., a few opcodes ending in a return) that enables a specific capability, like a ROP chain for control flow hijacking. In ONNX graphs for vision and audio models, I use "gadgets" to mean small, reusable structural motifs—like Conv followed by MaxPool without intermediate normalization (e.g., BatchNorm), or temporal pooling in audio transformers without dropout—that appear empirically correlated with, enable, or in some cases seem to amplify certain attack behaviors, based on graph traversal and motif mining techniques.

The following image shows an example of a sequence of operators as a structural motif, or gadget.

<img width="1267" height="305" alt="image" src="https://github.com/user-attachments/assets/dd68a72a-9da6-418c-9f11-dc205ce82fea" />

---

## Formal Definition: Gadgets

My attempt at a formal definition for gadgets as they pertain to computational graph analysis follows:

> **A gadget is a localized, reusable structural motif in a neural network computational graph (e.g., as represented in ONNX GraphProto) that enables, amplifies, or constrains a class of behaviors—including adversarial behaviors—in a manner that is independent of any specific set of learned weights or initializers.**
>
> A gadget may consist of a single operator (e.g., a MaxPool op with kernel >1), a sequence of operators (e.g., Conv2D → ReLU → GlobalAveragePool), or a combination of operators and their attributes (e.g., stride values or activation types), and may span one or more adjacent subgraphs.
>
> **A gadget does not imply the presence of a vulnerability, a backdoor, or exploitability in isolation;** instead, it describes structural affordances—e.g., what adversarial behaviors are facilitated or made more likely based on empirical correlations from attack simulations, not guaranteed outcomes in practice.

This weight-agnostic focus allowed for static analysis of model architectures, though additional work could integrate dynamic testing to quantify gadget impacts under specific threats as well as other data to create an entire picture of a model's susceptibility to attacks.

---

## Where This All Led

Graph structure does tell you something and in some cases it indicates which attack types are more likely to succeed against a given architecture. A model with certain pooling patterns might be more susceptible to patch attacks. A model with aliasing-inducing downsampling might be more vulnerable to physical-world attacks. While architectural differences are well known at a high level (e.g., CNN vs Transformer), this work focuses on sub-architectural motifs that recur across families and can be identified, compared, and reasoned about at the graph level.

**But I also discovered something I didn't expect:** graph structure alone is not a singular indicator of how robust a model actually is, which is what I had hoped would be possible. Two models with identical graphs but different training procedures—and different weights—had wildly different actual robustness. The structure shows the attack surface and what's possible, not whether attacks will actually succeed.

### What This Means

- You **cannot** determine robustness from structure alone
- You **cannot** replace empirical attacks with static analysis
- You **cannot** reliably detect all backdoors by visual inspection alone

That distinction of **attack surface vs. actual vulnerability** turned out to be crucial. And it led me somewhere I didn't anticipate. More on that in the next post though.

---

## What's Next

In Post 2, I'll walk through the gadget taxonomy I built, the patterns I thought would matter, how I tested them, and the surprising results that made me realize I was only seeing part of the picture.

---

## Key Takeaways

1. **ONNX is Protocol Buffers under the hood** — no integrity protection, fully inspectable and editable

2. **Many vision models appear to embed weights** — I suspect the "ONNX has no weights" I kept hearing from others comes from their focus on larger LLM graphs with external data

3. **Different model types have different graph signatures** — and they can look very different

4. **My research question was: "Can graph patterns predict vulnerability to specific adversarial attack types?"** — The short answer is that graph structure indicates potential attack surface (what's possible), but not actual robustness (whether attacks succeed). Robustness is an empirical property; the attack surface is a structural one

5. **Pre-deployment review moves from "does it work?" to "what kinds of attacks are even worth trying?"** — Red-team scoping and engagements are hypothesis-driven and not simply brute force model attacks

---

## About This Series

**Project Silent Scalpel** is a series documenting my research journey into computational graphs starting with ONNX, my thought process, my failures, the iterations, and the unexpected discoveries.

*This research was done independently outside of work on personal time, using all my own hardware, software and AI-assisted development R&D resources. Any conclusions or views expressed herein are my own and do not reflect those of my employer.*
