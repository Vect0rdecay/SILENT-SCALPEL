# What Predicts Vulnerability? The Hypothesis That Failed and What Replaced It

**Project Silent Scalpel, Post 3**

*In Post 1 of this series, I introduced ONNX computational graphs and the idea that structural patterns might indicate attack surface. In Post 2, I built a taxonomy of 20 gadgets and showed how they combine into chains that map to attack classes. Both posts ended with the same caveat: structure shows what's possible, not whether attacks will succeed.*

*This post is about what happened when I tried to close that gap.*

---

## Hypothesis

After building the gadget taxonomy, I then used that info build a structural scanner that could look at an ONNX model and tell me things like: "This model has CHAIN-PATCH-ATTACK-SURFACE active, it uses global average pooling with no spatial attention, so patch attacks are architecturally feasible." That's useful for scoping a red team engagement because it tells you where to start.

But I still wanted to know if a gadget score of some sort could predict how robust a model actually is. If two models both have the same active gadgets or chains, but one was adversarially trained and the other wasn't, they have identical graph structures but wildly different robustness. The graph alone can't see that difference. I knew this from the literature, but I wanted to test it empirically. Maybe there was something in the graph I was missing, some subtle structural signal that correlated with robustness even after controlling for architecture. To find out, I knew I had to start testing so up a proper validation study.

---

## Testing Different Models

I pulled 12 CIFAR-10 models from [RobustBench](https://robustbench.github.io/), a standardized benchmark for adversarial robustness. The models spanned different architectures and training procedures: standard training, various forms of adversarial training, pre-training strategies, and more.

For each model, I ran my scanner to compute a graph-based "vulnerability score," an aggregate of which gadgets were present, which chains were active, and how many attack paths the architecture exposed. I also computed several other metrics: weight statistics, gradient properties, and dynamic sensitivity (how much the model's output changed under random perturbation).

Then I ran real adversarial attacks against all of them: FGSM, PGD-10, and PGD-20 with epsilon = 8/255 (L-infinity). The point was to correlate scanner metrics with actual attack success rates and see if the graph score had any predictive value.

---

## Failure - No Correlation

The graph score showed essentially no correlation with robustness.

| Metric | Correlation with Attack Success |
|--------|-------------------------------|
| Graph Score | ρ = -0.26, p > 0.05 (not significant) |
| Graph Normalized | ρ = 0.50, p > 0.05 (not significant) |
| Dynamic Sensitivity | ρ ≈ 0 (no correlation) |

The graph score was particularly disappointing. Two Wide ResNet-28-10 (WRN-28-10) models with identical architecture and identical graphs had completely different robustness: one at 0% robust accuracy (standard trained), the other at 60% (adversarially trained). The scanner gave them the same graph score. Of course it did. They have the same graph.

An earlier, smaller pilot study (n=7) had shown a promising correlation (ρ = -0.89, p = 0.007), but expanding to n=12 and controlling for architecture killed it. The original correlation was a statistical artifact driven by the confound between "has adversarial training" and "has different architecture," not a genuine structural signal.

I spent time looking for subtler structural signals, like differences in normalization layer parameters or patterns in the graph topology that might correlate with training procedure, but none of it held up.

The graph predicts attack surface. It does not predict robustness and now I had more data to back it up definitively.

---

## Investigation Continued

The graph score was a dead end for robustness prediction, but the failure itself was informative. It told me exactly what was missing: **information about the weights.**

Two models with identical graphs but different training procedures differ in one and only one thing: their learned parameters. If training procedure determines robustness (which the literature strongly supports), then the signal has to be in the weights. The question was: which property of the weights carries that signal?

I started measuring everything I could extract from weight tensors without running inference:

- **L2 norms**: total magnitude of weight vectors
- **Sparsity**: fraction of near-zero weights
- **Spectral norms**: largest singular values
- **Weight distribution statistics**: mean, variance, skewness, kurtosis

Most of these showed either no correlation or weak, inconsistent patterns. Norms varied a lot across layers but didn't sort cleanly by robustness. Sparsity was similar across models. Spectral norms showed some interesting trends but nothing statistically significant. Kurtosis was the last thing on the list.

---

## Kurtosis Discovery

Kurtosis measures how "peaked" or "heavy-tailed" a distribution is compared to a normal distribution. A normal distribution has kurtosis of 3 (or excess kurtosis of 0). Higher kurtosis means heavier tails, meaning more extreme values.

I computed the average kurtosis across all weight tensors for each model, and the separation between standard-trained and adversarially-trained models was immediately obvious.

**Within-architecture comparison (WRN-28-10, same graph):**

| Model | Training | Robust Accuracy | Weight Kurtosis |
|-------|----------|-----------------|-----------------|
| Standard | None | 0.00% | **4.24** |
| Wang2020Improving | RST-AWP | 56.29% | 9.89 |
| Carmon2019Unlabeled | RST | 59.53% | 10.83 |
| Wu2020Adversarial_extra | AT+RST | 60.04% | 12.06 |
| Sridhar2021Robust | SCORE | 55.54% | 12.42 |
| Hendrycks2019Using | Pre-training | 54.92% | 17.59 |

The standard-trained model had a kurtosis of 4.24. Every adversarially-trained model had kurtosis between 9.89 and 17.59. **No overlap whatsoever.** Any threshold between 5 and 9 achieved perfect separation.

The correlation with actual attack success rates was equally clear:

| Metric | Correlation with FGSM Success |
|--------|-------------------------------|
| **Kurtosis** | **ρ = -0.90, p = 0.015** |
| Graph Score | ρ = -0.26 (not significant) |
| Dynamic Sensitivity | ρ ≈ 0.06 (not significant) |

Kurtosis was the only metric that reached statistical significance. Higher kurtosis meant lower attack success, meaning the model was more robust. This makes intuitive sense: adversarial training via projected gradient descent (as described by Madry et al.) forces the model to optimize over worst-case perturbations, which reshapes the loss landscape and pushes weight distributions toward heavier tails. The kurtosis signature is a *fingerprint of the training procedure* embedded in the weights.

---

## Limitations

The kurtosis finding is real and statistically significant but not breaking new ground except when used in conjunction with the other two methods. It also has a hard ceiling.

**What kurtosis can do:**
- Distinguish adversarially-trained models from standard-trained models with high confidence
- Serve as a binary classifier: "Was this model likely hardened?" Yes or no.

**What kurtosis cannot do:**
- Rank robustness among adversarially-trained models
- Predict the *degree* of robustness

Among the five AT models with the same architecture:

| Model | Kurtosis | Robust Accuracy |
|-------|----------|-----------------|
| Hendrycks2019Using | 17.59 | 54.92% |
| Sridhar2021Robust | 12.42 | 55.54% |
| Wu2020Adversarial_extra | 12.06 | 60.04% |

The model with the *highest* kurtosis (17.59) is the *least* robust among the AT group (54.92%). Within adversarially-trained models, the correlation drops to ρ = -0.09, which is essentially zero. Kurtosis separates the two groups cleanly but tells you nothing about ranking within each group.

It's easy to oversell this, so I want to be explicit. Kurtosis is a strong binary signal, not a precision instrument. If someone handed me an unknown model, I could tell them with high confidence whether it had been adversarially trained. I could not tell them whether it was 55% or 60% robust without actually running attacks.

---

## Combined Approach

The failure of the graph score and the limitation of kurtosis pointed toward a combined approach. Neither metric alone tells the whole story, but together they cover different dimensions:

- **Graph analysis** tells you *what attacks are architecturally enabled* (the attack surface)
- **Kurtosis** tells you *whether the model has been hardened* (the training signal)
- **Quick empirical validation** (FGSM on CPU, seconds per batch) tells you *how well defenses hold up*

I built a combined scanner that integrates all three. Here's how it classified the models I ran full attacks against:

| Scanner Risk Level | Actual Attack Success (PGD-20) | Assessment |
|-------------------|-------------------------------|------------|
| HIGH | 100% | Correct: standard-trained, wide open |
| MEDIUM | 50–56% | Correct: some hardening, partial resistance |
| LOW | 40–52% | Correct: meaningful hardening present |

Every model was correctly placed in its risk tier. The combined approach caught what neither component could alone: graph analysis identified the *type* of threat, kurtosis identified the *presence* of defenses, and the quick empirical check confirmed the *effectiveness* of those defenses.

---

## What This Taught Me

This was the most useful phase of the research, because the primary finding was that my original hypothesis was wrong. The graph does not predict robustness, and the data shows that clearly.

But the failure was productive. It forced me to look at weight properties, which led to the kurtosis discovery. And the kurtosis limitation forced me toward a combined approach that's more useful than either component alone.

The broader insight is about what static analysis can and cannot do:

**Static graph analysis can:**
- Map the architectural attack surface
- Identify which attack classes are structurally enabled
- Compare architectures for security-relevant differences
- Guide which attacks to prioritize in a red team assessment

**Static weight analysis (kurtosis) can:**
- Detect whether adversarial training was used
- Flag models that are likely unhardened
- Provide a quick triage signal without running attacks

**Neither can:**
- Predict exact robustness levels
- Replace empirical attack testing
- Account for deployment-specific factors (input preprocessing, system-level defenses)
- Rank models within the same hardening class

The combined scanner is a triage tool, not an oracle. It narrows the field, tells you "start here" and "this model probably hasn't been hardened." Confirming actual robustness still requires running attacks, but the combined approach tells you which attacks are worth running and whether defenses are likely present before you spend the compute.

---

## Kept Pulling the Kurtosis Thread

There was one more thing the kurtosis work revealed that I didn't expect. If adversarial training changes weight distributions in a way that's detectable via static analysis, what *else* does it change? And more importantly: if the weights are the difference between a robust and a non-robust model, could you *deliberately modify* weight distributions to degrade or enhance robustness?

That question, whether you could amplify or sabotage a model's vulnerability by modifying its weights after training, led me into territory I hadn't planned for. I'll cover that in Post 4.

---

## Summary

1. **Graph structure does not predict robustness.** Correlation analysis across 12 RobustBench models showed ρ ≈ -0.20 (p = 0.53) between graph score and attack success. Two models with identical graphs can have 0% and 60% robust accuracy depending on training.

2. **Weight kurtosis detects adversarial training.** Average weight kurtosis separates standard-trained models (kurtosis ~4) from adversarially-trained models (kurtosis 10–18) with no overlap. The correlation with FGSM attack success was ρ = -0.90, p = 0.015.

3. **Kurtosis is binary, not continuous.** It tells you whether a model was hardened, not how much. Within adversarially-trained models, ρ = -0.09, essentially zero correlation with robustness.

4. **The combined approach works.** Graph analysis (attack surface) + kurtosis (training signal) + quick empirical validation (confirmation) correctly classified all tested models by risk tier.

5. **Static analysis is a triage tool.** It narrows which attacks to try and whether defenses are likely present. It does not replace empirical testing.

---

*Project Silent Scalpel is a series documenting my research journey into computational graphs starting with ONNX, my thought process, my failures, the iterations, and the unexpected discoveries.*

*This research was done independently outside of work on personal time, using all my own hardware, software and AI-assisted development R&D resources. Any conclusions or views expressed herein are my own and do not reflect those of my employer.*
