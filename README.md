Project Silent Scalpel is a series documenting my research journey into computational graphs starting with ONNX, my thought process, my failures, the iterations, and the unexpected discoveries.

This research was done independently outside of work on personal time, using all my own hardware, software and AI-assisted development R&D resources. Any conclusions or views expressed herein are my own and do not reflect those of my employer.

In early January 2026, a colleague who had co-authored HiddenLayer's ShadowLogic research mentioned it in passing during a meeting. I went and read his team's blog post and was instantly interested in learning more. ShadowLogic demonstrated something I hadn't thought much about, namely that you can inject backdoors directly into neural network computational graphs without retraining and no weight modification needed. Just... edit the graph.

The backdoor injection was interesting, but something else caught my attention. The computational graphs themselves. This post documents how I started treating neural network computational graphs the way I used to treat disassembly, and more importantly whether structural patterns alone can reveal security weaknesses.

I spent years in offensive security looking at code, malware and disassembled programs. One of the things that always fascinated me was seeing actual code patterns like loops, arrays, structs, values being pushed onto the stack etc rendered in assembly. You could recognize patterns. A for loop has a recognizable pattern of mnemonics. A switch statement looks different from a series of if-else blocks. Experienced reverse engineers can glance at a chunk of disassembly and get a feel for what the code is doing.

Reading about ShadowLogic, I found myself wondering: "Are there similar patterns in ONNX computational graphs for vision models?"

Not backdoor patterns specifically, ShadowLogic already covered that. I was wondering something different: could you look at a graph and predict what kinds of attacks a vision or audio model might be vulnerable to? Could certain structural patterns indicate susceptibility to specific adversarial inputs?

And if you could identify those patterns... could you create them deliberately to help an attack along? My thought was that empirical attacks are expensive, incomplete, and often unguided. Graph analysis might inform which attacks are worth running, not necessarily whether attacks should be run at all. 
