# The real wiring beat scrambled wiring at smell, and at nothing else yet

*We ran a whole fruit fly brain on a home graphics card and asked the first honest question: does the actual wiring matter?*

Our first big test said no. We took the real wiring of a fruit fly brain, scrambled it, and asked both versions to steer a simulated fly toward food. The scrambled brains did as well as the real one. In one batch of runs the real brain scored 56 out of 64, and its scrambled twins scored between 51 and 61.

That result was fair. It was also only half the story. This article covers the other half: the one test where the real wiring won by a wide margin, and the three where it did not.

## The map tells you who is wired to whom, not what it does

A **neuron** is a brain cell that sends tiny electrical pulses, called spikes, to other cells. A **synapse** is the contact point where one neuron passes a pulse to the next. Think of neurons as houses and synapses as the phone lines between them.

A **connectome** is the complete phone book: every house, and every line between every pair. For the fruit fly, scientists sliced a brain into ultra-thin layers, photographed each layer under an electron microscope, and traced every cell through the stack. The map we use is called FlyWire. It holds about 139,000 neurons and about 10.5 million connections. The version the model reads holds 138,639 neurons.

A map is not a brain. It says who is wired to whom and how strongly. It does not say what the brain does. To find that out, you have to run it.

## Running the map on a gaming card, matched to the authors' numbers

A **simulation** is a program that imitates something real, step by step, so you can watch it without building it. Ours imitates the fly brain.

We use a published model from a 2024 paper by Shiu and colleagues. Each neuron in it is simple. It collects pulses from its neighbours like a bucket collecting rain. When the bucket is full, the neuron fires a spike and the bucket empties. After firing, it rests for a moment. Now do that for 138,639 buckets, thousands of times for every simulated second.

Every bucket follows the same rule, which makes this a job for a **GPU**, the chip in a graphics card. A normal processor is a few expert clerks. A GPU is a stadium full of clerks who each know one trick. Our card is an AMD RX 7900 XTX, the kind sold for gaming PCs. We wrote the program in a language called Mojo.

Before we trusted it, we checked it against the authors. We fed 21 sugar-sensing neurons a steady signal for one second and counted spikes: 16,984 from our version, 16,760 from the authors' reference. The GPU copy also had to agree with our slower processor copy, cell by cell. Five one-second runs take 2.5 seconds on the GPU and 15 on the processor.

One early mistake is worth telling. We had read a rule in the authors' code as "a neuron that just fired keeps collecting pulses while it rests." Our network fired 35 percent too much. The reference runs showed the real rule: a resting neuron ignores incoming pulses. Reading the equations was not enough. The published outputs caught it.

## Scrambling is only a fair test with the right opponent

How do you test whether wiring matters? Break it and see what gets worse. But some ways of breaking it are unfair.

```
real brain        who connects to whom: as measured
block shuffle     same number of lines between each KIND of cell;
                  only the exact partners are swapped
degree shuffle    each cell keeps how many lines it sends and receives;
                  partners and kinds are mixed freely
random            only the receiving counts are kept
```

The block shuffle is the hard opponent. It keeps the big plan of the brain and scrambles only the fine detail. If the real brain beats it, the fine detail matters.

## Steering to food: the wiring did not help

The first experiment gave a simulated fly a food smell and asked it to find the food. A small trained layer read the brain's output and turned it into steering. On 16 layouts the real brain found food in 15, the degree shuffle in 11, the random brain in 11. It looked like a win.

It was not. On the six layouts where real and shuffled disagreed, the real brain won five. A split that lopsided happens by luck about one time in five. The measure here is the **p-value**: the chance of a gap this big if the two brains were truly equal. Scientists usually want it under 0.05. Ours was 0.219.

So we ran 64 layouts against ten control brains. Every arm scored exactly 56 out of 64, with no layout where any two arms differed. That is too clean. A real null result shows small, scattered differences. Perfect agreement across ten rewirings means something is broken.

It was. The trained layer had the same fingerprint in every arm. All the arms had been running the real brain, because the code that swapped in scrambled wiring never took effect. We withdrew the result, rebuilt the runner so it stops if two arms share a fingerprint, and ran again.

The honest numbers: the real brain scored 56 out of 64. Its five degree-shuffled twins scored 51, 57, 61, 60 and 57. Pooled, the real brain was 1.9 points behind (p = 0.76). It did beat the random brains by 18.75 points (p = 0.0015). The real brain also missed its own pass mark, a 25 point lead over a control trained on shuffled rewards. It got 20.3.

Reading: for steering, the benefit comes from how many lines each cell has, not from which cell is wired to which.

## Smell: the wiring won, in two animals

We changed the question. Skip the walking. Ask only whether the brain can tell smells apart.

Smell enters through sensors in the antennae, passes a relay station, and arrives at the **mushroom body**, a structure with about 5,000 cells called Kenyon cells. Each odour switches on a different pattern of them. The mushroom body is where flies learn smells, so it returns in the next article.

The test: sniff one of six odours, wait 30 thousandths of a second, record which Kenyon cells fired, and ask a simple program to guess the odour. Chance is one in six.

- Real wiring: **0.94**. It names the odour about 19 times in 20.
- Block shuffles: between 0.28 and 0.56.
- Degree shuffles: the odour never reached the mushroom body. At most 0.09 percent of Kenyon cells fired.

Was the real brain just more active? We compared at matched activity: real at 30 milliseconds against block shuffle at 50, with a similar share of cells firing. The real brain still decoded 0.94 against 0.28 to 0.44. Across the 18 trial columns it won 17 on one measure and all 18 on the other. By luck that happens about once in 43,000 tries for the first (p = 0.000023) and once in 130,000 for the second.

Then we replicated it. The Janelia male central nervous system is a second map, of a second animal, the other sex, traced by a different pipeline. It has 165,733 neurons. Same test, same six odours: the real wiring won all 18 columns at 30 ms (p = 0.0000076).

One single-cell fact makes it concrete. We drove the 21 sugar-sensing neurons and watched a motor neuron for the fly's mouthpart, the proboscis. In the real wiring it fired 45 times a second. On all five degree shuffles it fired 0 times a second, though the sugar sensors fired about 105 times a second in both.

## Back to behaviour: still nothing

If smell coding works, a walker built on it should find food better. We ran four maps: a T-junction, an open field with patches, a wind plume, and a maze. Real against block shuffle, 48 trials each:

| Map | Real | Block shuffle | p |
|---|---|---|---|
| T-junction | +2.90 | +2.52 | 0.80 |
| Open field | +1.50 | +1.05 | 0.22 |
| Plume | +0.85 | +0.34 | 0.25 |
| Maze | -2.25 | -2.55 | 0.61 |

No map was significant. The maze was broken: a decoy beside the start trapped every walker, so it measured nothing.

The reason is plain in hindsight. The walker only needed to know roughly which odour was which, and the block shuffle gets that much right. The test could not separate the brains because it did not depend on anything the shuffled brains lost.

A side probe found one behavioural fingerprint. A looming shape on one side lit up four mirrored pairs of descending neurons, the cells that carry commands from the brain down toward the body (DNp04, DNp02, DNp05, DNp11, the known escape cells), only in the real brain. In block shuffles, 0 to 2 such cells moved. A steering test built on that is still on the list.

## What this does and does not show

It shows that real wiring carries smell identity that two kinds of scrambled wiring lose, in two animals. It does not show that a fly behaves better because of it. The 18 columns are input-jitter samples on one graph per arm, not 18 animals. And a model brain that decodes smells is not a fly that smells.

What would prove us wrong is a behaviour test the block shuffle cannot pass. We now know what it needs: it must depend on timing, on telling similar odours apart, or on learning. The next article takes the third.

Next: [Teaching a fly](02-teaching-a-fly.md).
