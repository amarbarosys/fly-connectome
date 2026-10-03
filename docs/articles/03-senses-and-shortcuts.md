# Sugar reaches the fly's mouth but never its teacher

*We mapped what each sense does inside the brain, and three attempts to make taste teach the fly all returned zero.*

We gave the brain a taste of sugar and turned up everything we could think of. Hunger gain on, sugar neurons at triple strength, the input to the reward neurons at triple strength. The motor neuron that moves the fly's mouthpart responded: its rate rose from 25.8 to 35.9 spikes a second as hunger rose. The reward neurons that teach the mushroom body fired zero spikes at every hunger level.

In this model sugar can make the fly eat. It cannot make the fly learn. This article is about how we found that out, and about the senses we mapped on the way.

## A preset is a recipe for one sense

Teaching needs a clean way to say "now the fly smells this" or "now it sees that." We built **presets**. A preset is a recipe: switch on one type of sensory cell, on one side of the head, and note which cells downstream respond.

The process is plain. For every sensory cell type with at least eight cells on each side, we drive it alone for 200 milliseconds and repeat the run eight times. A downstream cell counts as responding if its spike count is far above what silence gives (a z-score above 4, so one lucky spike does not count). A stimulus that makes anything respond becomes a candidate preset. Then comes the part that matters: we replay it with a fresh random seed. A preset is verified only if the replay gives nearly the same pattern (a similarity score of at least 0.9, where 1 means identical) and the same strongest cell. A preset that does not reproduce is noise, not a skill.

| Sense | Stimuli tried | Fired something | Verified on replay |
|---|---:|---:|---:|
| Vision | 154 | 95 | 87 |
| Taste | 39 | 21 | 16 |
| Touch, humidity, temperature, ocelli | 130 | 97 | 71 |
| Motion | 40 | 3 | 3 |
| Nerve cord reporting up | 74 | 28 | 24 |
| Smell | 106 | 58 | 52 |

Some of these have a satisfying shape. Two neuron types that detect looming objects, LC4 and LPLC2, drive a descending neuron called DNp01 on the same side. (Descending neurons carry commands from the brain down toward the body.) A looming object on the left ends up on the left side of the output.

Some are honest disappointments. Of 40 motion-sensing stimuli, only 3 made anything fire. The classic motion cells, called HS and VS, stayed silent under steady drive. Our notes say they need a proper eye model (a tool called flyvis), which is a separate project. For smell, the downstream cells did not reproduce well on replay (similarity about 0.71), so smell presets are verified by the score they produce instead: the approach versus avoid number from the last article, within 0.15.

One more map belongs here. To say which direction a visual cell looks, we placed 3,251 of 4,108 photoreceptors on the eye by letting each one inherit the eye column of the cells it connects to most strongly. The two eyes cover about 125 by 145 degrees. LC4 and LPLC2 cells, placed using only their inputs, landed on the eye on their own side of the head for 100 percent and 97 percent of cells.

## The taste map found the mouth and missed the teacher

```
sugar neurons ------------> mouth motor neuron MN9      fires: 25.8 to 35.9 spikes a second
sugar neurons --2 to 3 hops--> reward neurons (PAM)    fires: 0 spikes, at every hunger level
```

Taste presets were the ones we cared about. In real flies, sugar is a reward: it makes dopamine neurons fire, and those teach the mushroom body to like the smell that came before the sugar. Do the sugar neurons in our model reach the dopamine neurons?

We traced the wiring. Sugar reaches the proboscis motor neuron MN9. It reaches the reward dopamine neurons, called PAM cells, only two or three synapses downstream, and the sugar path supplies from under half a percent to 15 percent of the input a given PAM type receives, depending on the type. At the authors' weights, sugar drove zero PAM spikes.

One neuron type, called LB3b, looked like a bitter taste cell that fed the punishment neurons. In the sweep it drove reward neurons too, 687 to 797 spikes. So it is a broad dopamine driver, not a clean bitter channel. Humidity and temperature presets shared the same punishment hub as bitter. The punishment cells in this model are a meeting point of many senses, not a pain channel.

## Three ways to make hunger help, three zeros

**Attempt one: hunger as gain.** In real flies, a hungry fly values sugar more. We changed three edge groups with hunger (sugar neurons out ×(1 + 2h), the LB3b output ×(1 − h/2), excitatory input into PAM ×(1 + 2h)). Result: zero PAM spikes at hunger 0, 0.5 and 1. We should say plainly that those gain formulas were our guesses.

**Attempt two: a slow route.** The sweet-to-reward route in real flies is reported to run through octopamine, a chemical that works slowly and broadly. In the model every synapse is a fast excitatory one. We counted: the octopamine cells VPM3 and VPM4 connect to PAM through 29 of 109,448 excitatory synapses, 0.03 percent. A route that works by spreading a chemical through tissue is invisible to a synapse count.

**Attempt three: build the slow signal.** We added slow neuromodulation to the engine: dopamine, octopamine and serotonin cells raise a level that fades after their drive stops. First as extra gain, then as a slow current. Both are verified to work (the level outlives its drive: the target cells sit at 1.13 times baseline just after release and are back to 0.99 at 1.5 seconds, and a planted bug is caught). Sugar to PAM stayed at zero. Only 53 synapses run from octopamine cells to PAM, from 8 cells, and those cells fire about 13 times a second in total. Gain cannot create firing from nothing.

An option remains: release the modulator over a whole region at once, the way real tissue does. We have not built it.

## So the teacher is supplied by hand

In every learning experiment so far, the reward and punishment signals are given directly, when the fly bites food or touches a trap. The sensory side is real wiring. The teacher is ours. That is the main scope limit on the last article.

## What this does and does not show

It shows that, in this published model and at its weights, there is no fast synaptic path that lets sugar teach the mushroom body, and that three ways of adding hunger did not create one. It does not show that real flies lack the path. The model leaves out slow chemical signalling, and we cannot say whether the real route would work, because we did not build it.

The presets are also only as good as the stimulus they test. Our drive is a steady 150 spikes a second on one cell type, not a natural signal, and a preset is verified only against its own replay, not against a real fly.

Next: [Building a body](04-building-a-body.md).
