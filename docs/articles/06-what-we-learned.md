# Every result we trust first survived a check that could have failed

*Our most useful discoveries were our own mistakes, and the habits that caught them.*

This project is four days old. The first commit is from 30 September 2026, and there are 204 commits at the time of writing, 126 of them from 3 October. At that speed you do not get to be right the first time. You get to be caught quickly.

Here are the mistakes that cost the most, what exposed each one, and what we are still missing.

## Six results that looked like wins and were not

| What looked true | What was true | What exposed it |
|---|---|---|
| The scaled steering run showed no difference at all: 56/64 for the real brain and for ten controls | Every arm had run the real brain. The trained layer had one identical fingerprint in all arms | Zero layouts where any two arms differed. Real noise is never that clean |
| A pass mark: fewer than 10 percent of Kenyon cells active over 40 ticks | Our method capped each tick at 2 percent but rotated which cells, so the total over time was 12 to 25 percent at every setting. The mark could not be met | A sweep tool that reports whether each pass mark is reachable at all |
| Our brain port ran 35 percent hot | We had misread when a resting neuron accepts input. It accepts none | The authors' own reference outputs, not their equations |
| The pain loop passed at p = 0.000000011 | Credit for every punishment went to the wrong smell, which happened to be the trap | Asking what the "last sniff" actually was |
| "Worker crashes silently on trial 4" | Mojo's `input()` opens a copy of the input pipe on every call and never closes it: 402 copies after 400 sniffs, then "Too many open files". Mojo prints unhandled errors to the standard output, which our protocol swallowed | Making the worker's own error text visible, then counting open files |
| "The Kenyon cells have no feedback inhibition at all" | 588 other inhibitory cells connect to them, and the main one fires in 135 of 162 grid points | Counting both from the graph and the receipt before rewriting the sentence |

The third and last are the pattern: reading is not checking. The fourth is the one to remember. A tiny p-value is the most seductive number there is, because it feels like proof. This one was a measurement of the wrong thing.

One quieter correction: mirrored layouts are not independent trials, so our p-values over 64 layouts were really over 32 scenes. The pooled number moved from 0.699 to 0.7551. The conclusion stood. We published the correction beside the old number.

## Four habits caught most of it

**Gates with numbers, written before the run.** Every stage of the engine has a bar, such as 1e-5 of the field's own size, and some bars are derived from measured rounding floors rather than hope. A gate that has no number is not a gate.

**Planted bugs.** Before a check counts, we break something it should notice (a sign, a dropped term, the wrong reference frame) and watch it fail. A check that cannot fail told us nothing; three of our "stand" gates were exactly that.

**Controls, both kinds.** A positive control that must pass (flat ground) and a negative that must fail (an upside-down fly). Learner flies get paired twins with the teacher switched off. Training gets a sham that depresses cells active for neither odour. The odour-absent control produced zero output neurons, zero Kenyon cells and zero descending spikes, so the activity we measured was driven by the smell.

**Receipts.** Every number in the capability table points to a file and a command. Each row is marked PASS, FAIL, UNVERIFIED or a similar label, with the check named beside it. We use UNVERIFIED a lot. It means: no check has been run that exercises the thing the way someone would meet it.

## What is still open

Written plainly, because an honest list is more useful than a confident one.

- **Wiring and behaviour.** Real wiring beats scrambled wiring at smell identity, in two animals. On steering, four maps, and the closed loop, it does not beat block shuffles. The behavioural test that the shuffles cannot pass is not built.
- **Learning in a body.** The fly that learned lives on a flat map. The physics body walks, but the brain is not yet attached to it. The plan puts a gait generator first, then the learned fly's brain, then a learned readout.
- **A sugar teacher.** The model's synapses will not carry sugar to the reward neurons. Reward is given by hand. Volume release, a slow chemical signal spread over a region, is not built.
- **Live speed.** The world runs at 5 to 10 frames a second, not the 25 we want, with a fixed camera.
- **Engine gaps.** 440 box and ellipsoid contacts are unported. The spider's contact force is off against MuJoCo by 1.5e-05 to 5.2e-05 against a 1e-05 bar, cause unknown.
- **Biology.** Passing our checks shows the software does what we said. It does not show a real fly would. Our wet-lab oracle is rated UNVERIFIED. Replay of recorded calcium activity does not carry brain state across windows. A dim-stripe vision response peaks at 0.139 where a spike needs 1. In an insecticide test against published larval data, the real larvae moved less while the model's descending activity went up.

## What comes next

Three things are queued. Attach the brain to the body so the learned fly walks on legs in the world. Run a reversal test, where the fly learns food is on one side and then has to relearn it, for real, shuffled and block brains. Make the world run fast enough to watch. For the reversal we will choose the exact task first, so that the shuffled brains have something to lose, before spending more GPU time.

## The claim that would prove us wrong

If someone builds a task in which a block-shuffled fly brain performs as well as the real one at learning, timing and telling similar smells apart, then the wiring has only mattered for coding, and that is a smaller result than the first article suggested. We have not built that task. We would like to see it built.

Back to the series: [index](README.md).
