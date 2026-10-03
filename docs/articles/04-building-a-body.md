# We rebuilt MuJoCo for the GPU, and trusted it only after planting bugs in it

*A fly body engine that matches the reference to one part in 100,000, one function at a time, and the number-rounding traps that nearly fooled us.*

Our first physics prototype passed its gate while the toy fly was tipped over by 117 degrees. The gate printed PASS because it never measured tilt.

That is the problem with checking physics: a check can pass for the wrong reason. This article is about the engine we built to give the fly a body, and the habit that came out of that first embarrassment: never believe a check until you have watched it fail on purpose.

## A port gives us the rules, and a reference to check them against

A **physics engine** is a program that takes a body's pose and the forces on it and computes the next instant: where each leg goes, how hard a foot pushes the floor. We use **MuJoCo**, a standard engine for robots and animals. The fly body we use, 1.026 milligrams with 44 joint motors and 6 sticky foot pads, already runs in it on the processor.

We first tried the ready-made GPU version, MuJoCo Warp, on our card. It got the physics right after three fixes to its AMD support, then ran slower than six processor workers (70,582 steps a second). We closed that track.

This is a different bet. We port MuJoCo's own functions, one by one, into Mojo, for one fixed kind of body, with one GPU thread per animal. It buys three things. Every rule becomes ours to change: joints, adhesion, a new species. A new animal is a description file, not new engine code. And brain and body can eventually live in one process on one card.

## The method: port one function, compare to the original, break it on purpose

```
MuJoCo (float64, the referee)           our port (float32, on the GPU)
        |                                        |
   64 random body states  --------------------> same 64 states
        |                                        |
        +------ compare every output field ------+
                error relative to the field's own size must be < 0.00001
                           |
        plant one plausible bug: the gate must FAIL
```

We port in order: where every body part is, the mass matrix (a table of how hard each joint is to accelerate), its factorisation, the velocity-dependent forces, the step itself. Each stage is checked on the fly and on a test spider with eight legs and other joint types. Each stage also gets a **planted bug**: a flipped sign on a hinge, the sign of the spinning-body term, a body-frame rotation used as a world-frame one. If the gate does not catch the bug, the gate is broken, and the stage does not count as passed.

| Stage | Fly error | Planted bug caught |
|---|---:|---|
| Where every part is | 7.6e-07 | hinge direction flipped |
| Mass matrix | 1.3e-06 | lever arm changed |
| Its factorisation | 3.0e-06 | elimination sign |
| Velocity forces | 3.2e-07 | spinning-body term sign |
| One full step | 3.6e-06 | body rotation in the wrong frame |

(Errors are fractions of each field's own scale.)

## Speed: fast, then honestly slower

The gate for the first stage was a contact-free step under 20 microseconds (millionths of a second) per animal at 65,536 animals at once. We got 0.23: 4.37 million animal-steps a second against 70,582 for the processor. Then came contact, the hard part. Floor contact multiplied the cost by nine. A fly standing with all motors and sticky pads ran at 0.64 microseconds per animal-step, 22 times the processor, and at 1,024 animals the engine already beat it.

Two findings shaped the rest. MuJoCo finds contact forces by improving a guess again and again, and its stopping rules are tighter than float32 numbers can resolve, so it kept searching, up to fifty evaluations at a time, for improvements that were only noise. Stopping where float32 stops cut a step from 133 to 62 milliseconds. And for factorising the mass matrix, a wave of threads per animal was ten times faster than one thread per animal at 1,024 animals, but twice as slow at 65,536. Big and small batches want different kernels.

Then correctness took its toll. With every body part on the floor, with a floating origin for each animal, grooming contacts and noslip, the benchmark now reads 104.4 milliseconds at 65,536 animals. By our arithmetic that is about 1.6 microseconds per animal-step, roughly nine times the processor. The 22 times was for a fly that was incomplete. We kept the slower, correct number.

## Most of our hard bugs were rounding

Our engine uses 32-bit numbers (float32) for speed. MuJoCo's reference uses 64-bit. Many of our hardest bugs were float32 effects.

**The scale of the thing.** A 1-milligram fly has a mass matrix with entries between 0.000001 and 0.001. A check of "within 1" passes anything. Errors are now measured against each field's own size.

**A force that cannot be matched.** MuJoCo's foot contact is very stiff: the force is about 25 million times the penetration depth. Feed MuJoCo itself a state rounded to float32 and its own answer moves by 2.4e-05. No float32 engine can beat that, however carefully written. So the bar for that field is three times the measured rounding floor, not a number we picked. Extra solver iterations did not change our error at all.

**Far from home.** Float32 positions 20 to 40 millimetres from the world's origin resolve only about 4e-6 mm, and one field was off by 1.01e-05. Computing each animal's step relative to its own root cut that to 2.3e-07.

**A missing function, found late.** The AMD compiler has no atan2f. The error only appears when the kernel launches, not when it compiles.

**The stop rule that looked right.** We stopped the solver when "the active set stopped changing." One test state was off by 5.4e-05. The reason: the matrix there was badly conditioned, so a float32 step was inexact. We replaced it with a rule on step size.

## Noslip: a cancellation in one line

After the main contact solver, MuJoCo runs a clean-up pass called **noslip** that removes the last sideways creep of a foot on the floor. Our first port threw away valid updates in 38 of 64 test states.

The cause was one expression. The general form of a certain correction term subtracts two nearly equal numbers. In float32 the rounding noise in that subtraction was about 0.2, and the true answer was -0.1. The sign flipped, and a safeguard threw away the valid updates. The fix was to write the term in its exact form, where the second number is just the negative of the first, so nothing is subtracted. All stages pass with noslip on, and a version that skips the pass is caught.

A second wrinkle: our gate skips test states that sit on the edge of a contact switching on or off, because float32 cannot call those. It has to judge the edge from the solver's answer before noslip. Noslip pushed one state from 1.8e-05 to 8.5e-03 away from its switch, which hid that it had been a coin flip.

## The floor was our model's fault, not the engine's

For a while, only the six feet could touch the floor in our engine. The fly sank through it and thrashed. We assumed a port bug. Then we ran MuJoCo with the same six pairs: it sank too. We had exported only the six foot pairs, and the other 49 are what keep the body off the floor. With all 55 in, the fly stands like the original. Cost: 14.2 to 23.6 milliseconds per step at 64 animals.

## The engine is partial, and says so

The engine is PARTIAL by our own capability table. Of the fly's 502 contact pairs, 55 floor pairs and 7 grooming pairs are ported. The other 440 are the old arena's box and ellipsoid walls and stones, already replaced by planes and terrain, and porting them needs a different, heavier algorithm.

The jumping-spider body matches on every stage except one number: its contact force is off by 1.5e-05 to 5.2e-05 against a bar of 1e-05, while the motion it produces (qacc) passes. We ruled out summation order, solver convergence and cancellation. The cause is open.

And the engine matches MuJoCo, not a real fly. Passing every gate shows the port is faithful. It says nothing about whether the model of the fly is right.

Next: [A world to live in](05-a-world-to-live-in.md).
