# 🪰 THE FLY CONNECTOME WIKI

## A Comprehensive Overview of the MaleCNS v1.0 Phenomenon and Its Viral Aftermath

_Last Updated: 2026-09-30_

---

## 📑 INDEX

1.  **Foundational Datasets**
    - 1.1 MaleCNS v1.0 (Male Fruit Fly)
    - 1.2 FlyWire (Female Fruit Fly)

2.  **The Viral Phenomenon**
    - 2.1 Timeline of Release & Viral Spread
    - 2.2 The Core Appeal

3.  **GitHub Project Catalog**
    - 3.1 Gaming & Entertainment
    - 3.2 Robotics & Autonomous Vehicles
    - 3.3 Finance & Trading
    - 3.4 Scientific Computing & Reservoir Computing
    - 3.5 Language Models & AI
    - 3.6 Simulation Platforms & SDKs
    - 3.7 Curated Lists & Foundational Models

4.  **Unrelated "Fly" Viral Trends**
    - 4.1 Gigafly
    - 4.2 Филяй (Filyai)
    - 4.3 Catch The Fly Challenge

5.  **Key Scientific Concepts**
    - 5.1 Connectome vs. Effectome
    - 5.2 Reservoir Computing
    - 5.3 Neurogenesis & Hybrid Systems

6.  **Notable Individuals & Organizations**

7.  **Unexplored Frontiers & "Crazy" Ideas**

---

## 1. FOUNDATIONAL DATASETS

### 1.1 MaleCNS v1.0 (Male Fruit Fly) 🧠

- **What it is**: A complete connectome (wiring diagram) of the adult male _Drosophila melanogaster_ central nervous system, including the brain and ventral nerve cord.
- **Key Numbers**:
  - ~166,700 neurons[reference:0][reference:1]
  - ~25.6 million directed synaptic connections[reference:2][reference:3]
  - ~125 million total synapses[reference:4]
  - ~11,700 cell types[reference:5]
- **Release**: Version 1.0 was released on **June 8, 2026** by a collaboration between **HHMI Janelia (FlyEM)**, the **University of Cambridge**, **MRC LMB**, and **Google Research**[reference:6][reference:7].
- **License**: CC-BY (openly available)[reference:8].
- **Significance**: It was the first complete, publicly downloadable connectome of an adult male fly, which directly enabled the wave of viral projects.

### 1.2 FlyWire (Female Fruit Fly)

- **What it is**: A whole-brain connectome of an adult female _Drosophila melanogaster_ (the FAFB dataset).
- **Key Numbers**:
  - ~139,255 proofread neurons[reference:9][reference:10]
  - Over 8,000 cell types[reference:11]
  - ~50 million chemical synapses[reference:12]
- **Significance**: Released earlier (flagship paper in _Nature_ in October 2024), it was the first complete adult fly brain connectome and is used by several projects, including the viral Minecraft video[reference:13].

---

## 2. THE VIRAL PHENOMENON

### 2.1 Timeline of Release & Viral Spread

- **Oct 2024**: FlyWire (female) connectome published.
- **June 8, 2026**: MaleCNS v1.0 released.
- **Sept 3, 2026**: MaleCNS paper published in _Cell_.
- **Sept 7, 2026**: YouTuber **Ro0oney** posts _"I Put A Fly's Conscious Brain into Minecraft"_ (using FlyWire), which goes viral.
- **Sept 8-16, 2026**: Explosion of GitHub projects (DOOMFLY, Stonkfly, FlySweeper, FlyDrones, etc.) and media coverage.
- **Sept 12, 2026**: TweakTown article notes the fly brain "has played Doom more than 6,000 times and still hasn't beaten it"[reference:14].

### 2.2 The Core Appeal

The viral wave was driven by a simple, absurd premise: **a biological brain, reconstructed from a real fly, is now doing human things**—playing Doom, trading Bitcoin, driving drones. The fact that the dataset was **openly available** and the projects were **open-source on GitHub** made it a participatory internet phenomenon[reference:15]. The mismatch between the fly's simplicity and the tasks' complexity created endless comedic and scientific material.

---

## 3. GITHUB PROJECT CATALOG

_All repositories listed here are publicly available on GitHub as of late September 2026._

### 3.1 Gaming & Entertainment 🎮

| Project               | Creator / Repo                       | Description                                                                                                                                                                                                                                                                                     |
| --------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DOOMFLY**           | Ovaday/doomfly                       | Fly connectome controls a live Doom arena. Game frames stimulate sensory neurons; activity propagates through the retained MaleCNS wiring. Includes experimental dopamine-gated memory and a spectator UI. Status: live experimental training, not demonstrated learned survival[reference:16]. |
| **DOOM-x-Fly**        | Aur1ety/DOOM-x-Fly                   | Simulates the MaleCNS connectome to play Doom. Reports finishing the level 57% of the time.                                                                                                                                                                                                     |
| **FlySweeper**        | ASHR12/FlySweeper                    | The complete MaleCNS connectome (166,700 spiking neurons) wired to Minesweeper. The frozen connectome is a random player (0 wins). A trained version (fly-mb) wins **76 of 100** held-out games by training only Kenyon-cell → mushroom-body output synapses[reference:17].                     |
| **Fly-in-a-Tetris**   | Jahongir-Toshtemirov/fly-in-a-tetris | Fruit fly plays Tetris while its real connectome activity is displayed in 3D.                                                                                                                                                                                                                   |
| **fly-tetris**        | SLuke115/fly-tetris                  | Single-fly Tetris experiment with a Three.js cyberpunk fly and live neural telemetry.                                                                                                                                                                                                           |
| **FlyBrain-HalfLife** | Yusuftmle/FlyBrain-HalfLife          | Autonomous agent playing Half-Life. Every movement, turn, and jump is generated entirely by biophysical Leaky Integrate-and-Fire circuits from the connectome[reference:18].                                                                                                                    |
| **FlyBrain**          | snedea/flybrain                      | Interactive browser simulation of the Drosophila brain. You can feed it, touch it, blow wind, and change the light to see how it reacts.                                                                                                                                                        |
| **Super Mario 64**    | (Unspecified)                        | Fly brain connected to control Mario[reference:19].                                                                                                                                                                                                                                             |
| **Devil Daggers**     | (Unspecified)                        | Fly brain connected to the fast-paced shooter.                                                                                                                                                                                                                                                  |

### 3.2 Robotics & Autonomous Vehicles 🤖

| Project                   | Creator / Repo           | Description                                                                                                                                                                                                                      |
| ------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FlyDrones**             | SpikeCalls/FlyDrones     | A fruit fly connectome as a drone pilot. Camera → fly eyes → spiking brain → descending neurons → drone. Hand gestures create optic-flow illusions the fly interprets as flight commands. Live 3D demo in browser[reference:20]. |
| **haltere**               | skulitom/haltere         | MaleCNS v1.0 trained to fly an FPV drone in the Liftoff simulator, with live neural activity video.                                                                                                                              |
| **Fly.exe**               | Ibtisam-Mohammad/Fly.exe | Runs the complete released Drosophila male CNS connectome (165,122 neurons, 25.5M edges) inside a physical fly body, in closed loop, on one GPU.                                                                                 |
| **Flyhard**               | (Unspecified)            | Fly-brain model drives a car in the CARLA driving simulator. Can operate steering, pedals, and do a three-point turn.                                                                                                            |
| **fly-arena**             | artem-x-meta/fly-arena   | Experimental Drosophila simulator using MaleCNS connectome and FlyGym/MuJoCo physics. Includes autonomous hybrid behaviors and two-fly food competition[reference:21].                                                           |
| **FlyBrain Robot Bridge** | (Unspecified)            | Aims to connect the connectome simulation to a physical robot's camera and motor controls.                                                                                                                                       |
| **FLYNN**                 | (Research)               | Uses the fly brain's topology to build a robust neural network for robot navigation in MuJoCo[reference:22].                                                                                                                     |

### 3.3 Finance & Trading 📈

| Project                      | Creator / Repo    | Description                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Stonkfly**                 | nftechie/stonkfly | Fly-connectome simulation that operates a crypto trading account. Converts Bitcoin price charts into neural signals (3,335 brightness inputs, 811 R8 color inputs) and lets the 166,700 neurons and 25.6M connections decide buy/sell/hold. Positive P&L stimulates dopamine cells; negative P&L stimulates aversive cells. **Profitable learning has not been demonstrated**[reference:23]. |
| **FLM (Fly Language Model)** | nftechie/flm      | Couples the full retained MaleCNS connectome to a frozen LiquidAI LFM2.5-1.2B-Instruct backbone. The connectome acts as a reservoir computer. The project's own controls show the wiring **does not help** the language model[reference:24].                                                                                                                                                 |

### 3.4 Scientific Computing & Reservoir Computing 🔬

| Project                         | Creator / Repo                                | Description                                                                                                                                                                                                                                                                                                             |
| ------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **maleCNS-reservoir-computing** | JangYeongSil69420/malecns-reservoir-computing | GPU-accelerated multi-channel reservoir computer and Liquid State Machine powered by the full MaleCNS v1.0 graph. Uses parallel multi-neuropil signal injection across 151M biological synapses for chaotic time-series forecasting (Mackey-Glass). Achieves **Test R² = 0.939** for short-term dynamics[reference:25]. |
| **wetware**                     | Roxx0x/wetware                                | Uses the complete fruit-fly connectome as a reservoir computer to read handwriting. Achieves **94% accuracy** with only a simple linear readout trained on top. The wiring is fixed biology; only the readout learns[reference:26].                                                                                     |
| **fly_ocr**                     | jerryjliu/fly_ocr                             | A simulated fly circuit recognizes printed letters and numbers from PDF pixels. Achieved **87.6% accuracy** on a glyph benchmark and **5.7% character error rate** on PDF text[reference:27].                                                                                                                           |
| **Wordle vs. Fruit Fly**        | Quantum-Coded/fruitfly                        | Hooks the whole-brain wiring diagram into a Wordle solver via reservoir computing with reinforcement learning[reference:28].                                                                                                                                                                                            |
| **Drug discovery simulator**    | (Unspecified)                                 | Fly brain controls a drug molecule inside a protein, exploring new ways to model drug interactions.                                                                                                                                                                                                                     |
| **Fly TTS**                     | (Unspecified)                                 | Text-to-speech synthesis driven by connectome neural activity patterns.                                                                                                                                                                                                                                                 |

### 3.5 Language Models & AI 🧠

| Project        | Creator / Repo                       | Description                                                                                                       |
| -------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **FLM**        | nftechie/flm                         | See above (3.3). A frozen 1.2B LLM coupled to the fly connectome. Its own controls show the wiring does not help. |
| **claude-fly** | legacyindiesubmissions-ai/claude-fly | Uses Claude Opus as a replacement brain in the FlyWire Drosophila connectome simulation.                          |
| **fly.ai**     | alextitonis/fly.ai                   | Goal is a general-purpose "fly reservoir": plug any task into the same frozen brain, read out what it does.       |

### 3.6 Simulation Platforms & SDKs 🛠️

| Project          | Creator / Repo               | Description                                                                                                                                                                         |
| ---------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FlyBrain SDK** | jk06095-lang/flybrain        | AI-friendly Python SDK with sparse LIF simulation, semantic I/O, and a local REST API. "Plug a fruit-fly connectome into anything"[reference:29].                                   |
| **fly-brain**    | erojasoficial-byte/fly-brain | Embodied whole-brain spiking simulation of Drosophila combining FlyWire v783 connectome with a biomechanical body, multi-modal sensory processing, and Hebbian synaptic plasticity. |
| **fly-brain**    | onnxaa/fly-brain             | Data-only whole-brain model of Drosophila from FlyWire FAFB v783. Supports associative learning, sleep homeostasis, a circadian clock, and vision[reference:30].                    |
| **flybrain**     | snedea/flybrain              | Interactive browser simulation of the Drosophila brain with 139,255 neurons from FlyWire FAFB.                                                                                      |
| **fly-brain**    | eonsystemspbc/fly-brain      | Emulation of the Drosophila Fly brain using Brian2, Brian2CUDA, PyTorch, NEST GPU, and neuromorphic chips.                                                                          |

### 3.7 Curated Lists & Foundational Models 📚

| Project                     | Creator / Repo                  | Description                                                                                                                                                                                                                                                                                                      |
| --------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **awesome-fly**             | cobanov/awesome-fly             | A curated list of fruit fly connectome projects: MaleCNS, FlyWire, brain simulations, embodied models, games, and research tools. (Note: search results for "awesome-fly" returned mostly Fly.io hosting lists; the connectome-specific list is referenced in the ecosystem but its exact repo name may differ.) |
| **fly-connectome-template** | cobanov/fly-connectome-template | A template to build your own fly experiment, with a real MaleCNS soma atlas and a React + Three.js workbench.                                                                                                                                                                                                    |

---

## 4. UNRELATED "FLY" VIRAL TRENDS

_These are viral trends involving flies that are separate from the connectome phenomenon._

### 4.1 Gigafly 🦾🪰

- **What it is**: An absurd derivative of the GigaChad meme where GigaChad is shown riding a giant housefly. The meme first appeared as early as October 2024 and became increasingly popular in the first half of 2026 on X/Twitter and TikTok[reference:31].
- **Origins**: Unclear. According to SoyjakWiki, it started spreading in 2024 on soyjak.party. The earliest known user to post a meme of GigaChad riding a fly was Redditor KaiserKota on October 8, 2024[reference:32].
- **Nature**: Often AI-generated, posted as a form of absurd trolling. The meme's ironic use makes it intentionally difficult to determine the poster's intentions[reference:33]. It has spawned a cryptocurrency token (GIGAFLY) on Coinbase[reference:34].

### 4.2 Филяй (Filyai) 💃

- **What it is**: A German-origin Instagram dance trend that went viral in Russian-speaking spaces in late 2025 and early 2026[reference:35].
- **Origin**: A video of a young German raver and model, **Samuël Domen**, dancing expressively to the techno track _"Vielleicht Vielleicht"_ by MilleniumKid x JBS. The video has over **19 million views**[reference:36].
- **Meaning**: "Филяй" is a playful Russian corruption of the German word _"vielleicht"_ (meaning "perhaps")[reference:37]. The trend evolved into a movement of female empowerment and self-irony, with Russian women "claiming" the German dancer and creating their own versions of the trend[reference:38].

### 4.3 Catch The Fly Challenge 🖐️🪰

- **What it is**: A viral hand-motion trend where people try to "catch" an imaginary (or AR) fly using quick hand movements[reference:39].
- **Platform**: Spread primarily across Snapchat and TikTok. Some versions use augmented reality to place a digital fly on the screen, and users try to catch it with virtual chopsticks before time runs out[reference:40].
- **Nature**: A quick, physical challenge that is easy to replicate, similar to other short-lived social media trends.

---

## 5. KEY SCIENTIFIC CONCEPTS

### 5.1 Connectome vs. Effectome

- **Connectome**: The physical wiring diagram of the brain—which neurons are connected to which. The MaleCNS and FlyWire datasets are connectomes.
- **Effectome**: A causal model of the brain that specifies **how strongly** neurons affect each other _in vivo_. The connectome shows the paths; the effectome shows the functional strength of those paths[reference:41].
- **Why it matters**: A major limitation of the connectome is that it is static. The effectome is the dynamic, functional counterpart. Mapping it is one of the biggest open challenges in neuroscience, and the fly connectome provides a path to it[reference:42].

### 5.2 Reservoir Computing

- **What it is**: A computational paradigm where a fixed, recurrent network (the "reservoir") transforms input signals into a high-dimensional space. Only a simple readout layer is trained.
- **How the fly is used**: The fly connectome's fixed wiring is used as the reservoir. Its complex, recurrent dynamics naturally project inputs into a rich feature space. Projects like **wetware** and **maleCNS-reservoir-computing** use this approach[reference:43][reference:44].
- **Why it's interesting**: It tests how much computation is "for free" in a biological brain's wiring, before any learning happens in the recurrent connections[reference:45].

### 5.3 Neurogenesis & Hybrid Systems

- **Can we build more neurons on top of it?** Technically, yes, but it's a significant engineering and scientific challenge.
- **Approaches**:
  - **Sensory/Motor Prosthetics**: Interfacing external systems with the fly's inputs and outputs (e.g., FlyDrones, Fly.exe).
  - **Artificial Neural Layers**: Inserting new computational layers between sensory and motor regions (e.g., via the FlyBrain SDK).
  - **Simulating Neurogenesis**: The most radical approach—inserting new, biologically plausible neurons with initially random connections and using activity-dependent plasticity to refine their wiring【Conversation History】.
- **Implications**: This allows researchers to test how new behaviors emerge from a known biological substrate and study structural plasticity in a controlled digital environment.

---

## 6. NOTABLE INDIVIDUALS & ORGANIZATIONS

| Name                                  | Role / Contribution                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Alex Wormuth (nftechie)**           | Coinbase engineer. Created **DOOMFLY**, **Stonkfly**, and **FLM**. A central figure in the viral wave. |
| **Lyra Bubbles**                      | Developer who trained the fly brain to play **Beat Saber** using reinforcement learning.               |
| **Ro0oney**                           | YouTuber whose _"I Put A Fly's Conscious Brain into Minecraft"_ video helped ignite the viral trend.   |
| **HHMI Janelia (FlyEM)**              | Lead institution for the MaleCNS v1.0 dataset.                                                         |
| **Google Research**                   | Collaboration partner on MaleCNS v1.0.                                                                 |
| **University of Cambridge / MRC LMB** | Collaboration partners on MaleCNS v1.0.                                                                |
| **FlyWire Consortium**                | Created the female fly brain connectome (FAFB).                                                        |
| **Samuël Domen**                      | German raver and model whose dance video became the **Филяй** trend.                                   |

---

## 7. UNEXPLORED FRONTIERS & "CRAZY" IDEAS

_These are directions that no one appears to have attempted yet, based on the current GitHub landscape._

1.  **The Effectome Mapping Project**: Systematically map the functional strength of every synapse by running millions of stimulus-response simulations.
2.  **Fly Brain as a Co-Author**: Let the fly's _spontaneous_ neural activity (not responses to stimuli) drive generative art or music.
3.  **In-Silico Evolutionary Arms Race**: Simulate a population of fly connectomes competing for resources, with "mutations" in their wiring, and observe what novel behaviors emerge over thousands of generations.
4.  **Cross-Species Connectome Chimeras**: Transplant functional circuits from the fly connectome into a simulated mammalian cortical column (or vice versa).
5.  **Real-Time Multiplayer Fly Brain**: Two or more connectomes running in parallel in a shared virtual world, with a communication channel between them.
6.  **The Fly as a "Chaos Oracle"**: Deploy the connectome's reservoir dynamics against real-world chaotic data (weather, financial markets) and compare to state-of-the-art ML.
7.  **"Drug" the Fly Brain**: Introduce simulated neuromodulators (dopamine, serotonin) into the connectome and screen for "virtual drugs" that produce desired behavioral outcomes.
8.  **Connectome-Based Cryptographic Primitive**: Use the fly's chaotic neural dynamics as a source of randomness for cryptographic key generation.
9.  **Embodied Fly with Real Biomechanics**: Put the connectome on a physical robot with realistic fly-like biomechanics (compliant legs, halteres, wing actuators).
10. **Connectome-Based "Turing Test"**: A blind experiment where human judges try to distinguish a connectome-driven fly from a real fly.

---

## 📝 NOTES & CAVEATS

- **Not a literal fly brain**: Almost every project explicitly states that the simulation is not a literal reconstructed living fly brain. The wiring comes from biological reconstruction, but the dynamics, sensory interfaces, and control assignments are engineered models and choices[reference:46].
- **Rapidly evolving**: The ecosystem is expanding daily. New repositories and projects are appearing constantly. The `awesome-fly` curated list (if you can find the correct repository) is a good place to track updates.
- **Mixed results**: Some projects (like FlySweeper's trained version) show impressive performance. Others (like FLM) show that the connectome's wiring _does not_ help with certain tasks. The scientific value is often in the negative results as much as the positive ones.

---

_End of Wiki._
