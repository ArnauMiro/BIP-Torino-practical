# Aerospace Innovations Driven by Artificial Intelligence

Blended Intensive Programme, Politecnico di Torino, 7–11 September 2026.

## Part 1. From Airfoil Geometry to Lift and Drag

You have a budget of **2000 aerodynamic evaluations** and four design variables. Everything you build this afternoon is bounded by how you spend them.

---

### Open in Colab

No installation. Any Google account works, use a personal or a university account.

| | |
|---|---|
| **Part I — generating the data** | [Open in Colab](https://colab.research.google.com/github/ArnauMiro/BIP-Torino-practical/blob/main/P1_part1_data_generation.ipynb) |
| **Part II — training the surrogate** | [Open in Colab](https://colab.research.google.com/github/ArnauMiro/BIP-Torino-practical/blob/main/P1_part2_training.ipynb) |

First cell of each notebook pulls the module and installs the dependencies:

```python
!git clone -q https://github.com/USER/REPO.git
%cd REPO
!pip install -q -r requirements.txt
```

**Colab does not keep your files.** When the runtime disconnects, everything in the working directory is gone. Download `campaign.npz` and `campaign.json` as soon as Part I finishes, that is your 2000 evaluations, and they cannot be bought again.

```python
from google.colab import files
files.download("campaign.npz")
files.download("campaign.json")
```

Re-upload them at the start of Part II with the file browser in the left sidebar.

---

### Running locally instead

```bash
git clone https://github.com/ArnauMiro/BIP-Torino-practical.git
cd BIP-Torino-practical
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Part I needs only numpy, scipy, matplotlib and neuralfoil. Part II adds torch and pyLOM.

---

### What is here

```
P1_ancillary.py                    everything provided: geometry, the metered
                                   evaluator, the AeroSurrogate contract, training
P1_part1_data_generation.ipynb     design and spend your campaign
P1_part2_training.ipynb            build the surrogate
```

`P1_ancillary.py` is provided and should not be modified. What is yours: the sampler, the bounds you choose, how you split the data, the envelope, and the model.

Check it works before you start:

```bash
python P1_ancillary.py
```

---

### The contract

Whatever you build must expose two methods, because that is all the reinforcement-learning agent can see of it:

```python
predict(X)      # (N, 4) of [m, p, t, alpha]  ->  (N, 2) of [Cl, log10(Cd)]
in_envelope(X)  # (N, 4)                      ->  (N,) bool
```

Note the `log10`. The dataset does not do that transform for you.

Before you submit, run:

```python
p1.check_contract(model)
```

It reloads your surrogate from disk exactly as Practical 3 will and checks the four things that would otherwise fail in front of the room.

---

### Roles

Five, agreed before you write any code: campaign designer, generator, trainer, validator, **adversary**. The adversary's job for the first thirty minutes is to argue against whatever the campaign designer proposes. It is the most useful role in the group.


## Part 2. From Surrogate to Designer

In part 1 you built something that predicts lift and drag in microseconds. Now you point an optimiser at it and ask for the best wing section it can find.

Then you check the answer against a better model, and find out whether your surrogate was telling the truth.

---

### Open in Colab

| | |
|---|---|
| **Practical 3 — training the designer** | [Open in Colab](https://colab.research.google.com/github/ArnauMiro/BIP-Torino-practical/blob/main/P3_student.ipynb) |

Same first cell as part 1:

```python
!git clone -q https://github.com/ArnauMiro/BIP-Torino-practical.git
%cd BIP-Torino-practical
!pip install -q -r requirements.txt
```

**Bring part 1's files.** Upload `campaign.npz`, `campaign.json` and your saved surrogate with the file browser in the left sidebar before you run anything. If you lost them, or your surrogate fails its contract check, the notebook hands you a reference one automatically and you can do the entire practical with it. Say which one you used when you report your result.

---

### What is here

```
P1_ancillary.py       Part 1's module. Imported unchanged. If you edited it,
                      re-download it, today's code imports the same file
P3_ancillary.py       the plumbing: shape, action space, objective, environment,
                      the DE baseline, the referee
P3_physics.py         the referee's physics. Contains a spoiler
P3_student.ipynb      the practical
```

Check both modules run before you start:

```bash
python P1_ancillary.py
python P3_ancillary.py
```

---

### What you change

Two things:

- **the bounds** you hand the agent, in `NACA4Parameterizer`
- **the objective** it maximises, in `LiftToDragObjective`

Everything else is provided so you are not distracted writing code.

The objective is one Python object and *both* optimisers use it. Not a copy, the same instance. The moment they differ, the comparison you are about to make stops meaning anything while still producing a table that looks fine.

---

### The two optimisers

**Differential evolution.** No training, no hyperparameters worth arguing about, roughly 800 evaluations. It will find the maximum of a three-parameter function without difficulty.

**PPO.** Learns a policy: something that takes a section and improves it. Around 30,000 evaluations and a minute of training.

DE will probably win. Work out why you would ever use the other one, and hold that answer until the last section, where the problem changes and so does the answer.

Report both the result **and** the evaluation count. A comparison that reports only the winner is not a comparison.

---

### The referee

Your surrogate learned from a fast model. Part 2's judge is a better one.

You would never optimise against the referee, it is far too slow. But you can afford to *check* against it, and the gap between what your surrogate promised and what the referee delivers is the number this practical is about.

`P3_physics.py` explains exactly how the referee works.

---

### The fix ladder

Four responses to the promise gap. Try them in order and record what each buys you.

| rung | what it does | what you change |
|---|---|---|
| 1 | keep the agent inside the box your data covers | parameterizer bounds |
| 2 | impose a design requirement | `min_t`, `cl_min` |
| 3 | penalise leaving the envelope | `w_envelope` |
| 4 | spend a few referee calls and correct | `mf_loop` |

**One of them will do nothing at all.** Working out why is the most useful thing you will do here, and the notebook will not let you move on until your group has an answer.

Rungs 1 to 3 reason from data you already had. Rung 4 is the only one that consults anything new.

---

### Submitting

One design, `(m, p, t)`. It goes on the wall ranked on what it **delivers**, not on what your surrogate promised. A design can be honestly predicted and still poor.

Plus one sentence: which rung earned you the most, and why.

---

### Roles

Same five as part 1, rotated. Whoever was campaign designer should not be the one choosing the objective. The **adversary** keeps the job, and today it is a bigger one: their task is to find a design your surrogate loves and the referee does not.

If the adversary finds nothing, that is a result. Report it.