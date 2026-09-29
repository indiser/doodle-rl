# 🪄 Doodle-RL

> *"I trained a neural network to play Doodle Jump so I don't have to. It's better at it than me. I have mixed feelings."*

A **Proximal Policy Optimization (PPO)** agent that learns to play a custom Doodle Jump clone — built from scratch with Pygame and trained using Stable-Baselines3. The agent starts knowing absolutely nothing (relatable), and after 500,000 timesteps, it jumps better than most humans. Still can't make coffee though.

---

## 🧠 How It Works

The game world is wrapped in a custom [Gymnasium](https://gymnasium.farama.org/) environment (`doodle_env.py`). The agent observes the world as a flat vector and outputs one of three actions:

| Action | Meaning |
|--------|---------|
| `0` | Do nothing (the agent's default mood) |
| `1` | Move left |
| `2` | Move right |

### Observation Space — `23 floats` of pure anxiety

- Player `(x, y)` position + `y_velocity` — *"where am I and how fast am I falling"*
- Nearest **6 platforms**: relative `(dx, dy)` + a `bad_platform` flag — *"should I land here or is this a trap?"*
- Nearest **monster**: relative `(dx, dy)` — *"is that bat looking at me?"*

All values are clipped to `[-3.0, 3.0]` and normalized. The agent is rewarded for climbing and penalized `-2.0` for dying. Skill issue.

---

## 🎮 The Game

A custom Doodle Jump built with **Pygame**, featuring:

- 🟩 **Normal platforms** — safe, boring, reliable (like a senior dev)
- 💥 **Exploding platforms** — yellow → red → 💥 (2-second fuse, no refunds)
- ↔️ **Horizontal moving platforms** — speed scales with score
- ↕️ **Vertical moving platforms** — same deal
- 🪨 **Breakable platforms** — crumbles on first touch, animated
- 🦇 **Bat monsters** — stomp them for a bonus jump, get touched and die
- 🌀 **Trampolines** — 2.5× jump boost, does a little spin
- 🍄 **Feeders** — 1.8× jump boost, unlocks after 3000 points

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the agent (grab a coffee, takes a while)
python train.py

# 3. Watch your creation suffer and succeed
python watch.py
```

> ⚠️ Audio is disabled during training (`SDL_AUDIODRIVER=dummy`). The agent doesn't need motivation music. You might.

---

## 📁 Project Structure

```
Doodle-RL/
├── doodle_env.py       # Gymnasium env — the agent's entire universe
├── train.py            # PPO training loop (500k timesteps)
├── watch.py            # Load model & watch it play
├── doodle_ppo.zip      # Pre-trained weights (skip the wait)
├── requirements.txt
└── Game/
    ├── main.py         # Human-playable version
    ├── Doodle.py       # Player class (also does backflips on trampolines)
    ├── Platform.py     # All platform types
    ├── Monster.py      # Bat enemy
    ├── Feeder.py       # Jump booster
    ├── Trampoline.py   # Super jump booster
    ├── config.py       # Game constants (gravity, speed, etc.)
    └── assets.py       # Sprite & sound loader
```

---

## ⚙️ Tech Stack

| What | Why |
|------|-----|
| `pygame` | Game engine |
| `gymnasium` | RL environment interface |
| `stable-baselines3` | PPO implementation |
| `torch` | Neural network backend |
| `numpy` | Observation vector math |

---

## 🎲 Reward Function

```
reward = (score_delta) / 100.0
       - 2.0  if terminated (fell or got hit)
```

Simple. Brutal. Effective. Like a code review from a senior engineer.

---

## 🏃 Play It Yourself

```bash
cd Game
python main.py
```

Arrow keys to move. Try to beat the AI. You probably won't.

---

## 📊 Training Details

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (MlpPolicy) |
| Timesteps | 500,000 |
| Max steps/episode | 5,000 |
| Observation size | 23 |
| Action space | Discrete(3) |

---

*Built with pygame, patience, and the quiet hope that the agent would eventually stop walking off the edge.*
