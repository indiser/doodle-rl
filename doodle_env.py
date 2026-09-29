import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # no sound, no audio device errors

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Game"))

import random
import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import assets as A
from config import *
from Doodle import Player
from Platform import Platform, ExplodingPlatform, MovingPlatform
from Feeder import Feeder
from Monster import Monster
from Trampoline import Trampoline

N_PLATS = 6
N_OBS = 3 + N_PLATS * 3 + 2
MAX_STEPS = 5000

BROKEN_FRAMES = [A.broken_platform0, A.broken_platform1, A.broken_platform2, A.broken_platform3]
EXPLOSIVE_FRAMES = [A.exploding_platform0, A.exploding_platform1, A.exploding_platform2]
BAT_FRAMES = [A.bat1, A.bat2, A.bat3]


class DoodleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = Box(-3.0, 3.0, shape=(N_OBS,), dtype=np.float32)
        self.action_space = Discrete(3)  # 0 nothing, 1 left, 2 right
        self.screen = None
        self.clock = None
        self.platforms = []
        self.feeders = []
        self.monsters = []
        self.trampolines = []
        self.player = None
        self.score = 0
        self.steps = 0

    # ---------- world building (copied from your main.py) ----------
    def _make_platform(self, x, y):
        roll = random.randint(1, 100)
        if roll <= 10:
            return MovingPlatform(x, y, A.horizental_platform, 'horizontal', self.score)
        elif roll <= 20:
            return MovingPlatform(x, y, A.vertical_platform, 'vertical', self.score)
        elif roll <= 35:
            return ExplodingPlatform(x, y, EXPLOSIVE_FRAMES)
        elif roll <= 50:
            return Platform(x, y, A.broken_platform0, breakable=True, broken_frames=BROKEN_FRAMES)
        return Platform(x, y, A.platform)

    def _spawn_entities(self, p):
        if type(p) is Platform and not p.breakable:
            r = random.randint(1, 100)
            if r <= 5 and self.score > 1000:
                self.trampolines.append(Trampoline(p.rect.centerx + 15, p.rect.top - 5,
                                                   A.trampoline_down, A.trampoline_up))
            elif 5 < r <= 15 and self.score > FEEDER_THRESHOLD:
                self.feeders.append(Feeder(p.rect.centerx + 15, p.rect.top - 5,
                                           A.feeder_down, A.feeder_up))
            elif 15 < r <= 25 and self.score > 1:
                self.monsters.append(Monster(p, 0, BAT_FRAMES))

    def _add_platform(self):
        x = random.randint(0, WIDTH - PLATFORM_WIDTH)
        highest = min(q.rect.y for q in self.platforms)
        y = highest - random.randint(60, 120)
        p = self._make_platform(x, y)
        self.platforms.append(p)
        self._spawn_entities(p)

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.platforms.clear()
        self.feeders.clear()
        self.monsters.clear()
        self.trampolines.clear()
        self.score = 0
        self.steps = 0

        start = Platform(WIDTH // 2 - PLATFORM_WIDTH // 2, HEIGHT - 20, A.platform)
        self.platforms.append(start)
        self.player = Player(WIDTH // 2, start.rect.top, A.doodle_right, A.doodle_left)

        y = start.rect.top
        while y > 0:
            y -= random.randint(60, 120)
            self.platforms.append(self._make_platform(random.randint(0, WIDTH - PLATFORM_WIDTH), y))
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        prev_score = self.score
        terminated = False
        keys = {pygame.K_LEFT: action == 1, pygame.K_RIGHT: action == 2}

        pl = self.player
        pl.move(keys, MOVE_SPEED, WIDTH)
        pl.apply_gravity(GRAVITY)

        # collisions
        if pl.y_velocity > 0:
            hit = False
            for t in self.trampolines:
                if pl.rect.colliderect(t.rect):
                    pl.rect.bottom = t.rect.top
                    pl.jump(JUMP_STRENGTH, boost_multiplier=2.5, flip=True)
                    t.trigger()
                    self.score += 700
                    hit = True
                    break
            if not hit:
                for f in self.feeders:
                    if pl.rect.colliderect(f.rect):
                        pl.rect.bottom = f.rect.top
                        pl.jump(JUMP_STRENGTH, boost_multiplier=1.8)
                        f.trigger()
                        hit = True
                        break
            if not hit:
                for p in self.platforms:
                    if not p.broken and pl.rect.colliderect(p.rect):
                        pl.rect.bottom = p.rect.top
                        pl.jump(JUMP_STRENGTH)
                        if p.breakable:
                            p.break_platform()
                        break

        if pl.rect.top > HEIGHT:
            terminated = True

        for m in self.monsters[:]:
            if pl.rect.colliderect(m.rect):
                if pl.y_velocity > 0 and pl.rect.bottom < m.rect.centery + 10:
                    pl.jump(JUMP_STRENGTH)
                    self.monsters.remove(m)
                else:
                    terminated = True
                    break

        # update / cleanup
        for p in self.platforms[:]:
            p.update()
            if p.finished or p.rect.top >= HEIGHT:
                self.platforms.remove(p)
                self._add_platform()

        for m in self.monsters[:]:
            m.update()
            if m.platform not in self.platforms:
                self.monsters.remove(m)

        # camera
        threshold = HEIGHT // 3
        if pl.rect.top <= threshold:
            pl.rect.top = threshold
            shift = pl.y_velocity
            self.score -= shift
            for p in self.platforms[:]:
                p.scroll(shift)
                if p.rect.top >= HEIGHT:
                    self.platforms.remove(p)
                    self._add_platform()
            for f in self.feeders[:]:
                f.scroll(shift)
                if f.rect.top >= HEIGHT:
                    self.feeders.remove(f)
            for t in self.trampolines[:]:
                t.scroll(shift)
                if t.rect.top >= HEIGHT:
                    self.trampolines.remove(t)

        reward = (self.score - prev_score) / 100.0
        if terminated:
            reward -= 2.0
        truncated = self.steps >= MAX_STEPS

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, {"score": self.score}

    def _obs(self):
        r = self.player.rect
        obs = [r.centerx / WIDTH, r.centery / HEIGHT, self.player.y_velocity / 20.0]

        plats = [p for p in self.platforms if not p.broken]
        plats.sort(key=lambda p: math.hypot(p.rect.centerx - r.centerx, p.rect.centery - r.centery))
        for p in plats[:N_PLATS]:
            bad = 1.0 if (p.breakable or getattr(p, "is_explosive", False)) else 0.0
            obs += [(p.rect.centerx - r.centerx) / WIDTH, (p.rect.top - r.bottom) / HEIGHT, bad]
        obs += [0.0] * (3 * (N_PLATS - min(len(plats), N_PLATS)))

        if self.monsters:
            m = min(self.monsters, key=lambda m: math.hypot(m.rect.centerx - r.centerx,
                                                            m.rect.centery - r.centery))
            obs += [(m.rect.centerx - r.centerx) / WIDTH, (m.rect.centery - r.centery) / HEIGHT]
        else:
            obs += [2.0, 2.0]
        return np.clip(np.array(obs, dtype=np.float32), -3.0, 3.0)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        pygame.event.pump()
        self.screen.blit(A.background, (0, 0))
        for group in (self.platforms, self.feeders, self.monsters, self.trampolines):
            for o in group:
                o.draw(self.screen)
        self.player.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None