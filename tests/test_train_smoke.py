# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Training smoke: a few hundred real DQN steps on the survive task. Proves
the torch path (net, replay buffer, TD update) end to end without a full run.
Skipped cleanly when torch isn't installed (it's the [train] extra)."""

import random

import pytest

torch = pytest.importorskip("torch")


def test_dqn_survive_smoke():
    from agents.dqn_survive import QNet, Replay, greedy_action
    from orbitduel.survive import SurviveEnv, ACTIONS

    torch.manual_seed(0)
    random.seed(0)
    q, tgt = QNet(), QNet()
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    buf = Replay(2_000)

    env = SurviveEnv(seed=1)
    obs, _ = env.reset()
    for step in range(500):
        a = random.randrange(len(ACTIONS)) if random.random() < 0.5 \
            else greedy_action(q, obs)
        obs2, r, term, trunc, _ = env.step(a)
        buf.push(obs, a, r, obs2, term)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()
        if step > 100:
            o, act, rew, nxt, done = buf.sample(64)
            with torch.no_grad():
                a_star = q(nxt).argmax(1, keepdim=True)
                target = rew + 0.99 * (1 - done) \
                    * tgt(nxt).gather(1, a_star).squeeze(1)
            pred = q(o).gather(1, act.unsqueeze(1)).squeeze(1)
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert torch.isfinite(loss)


def test_league_selection_rule():
    """The confirm pass's pick: ladder mean first, ties inside TIE_BAND resolve
    toward the better confirmed L10, and the band does not let a clearly worse
    mean win on L10 alone. Asserted here because the two tablet seeds proved a
    mean alone picks between different pilots by coin flip (Stage 6)."""
    from agents.league_duel import pick_winner, TIE_BAND

    def row(tag, conf, l10):
        per = {lv: conf for lv in range(1, 11)}
        per[10] = l10
        return (tag, conf, conf, per, None)

    inside = TIE_BAND / 2
    tied = [row("lead", 0.80, 0.30), row("split", 0.80 - inside, 0.45)]
    assert pick_winner(tied)[0] == "split"          # in-band tie -> L10 decides
    clear = [row("lead", 0.80, 0.30), row("l10er", 0.80 - 2 * TIE_BAND, 0.60)]
    assert pick_winner(clear)[0] == "lead"          # out of band: mean stands
    assert pick_winner([tied[0]])[0] == "lead"      # degenerate single row


def test_ppo_export_roundtrip(tmp_path):
    """export_json -> PolicyNet reproduces the torch net's argmax decisions."""
    from agents.ppo_duel import ActorCritic, export_json
    from orbitduel.netpilot import PolicyNet

    torch.manual_seed(3)
    net = ActorCritic()
    path = tmp_path / "net.json"
    export_json(net, path)
    ref = PolicyNet(path)
    for i in range(50):
        g = torch.Generator().manual_seed(i)
        obs = torch.rand(19, generator=g) * 2 - 1
        with torch.no_grad():
            logits, _ = net(obs.unsqueeze(0))
        assert int(logits.argmax()) == ref.act(obs.tolist())
