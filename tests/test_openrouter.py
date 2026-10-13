# Part of qqq-microstructure.
#
# The hosted engine (decide_local --engine hosted): Perplexity Decider and
# Jev through OpenRouter's Decisions API, against a stand-in for it. The
# request is Jev's shape with the pinned model and the key from the
# environment; the record logs the dated build and the provider; an answer
# from any other model is refused; one decider failing never costs the
# others' rows; there is no hosted night without a key.

import os, sys, io, json, datetime as dt
import urllib.error
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import guard as g
import decide_local as dl

NAMES = ['MU', 'STX', 'INTC']
NOW = dt.datetime(2026, 10, 13, 9, 17, tzinfo=g.ET)
ST = {'names': {'MU': {'earnings_tonight': True}, 'STX': {}, 'INTC': {}}}
BUILD = {'perplexity/pplx-decider-v1.1-27b':
         ('perplexity/pplx-decider-v1.1-27b-20261010', 'Perplexity'),
         'typesafe/jev-1.13': ('typesafe/jev-1.13-20260917', 'TypeSafe')}


class Decisions:
    """Stands in for POST https://openrouter.ai/api/alpha/decisions."""

    def __init__(self, p=0.7, answer_as=None, fail=None):
        self.p, self.answer_as, self.fail = p, answer_as or {}, fail
        self.calls = []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data)
        self.calls.append((req.full_url, dict(req.header_items()), body))
        if body['model'] == self.fail:
            raise urllib.error.HTTPError(req.full_url, 502, 'Bad Gateway', {},
                                         io.BytesIO(b'{"error":"upstream"}'))
        model, prov = self.answer_as.get(body['model'], BUILD[body['model']])
        return io.BytesIO(json.dumps({
            'id': 'gen-dec-1', 'model': model, 'provider': prov,
            'answers': {k: {'type': 'noul', 'noul': self.p}
                        for k in body['questions']},
            'usage': {'input_tokens': 700, 'output_tokens': 20,
                      'cost': 1e-5}}).encode())


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('QQQ_DECIDER_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-test')
    monkeypatch.setattr(g.time, 'sleep', lambda s: None)   # no real back-off
    return tmp_path


def run(env, opener, name='g.csv'):
    path = str(env / name)
    done = dl.run('kev-4b-q8_0', tonight=lambda t: ('20261013', NAMES, ST),
                  path=path, now=NOW, opener=opener)
    return done, g.read_log(path)


def test_the_decider_and_jev_are_asked_in_jevs_shape_and_logged_by_build(env):
    o = Decisions()
    done, log = run(env, o)
    assert done == ['rule', 'http', 'jev']
    assert {c[2]['model'] for c in o.calls} == set(dl.HOSTED.values())
    for url, headers, body in o.calls:
        assert url == dl.DECISIONS
        assert headers['Authorization'] == 'Bearer sk-test'
        assert set(body) == {'state', 'model', 'questions'}
        assert set(body['questions']) == {f'hold_{n}' for n in NAMES} | \
            {'hedge_book'}
    models = log.groupby('backend').model.first()
    assert models['http'] == \
        'perplexity/pplx-decider-v1.1-27b-20261010@Perplexity'
    assert models['jev'] == 'typesafe/jev-1.13-20260917@TypeSafe'
    assert models['rule'] == 'rule-earnings-v1'
    assert log.state_sha.nunique() == 1                 # one state, all three
    # a second run the same night asks nothing
    o2 = Decisions()
    assert run(env, o2)[0] == [] and o2.calls == []


@pytest.mark.parametrize('impostor', ['typesafe/jev-2.0-20261101',
                                      'typesafe/jev-1.13-mini-20261101',
                                      'typesafe/jev-1.13x'])
def test_an_answer_from_another_model_is_not_logged(env, impostor):
    o = Decisions(answer_as={'typesafe/jev-1.13': (impostor, 'TypeSafe')})
    done, log = run(env, o)
    assert done == ['rule', 'http']                     # Jev refused
    assert 'jev' not in set(log.backend)


def test_one_hosted_failure_never_costs_the_other_rows(env):
    o = Decisions(fail='perplexity/pplx-decider-v1.1-27b')
    done, log = run(env, o)
    assert done == ['rule', 'jev']
    # retried before giving up: a 502 is a provider hiccup, not a verdict
    assert sum(c[2]['model'] == 'perplexity/pplx-decider-v1.1-27b'
               for c in o.calls) == 4


def test_no_key_no_hosted_night(env, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY')
    with pytest.raises(SystemExit, match='OPENROUTER_API_KEY'):
        run(env, Decisions())


def test_proxy_auth_sends_no_key_at_all(env, monkeypatch):
    """Where a network proxy adds the key (measured in a cloud environment,
    RESULTS 27), --proxy-auth sends requests with no Authorization header."""
    monkeypatch.delenv('OPENROUTER_API_KEY')
    o = Decisions()
    done = dl.run('kev-4b-q8_0', tonight=lambda t: ('20261013', NAMES, ST),
                  path=str(env / 'p.csv'), now=NOW, opener=o,
                  proxy_auth=True)
    assert done == ['rule', 'http', 'jev']
    assert all('Authorization' not in h for _, h, _ in o.calls)


def test_the_record_gate_is_split_three_ways():
    # three records can pass (decider, reference, control): Bonferroni on the
    # one-sided 2.5% of t > 2 at 62 df
    assert g.GATE_T == 2.46
