# Part of qqq-microstructure.
#
# jev_local.py serves the documented TypeSafe /v1/systemone API from a local
# model. These tests use a stand-in scorer (no model download): they pin the
# HTTP contract -- 422 on what TypeSafe documents as invalid, the documented
# answer shapes for noul / choice / score -- and drive guard.py's real client
# against the real server end to end. A real-model smoke test runs only when
# JEV_LOCAL_MODEL is set (it downloads weights).

import os, sys, json, math, threading, urllib.request, urllib.error
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import jev_local as jl
import guard as g


class Stub:
    name = 'stub-0'

    def noul(self, state, instructions, criteria):
        return (0.25 if 'STX' in str(instructions) else 0.8), 7

    def choice(self, state, instructions, options):
        k = list(options)
        return {x: (0.7 if i == 0 else 0.3 / (len(k) - 1))
                for i, x in enumerate(k)}, 7

    def score(self, state, instructions, levels):
        return [0.1, 0.2, 0.7][:len(levels)], 7


@pytest.fixture
def server():
    srv = jl.serve(Stub(), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_port}'
    srv.shutdown()


def _post(url, body):
    req = urllib.request.Request(url + '/v1/systemone',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as ex:
        return ex.code, json.loads(ex.read())


def test_invalid_requests_get_422_like_the_hosted_api(server):
    for bad in ({}, {'state': 1, 'model': 'm'},
                {'state': 1, 'model': 'm', 'questions': {'q': {'type': 'x',
                 'instructions': 'i'}}},
                {'state': 1, 'model': 'm', 'questions': {'q': {
                    'type': 'score', 'instructions': 'i', 'criteria': ['one']}}}):
        code, body = _post(server, bad)
        assert code == 422 and 'error' in body


def test_the_answer_shapes_are_the_documented_ones(server):
    code, r = _post(server, {'state': {'x': 1}, 'model': 'jev-latest',
        'questions': {
            'a': {'type': 'noul', 'instructions': 'hold STX?',
                  'criteria': {'true': 'y', 'false': 'n'}},
            'b': {'type': 'choice', 'instructions': 'pick',
                  'criteria': {'hold': None, 'skip': 'risky', 'hedge': None}},
            'c': {'type': 'score', 'instructions': 'rate',
                  'criteria': ['low', 'mid', 'high']}}})
    assert code == 200 and r['model'] == 'local:stub-0'
    assert r['answers']['a'] == {'type': 'noul', 'noul': 0.25}
    b = r['answers']['b']
    assert b['choice'] == 'hold' and math.isclose(sum(b['probabilities'].values()), 1)
    c = r['answers']['c']
    assert math.isclose(c['score'], 0 * 0.1 + 1 * 0.2 + 2 * 0.7)
    assert c['legend'] == {'0': 'low', '1': 'mid', '2': 'high'}
    assert r['usage'] == {'input_tokens': 21, 'output_tokens': 0}


def test_guard_talks_to_the_local_server_end_to_end(server):
    p, model, ms = g.ask_http({'names': {}}, ['MU', 'STX'],
                              server + '/v1/systemone')
    assert model == 'local:stub-0'
    assert p == {'MU': 0.8, 'STX': 0.25, g.BOOK: 0.8}


@pytest.mark.skipif(not os.environ.get('JEV_LOCAL_MODEL'),
                    reason='set JEV_LOCAL_MODEL to run a real model')
def test_a_real_model_gives_probabilities():
    sc = jl.Scorer(os.environ['JEV_LOCAL_MODEL'])
    p, _ = sc.noul({'sky': 'blue'}, 'Is the sky in the state blue?',
                   {'true': 'it is blue', 'false': 'it is not blue'})
    q, _ = sc.noul({'sky': 'blue'}, 'Is the sky in the state green?',
                   {'true': 'it is green', 'false': 'it is not green'})
    assert 0 <= p <= 1 and 0 <= q <= 1 and p > q
