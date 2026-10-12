# Part of qqq-microstructure.
#
# A local, Jev-compatible decision server: the documented TypeSafe
# POST /v1/systemone API (RESULTS 27), answered by a small open model running
# on your own machine. No API key, no data leaves the box.
#
# Jev itself cannot be self-hosted: TypeSafe serves it only through its hosted
# API (early access, an API key) and publishes no weights. What TypeSafe does
# describe is the method: typed questions answered with calibrated
# probabilities from a pretrained language model. That is what this server
# does, the way the open "SemIf" re-implementation does: it never generates
# text, it READS THE MODEL'S OWN PROBABILITIES --
#   noul    p("Yes") vs p("No") for the next token  -> {"noul": p}
#   choice  the model's likelihood of each option   -> choice, probabilities,
#                                                       confidence
#   score   p over the level numbers 1..k          -> probability-weighted
#                                                       score, legend
# -- so every answer is deterministic and comes with a probability, like Jev's.
#
# Point src/guard.py at it:
#   python src/jev_local.py --model Qwen/Qwen3.5-0.8B --port 8787
#   set DECIDER_URL=http://127.0.0.1:8787/v1/systemone
#   python src/guard.py --decide
#
# Honesty notes:
#   - A 0.8B model is a small decider. Its answers are graded in shadow mode
#     (guard.py) like any other; nothing here trades.
#   - TypeSafe's own notes for jev-1.13 say it is weak at numeric precision
#     and reads dates as text ("keep math in code"); a small local model is no
#     better. guard.py hands it a few rounded numbers, not a spreadsheet.
#   - Only the TypeSafe-documented request and response fields are produced;
#     "model" in the response names the local weights actually used.
#
# Needs: pip install torch --index-url https://download.pytorch.org/whl/cpu
#        pip install transformers

import os, sys, json, argparse, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SYSTEM = ('You are a careful decision model. Read the STATE, then answer the '
          'QUESTION exactly as instructed. Be calibrated: answer according to '
          'what the state supports, not what sounds confident.')


def _fmt(x):
    return x if isinstance(x, str) else json.dumps(x, sort_keys=True)


class Scorer:
    """Probabilities from a causal LM's next-token distribution."""

    def __init__(self, name, threads=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        try:                                     # transformers >= 5
            self.lm = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.float32)
        except TypeError:                        # older releases
            self.lm = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float32)
        self.lm.eval()
        self.lock = threading.Lock()

    def _prompt(self, user, answer_prefix=''):
        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': user}]
        try:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        return text + answer_prefix

    def _next_logprobs(self, text):
        ids = self.tok(text, return_tensors='pt').input_ids
        with self.lock, self.torch.no_grad():
            logits = self.lm(ids).logits[0, -1]
        return self.torch.log_softmax(logits.float(), -1), ids.shape[1]

    def _first_ids(self, words):
        """Token id of each word's first token, with and without a leading
        space -- tokenizers differ in which one an answer starts with."""
        out = {}
        for w in words:
            ids = set()
            for v in (w, ' ' + w):
                t = self.tok.encode(v, add_special_tokens=False)
                if t:
                    ids.add(t[0])
            out[w] = sorted(ids)
        return out

    def _dist(self, lp, ids):
        import math
        raw = {w: math.log(sum(math.exp(float(lp[i])) for i in v))
               for w, v in ids.items()}
        m = max(raw.values())
        z = sum(math.exp(x - m) for x in raw.values())
        return {w: math.exp(x - m) / z for w, x in raw.items()}

    def noul(self, state, instructions, criteria):
        user = (f'STATE:\n{_fmt(state)}\n\nQUESTION:\n{_fmt(instructions)}')
        if criteria:
            user += (f'\n\nAnswer Yes if: {_fmt(criteria.get("true", ""))}'
                     f'\nAnswer No if: {_fmt(criteria.get("false", ""))}')
        user += '\n\nAnswer with exactly one word: Yes or No.'
        lp, n = self._next_logprobs(self._prompt(user))
        return self._dist(lp, self._first_ids(['Yes', 'No']))['Yes'], n

    def choice(self, state, instructions, options):
        keys = list(options)
        letters = [chr(ord('A') + i) for i in range(len(keys))] \
            if len(keys) <= 26 else None
        if letters is None:
            raise ValueError('more than 26 options is not supported locally')
        lines = '\n'.join(f'{l}. {k}' + (f' -- {_fmt(options[k])}'
                                          if options[k] else '')
                          for l, k in zip(letters, keys))
        user = (f'STATE:\n{_fmt(state)}\n\nQUESTION:\n{_fmt(instructions)}'
                f'\n\nOPTIONS:\n{lines}\n\nAnswer with the letter only.')
        lp, n = self._next_logprobs(self._prompt(user))
        d = self._dist(lp, self._first_ids(letters))
        return {k: d[l] for k, l in zip(keys, letters)}, n

    def score(self, state, instructions, levels):
        nums = [str(i + 1) for i in range(len(levels))]
        lines = '\n'.join(f'{i}. {_fmt(l)}' for i, l in zip(nums, levels))
        user = (f'STATE:\n{_fmt(state)}\n\nQUESTION:\n{_fmt(instructions)}'
                f'\n\nLEVELS (1 = first):\n{lines}\n\nAnswer with the level '
                f'number only.')
        lp, n = self._next_logprobs(self._prompt(user))
        d = self._dist(lp, self._first_ids(nums))
        return [d[x] for x in nums], n


def validate(body):
    """The documented required fields; returns an error string or None."""
    if not isinstance(body, dict):
        return 'body must be a JSON object'
    for f in ('state', 'model', 'questions'):
        if f not in body:
            return f'missing required field `{f}`'
    if not isinstance(body['questions'], dict) or not body['questions']:
        return '`questions` must be a non-empty map'
    for k, q in body['questions'].items():
        t = q.get('type')
        if t not in ('noul', 'choice', 'score'):
            return f'questions.{k}.type must be noul, choice or score'
        if 'instructions' not in q:
            return f'questions.{k}.instructions is required'
        c = q.get('criteria')
        if t == 'choice' and (not isinstance(c, dict) or not c or len(c) > 255):
            return f'questions.{k}.criteria must map 1-255 options'
        if t == 'score' and (not isinstance(c, list) or not 2 <= len(c) <= 10):
            return f'questions.{k}.criteria must list 2-10 levels'
    return None


def answer(body, scorer):
    out, tokens = {}, 0
    for k, q in body['questions'].items():
        t, c = q['type'], q.get('criteria')
        if t == 'noul':
            p, n = scorer.noul(body['state'], q['instructions'], c or {})
            out[k] = {'type': 'noul', 'noul': p}
        elif t == 'choice':
            d, n = scorer.choice(body['state'], q['instructions'], c)
            top = max(d, key=d.get)
            out[k] = {'type': 'choice', 'choice': top, 'probabilities': d,
                      'confidence': d[top]}
        else:
            ps, n = scorer.score(body['state'], q['instructions'], c)
            out[k] = {'type': 'score',
                      'score': sum(i * p for i, p in enumerate(ps)),
                      'legend': {str(i): _fmt(l) for i, l in enumerate(c)},
                      'probabilities': {str(i): p for i, p in enumerate(ps)},
                      'confidence': max(ps)}
        tokens += n
    return {'model': f'local:{scorer.name}', 'answers': out,
            'usage': {'input_tokens': tokens, 'output_tokens': 0}}


def make_handler(scorer):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, fmt, *a):
            sys.stderr.write('jev_local: ' + fmt % a + '\n')

        def do_GET(self):
            if self.path.rstrip('/') == '/v1/models':
                return self._send(200, {'data': [{'id': scorer.name}]})
            self._send(404, {'error': 'not found'})

        def do_POST(self):
            if self.path.rstrip('/') != '/v1/systemone':
                return self._send(404, {'error': 'not found'})
            try:
                n = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(n) or b'null')
            except ValueError:
                return self._send(422, {'error': 'body is not valid JSON'})
            err = validate(body)
            if err:
                return self._send(422, {'error': err})
            try:
                return self._send(200, answer(body, scorer))
            except Exception as ex:
                return self._send(500, {'error': f'{type(ex).__name__}: {ex}'})
    return H


def serve(scorer, host='127.0.0.1', port=8787):
    srv = ThreadingHTTPServer((host, port), make_handler(scorer))
    return srv


def main():
    ap = argparse.ArgumentParser(description='Local Jev-compatible '
                                 '/v1/systemone server.')
    ap.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8787)
    ap.add_argument('--threads', type=int, default=None)
    a = ap.parse_args()
    print(f'loading {a.model} ...', flush=True)
    sc = Scorer(a.model, a.threads)
    srv = serve(sc, a.host, a.port)
    print(f'jev_local serving {a.model} on http://{a.host}:{a.port}/v1/systemone'
          f'  (localhost only by default)', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
