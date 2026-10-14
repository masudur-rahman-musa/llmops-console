// The gateway under load: 50 requests a second for ten minutes, offered at a
// constant arrival rate, against the fixed-delay mock provider.
//
// `constant-arrival-rate` is an open model: k6 starts an iteration every 20 ms
// whether or not the last one has answered, so a slow gateway shows up as
// latency and, past the VU pool, as dropped iterations. A closed model would
// quietly offer less traffic instead. `dropped_iterations` is a threshold for
// exactly that reason: a run that did not really offer 50 RPS does not pass.
//
// The mix comes from the global iteration counter, never Math.random(), so two
// runs send the same traffic. Of every 100 iterations: 80 chat, 10 streaming
// chat, 10 embeddings. One chat iteration in five (18 of the 90 chat buckets)
// sends one fixed prompt so the caches take part; every other prompt carries the
// counter and is unique. Pinned by `infra/tests/load/test_k6_script.py`.
import http from 'k6/http';
import exec from 'k6/execution';
import { check } from 'k6';

const BASE_URL = __ENV.GATEWAY_URL;
const REPEATED_PROMPT = 'How do I reset my password?';

// The SLO values: chat p95 under 6 s and 99.5% availability. `kind` tags keep
// embeddings out of the chat latency threshold.
export const options = {
  scenarios: {
    gateway: {
      executor: 'constant-arrival-rate',
      rate: 50,
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 100,
      maxVUs: 300,
    },
  },
  thresholds: {
    'http_req_duration{kind:chat}': ['p(95)<6000'],
    http_req_failed: ['rate<0.005'],
    dropped_iterations: ['count==0'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const HEADERS = {
  Authorization: `Bearer ${__ENV.LOAD_API_KEY}`,
  'X-Tenant': 'load',
  'Content-Type': 'application/json',
};

export default function () {
  const iteration = exec.scenario.iterationInTest;
  const bucket = iteration % 100;

  if (bucket < 80) {
    chat(prompt(bucket, iteration), false);
  } else if (bucket < 90) {
    chat(prompt(bucket, iteration), true);
  } else {
    embed(iteration);
  }
}

function prompt(bucket, iteration) {
  if (bucket % 5 === 0) {
    return REPEATED_PROMPT;
  }
  return `Summarise support ticket number ${iteration} in one sentence.`;
}

function chat(content, stream) {
  const body = JSON.stringify({
    model: 'agent-default',
    messages: [{ role: 'user', content }],
    stream,
  });
  const response = http.post(`${BASE_URL}/v1/chat/completions`, body, {
    headers: HEADERS,
    tags: { kind: 'chat' },
  });
  check(response, { 'chat answered 200': (r) => r.status === 200 });
}

function embed(iteration) {
  const body = JSON.stringify({ model: 'embed', input: [`load embedding input ${iteration}`] });
  const response = http.post(`${BASE_URL}/v1/embeddings`, body, {
    headers: HEADERS,
    tags: { kind: 'embed' },
  });
  check(response, { 'embeddings answered 200': (r) => r.status === 200 });
}

function metric(data, name, stat) {
  const found = data.metrics[name];
  return found && found.values[stat] !== undefined ? found.values[stat] : null;
}

// The whole summary goes to the mounted results directory for the README and the
// workflow artifact; a few headline lines go to stdout, since defining this
// function replaces k6's own end-of-test summary.
export function handleSummary(data) {
  const lines = [
    `iterations/s: ${metric(data, 'iterations', 'rate')}`,
    `dropped iterations: ${metric(data, 'dropped_iterations', 'count') || 0}`,
    `http_req_failed rate: ${metric(data, 'http_req_failed', 'rate')}`,
    `chat p50/p95/p99 ms: ${metric(data, 'http_req_duration{kind:chat}', 'med')} / ` +
      `${metric(data, 'http_req_duration{kind:chat}', 'p(95)')} / ` +
      `${metric(data, 'http_req_duration{kind:chat}', 'p(99)')}`,
  ];
  return {
    '/results/summary.json': JSON.stringify(data, null, 2),
    stdout: `${lines.join('\n')}\n`,
  };
}
