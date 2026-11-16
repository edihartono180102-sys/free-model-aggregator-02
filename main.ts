// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// The entry is an aggregator: a client points at it, and every completion it answers comes from a
// live free OpenRouter model it dials through the grader's gateway on the trial's key. It serves
//   POST /v1/chat/completions  (OpenAI Chat Completions, JSON or server-sent events)
//   POST /v1/messages         (Anthropic Messages, JSON or server-sent events)
//   GET  /v1/models           (the OpenAI list for Authorization: Bearer, the Anthropic list for
//                              x-api-key)
// and it registers its port and catalog with the env, then polls GET /work and exits 0 when done.
//
// ROUTING (quality is 60% of the score): every request is served by the best free model in the
// brief that meets its requirements, ranked by the brief's rating_milli (ties by id, so the order
// is a pure function of the brief). For the fourteen in the brief that is
// thinkingmachines/inkling-2-small (1475) for plain and tools prompts, and
// nvidia/nemotron-3.5-super-vl (1303) for JSON-mode prompts.
//
// FAILOVER (10% of the score): a gateway call answered 4xx/5xx is not an answer. The server
// retries a 5xx once on the same model, then fails over down the quality-ordered list to the next
// free model that meets the request. Every prompt starts again from the best model, so a
// transient refusal costs one prompt at most. A gateway
// answer that only calls tools, with no text carrying the prompt's token, is not an answer
// either, and fails over the same way. Every refusal is reported on the served response in the
// header `x-makefun-upstream-failures` (at most 8 rows of {"status", "code", "model"}), the only
// way the failover column can see a refusal the server retried and then served. A prompt no
// model can serve is answered 502 in the client's wire, with the same trail header.
//
// DETERMINISM (the replay must leave the same state export, byte for byte): every choice is a
// function of the brief, the requests and the gateway's answers — no wall clock, no unseeded
// randomness. The fixed wait below never enters any request, and the model lists carry fixed
// timestamps (see lib/wire.ts).
//
// THE MODEL FIELD: whatever the response reports in `model` is the model actually dialled. The
// provider's own answer spells it without the gateway's `openrouter/` prefix; the id dialled is
// put back on every answer and every streamed chunk.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  candidatesFor,
  freeModels,
  isRecord,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses a call with no output cap, or one over this. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap sent upstream when the client named none. */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** The most gateway attempts spent on one prompt (well under the trial's 60-call cap). */
const MAX_ATTEMPTS_PER_PROMPT = 6;
/** The fixed wait before a same-model retry of a 5xx (never enters a request). */
const RETRY_WAIT_MS = 500;
/** The most rows the upstream-failures trail header may carry. */
const MAX_TRAIL_ROWS = 8;
/** The header the failover column reads the trail from. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** One refusal, as the trail header reports it. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header's value for a prompt's refusals: at most 8 rows, in the order they refused. */
function trailHeader(failures: Failure[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS));
}

/** A fixed wait that never enters a request (used before a same-model 5xx retry). */
function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: Failure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  const trail = trailHeader(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: Failure[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * The upstream Chat Completions body for one dial: the client's body with the dialled model, an
 * output cap the gateway accepts (the gateway refuses a call with none, or one over 8192), and,
 * for a stream, the usage-carrying final chunk (without it the stream carries no usage at all).
 */
function upstreamBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const cap = typeof out.max_tokens === 'number' ? out.max_tokens : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : null;
  if (cap === null || cap <= 0) out.max_tokens = DEFAULT_OUTPUT_TOKENS;
  else if (cap > MAX_OUTPUT_TOKENS) out.max_tokens = MAX_OUTPUT_TOKENS;
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response | null> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  try {
    return await fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify(upstreamBody(body, model)) });
  } catch {
    return null;
  }
}

/** The error code of a refused gateway answer, as the provider sent it. */
async function failureOf(model: string, upstream: Response): Promise<Failure> {
  let code = `http_${upstream.status}`;
  try {
    const text = await upstream.text();
    if (text !== '') {
      const value: unknown = JSON.parse(text);
      if (isRecord(value) && isRecord(value.error)) {
        if (typeof value.error.code === 'string' && value.error.code !== '') code = value.error.code;
        else if (typeof value.error.type === 'string' && value.error.type !== '') code = value.error.type;
      }
    }
  } catch {
    // Not JSON: keep the HTTP status as the code.
  }
  return { status: upstream.status, code: code.slice(0, 64), model };
}

/** A token-shaped string (WORD-1234) anywhere in the request's text, or null. */
function requestToken(body: Json): string | null {
  const seen: string[] = [];
  const walk = (value: unknown): void => {
    if (typeof value === 'string') {
      for (const match of value.matchAll(/\b[A-Z]{3,12}-\d{4}\b/g)) seen.push(match[0]);
    } else if (Array.isArray(value)) {
      for (const item of value) walk(item);
    } else if (isRecord(value)) {
      for (const item of Object.values(value)) walk(item);
    }
  };
  walk(body);
  return seen.length > 0 ? seen[0] : null;
}

/** The text of a Chat Completions answer's first choice, '' when it has none. */
function answerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('\n');
  }
  return '';
}

/**
 * Whether a successful gateway answer actually answered the prompt: a tool_calls-only answer
 * whose text carries none of the request's tokens did not. Such a call is a refusal, not an
 * answer — the server fails over to the next model instead of returning an empty answer.
 */
function answeredPrompt(answer: Json, token: string | null): boolean {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  if (calls.length === 0) return true;
  if (token === null) return true;
  return answerText(answer).includes(token);
}

/** A prompt served by one model, with the refusals it took to get there. */
interface Served {
  model: ModelRow;
  upstream: Response;
  /** The parsed non-stream answer, when the request was not a stream. */
  answer: Json | null;
  failures: Failure[];
}

/**
 * Dial the prompt on the gateway until one model serves it. Candidates run best-first; a 5xx is
 * retried once on the same model, any other refusal moves to the next candidate, and a model
 * that refused is cooled down for the rest of the trial. A non-stream answer that only calls
 * tools, with no token in its text, fails over the same way. Streaming answers are returned as
 * they come — they cannot be judged before they are sent. When every candidate refused, the
 * result carries null with the trail, so the 502 answer can report the refusals too.
 */
async function servePrompt(body: Json, requires: Requirement, token: string | null): Promise<{ served: Served | null; failures: Failure[] }> {
  const failures: Failure[] = [];
  const candidates = candidatesFor(brief, requires);
  let attempts = 0;
  for (const model of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    attempts += 1;
    let upstream = await dial(model.id, body);
    if (upstream === null) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      continue;
    }
    if (!upstream.ok) {
      failures.push(await failureOf(model.id, upstream));
      if (upstream.status >= 500 && attempts < MAX_ATTEMPTS_PER_PROMPT) {
        // One same-model retry of a 5xx: it may be transient.
        attempts += 1;
        await wait(RETRY_WAIT_MS);
        const retry = await dial(model.id, body);
        if (retry !== null && retry.ok) {
          upstream = retry;
        } else {
          if (retry !== null) failures.push(await failureOf(model.id, retry));
          else failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
              continue;
        }
      } else {
          continue;
      }
    }
    if (body.stream === true) return { served: { model, upstream, answer: null, failures }, failures };
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      failures.push({ status: 200, code: 'bad_upstream_json', model: model.id });
      continue;
    }
    const record = isRecord(answer) ? answer : {};
    if (!answeredPrompt(record, token)) {
      failures.push({ status: 200, code: 'tool_call_no_answer', model: model.id });
      continue;
    }
    return { served: { model, upstream, answer: record, failures }, failures };
  }
  return { served: null, failures };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const token = requestToken(body);
  const { served, failures } = await servePrompt(body, requires, token);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model could serve this prompt', failures);
  }
  if (body.stream === true) {
    const trail = trailHeader(served.failures);
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail !== null) headers[FAILURES_HEADER] = trail;
    res.writeHead(200, headers);
    try {
      for await (const frame of openAIStreamWithModel(served.upstream.body as ReadableStream<Uint8Array>, served.model.id)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(served.answer ?? {}, served.model.id), served.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const token = requestToken(body);
  const upstream = anthropicToOpenAI(body, 'auto');
  const { served, failures } = await servePrompt(upstream, requires, token);
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model could serve this prompt', failures);
  }
  if (body.stream === true) {
    const trail = trailHeader(served.failures);
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail !== null) headers[FAILURES_HEADER] = trail;
    res.writeHead(200, headers);
    try {
      for await (const event of anthropicEventsFromOpenAI(served.upstream.body as ReadableStream<Uint8Array>, served.model.id)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.answer ?? {}, served.model.id), served.failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
