// Connectivity check for the judge panel — pings each judge with a trivial prompt.
// Validates that .env keys + model ids work before running the scored eval.
import { JUDGES, callJudge } from './judges.mjs';

for (const judge of JUDGES) {
  try {
    const reply = await callJudge(judge, { system: 'Reply with exactly: OK', user: 'ping' }, { retries: 1 });
    const ok = /ok/i.test(reply || '');
    console.log(`${ok ? '✅' : '⚠️ '} ${judge.id.padEnd(18)} (${judge.model}) -> ${JSON.stringify((reply || '').slice(0, 40))}`);
  } catch (e) {
    console.log(`❌ ${judge.id.padEnd(18)} (${judge.model}) -> ${e.message}`);
  }
}
