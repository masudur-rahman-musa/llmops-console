import Link from "next/link";
import type { ReactElement } from "react";

import { Panel } from "@/components/ui/panel";
import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { getPlay } from "@/lib/console-api";

export const dynamic = "force-dynamic";

type Params = { run: string; case: string; play: string };

const TITLE = "Eval play";

/**
 * One play of a stored eval run, turn by turn: what the simulated person said, what the
 * help desk answered and which tools it called. Part of the Quality tab, reached from a
 * Needs a look row. Eval plays hold made-up case data only. Pinned by tests/play-page.test.tsx.
 */
export default async function PlayPage({ params }: { params: Promise<Params> }): Promise<ReactElement> {
  const { run, case: caseId, play } = await params;
  const result = await getPlay(run, caseId, play);
  if (!result.ok) {
    return <Unavailable title={TITLE} reason={result.reason} />;
  }
  const found = result.body.play;
  return (
    <>
      <h1>{TITLE}</h1>
      <p className="question">What did the help desk say in this play, turn by turn?</p>
      <SourceNotice sources={result.body.sources} />
      <p>
        <Link href="/quality">Back to Quality</Link>
      </p>
      {found === null ? (
        <p className="note">This play is not in the store.</p>
      ) : (
        <>
          <p className="window">
            Case <code>{found.case_id}</code>, play {found.play}, run <code>{found.run_id}</code>.
          </p>
          {found.turns.map((turn, index) => (
            <Panel key={index} title={`Turn ${index + 1}`}>
              <p>
                <strong>Person:</strong> {turn.message}
              </p>
              <p>
                <strong>Help desk:</strong> {turn.reply}
              </p>
              <p className="note">
                Tools called:{" "}
                {turn.tools.length === 0
                  ? "none"
                  : turn.tools.map((tool, at) => (
                      <span key={at}>
                        {at > 0 && ", "}
                        <code>{tool}</code>
                      </span>
                    ))}
              </p>
            </Panel>
          ))}
        </>
      )}
    </>
  );
}
