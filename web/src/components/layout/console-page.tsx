import type { ReactElement, ReactNode } from "react";

import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { getConsolePage, type PageBodies, type PageName } from "@/lib/console-api";

/**
 * The opening every console page shares: read the page body, show the
 * Unavailable notice when it fails, else print the `<h1>`, the one-line
 * question the page answers, and the SourceNotice once when the body has
 * sources (the models body reads the live policy and has none). The page passes only
 * its body builder, which may be async so the cost page can still read the
 * tradeoff file. Pinned by tests/shared-parts.test.tsx and tests/clarity.test.tsx.
 */
export function consolePage<N extends PageName>(
  name: N,
  title: string,
  question: string,
  render: (body: PageBodies[N]) => ReactNode | Promise<ReactNode>,
): () => Promise<ReactElement> {
  return async function Page(): Promise<ReactElement> {
    const result = await getConsolePage(name);
    if (!result.ok) {
      return <Unavailable title={title} reason={result.reason} />;
    }
    return (
      <>
        <h1>{title}</h1>
        <p className="question">{question}</p>
        {"sources" in result.body && <SourceNotice sources={result.body.sources} />}
        {await render(result.body)}
      </>
    );
  };
}
