import type { ReactElement, ReactNode } from "react";

import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { getConsolePage, type PageBodies, type PageName, type PageQuery } from "@/lib/console-api";
import { PAGE_PARAMS, parsePage, type Params } from "@/lib/paging";

type PageProps = { searchParams?: Promise<Params> };

/** Callable bare in tests; Next reads the last shape, which takes its props. */
type PageComponent = {
  (): Promise<ReactElement>;
  (props: PageProps): Promise<ReactElement>;
};

/**
 * The opening every console page shares: read the page body, show the
 * Unavailable notice when it fails, else print the `<h1>`, the one-line
 * question the page answers, and the SourceNotice once when the body has
 * sources. The page passes only
 * its body builder, which may be async so the cost page can still read the
 * tradeoff file, and which gets the URL params so a panel can read its page.
 * Requests asks console-api for the page in `requests_page`, parsed to a whole
 * number first. Pinned by tests/shared-parts.test.tsx, tests/clarity.test.tsx
 * and tests/paging-pages.test.tsx.
 */
export function consolePage<N extends PageName>(
  name: N,
  title: string,
  question: string,
  render: (body: PageBodies[N], params: Params) => ReactNode | Promise<ReactNode>,
): PageComponent {
  return async function Page(props?: PageProps): Promise<ReactElement> {
    const params = (await props?.searchParams) ?? {};
    const query: PageQuery | undefined =
      name === "requests" ? { page: parsePage(params[PAGE_PARAMS.requests]) } : undefined;
    const result = await getConsolePage(name, {}, query);
    if (!result.ok) {
      return <Unavailable title={title} reason={result.reason} />;
    }
    return (
      <>
        <h1>{title}</h1>
        <p className="question">{question}</p>
        {"sources" in result.body && <SourceNotice sources={result.body.sources} />}
        {await render(result.body, params)}
      </>
    );
  };
}
