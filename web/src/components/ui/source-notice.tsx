import type { Sources } from "@/lib/shapes";

const NAMES = { postgres: "Postgres", prometheus: "Prometheus" } as const;

/** console-api's per-source flags as one line each; `ok` shows nothing. */
export function SourceNotice({ sources }: { sources: Sources }) {
  const lines = (Object.keys(NAMES) as Array<keyof typeof NAMES>).flatMap((source) => {
    const state = sources[source];
    if (state === "error") return [`${NAMES[source]} unavailable.`];
    if (state === "unconfigured") return [`${NAMES[source]} not configured.`];
    return [];
  });
  return (
    <>
      {lines.map((line) => (
        <p key={line} role="status" className="note">
          {line}
        </p>
      ))}
    </>
  );
}
