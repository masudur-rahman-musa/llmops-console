import type { Reason } from "@/lib/console-api";

/** What a page renders instead of its data when console-api could not answer (decision D). */
export function Unavailable({ title, reason }: { title: string; reason: Reason }) {
  return (
    <>
      <h1>{title}</h1>
      <p role="alert" className="alert">Console data unavailable ({reason}).</p>
    </>
  );
}
