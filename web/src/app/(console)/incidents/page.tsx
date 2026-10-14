import { ExternalLink } from "@/components/ui/external-link";
import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { getConsolePage } from "@/lib/console-api";
import { utcTime } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function IncidentsPage() {
  const result = await getConsolePage("incidents");
  if (!result.ok) {
    return <Unavailable title="Incidents" reason={result.reason} />;
  }
  const body = result.body;
  return (
    <>
      <h1>Incidents</h1>
      <SourceNotice sources={body.sources} />
      <ul>
        {body.incidents.map((incident) => (
          <li key={incident.id}>
            <h2>
              {incident.open && <strong>Open</strong>} {incident.severity} {incident.title}
            </h2>
            <p>{incident.summary}</p>
            <p>
              Started {utcTime(incident.started_at)}, resolved {utcTime(incident.resolved_at)}
            </p>
            <ExternalLink url={incident.postmortem_url} label="Postmortem" />
          </li>
        ))}
      </ul>
    </>
  );
}
